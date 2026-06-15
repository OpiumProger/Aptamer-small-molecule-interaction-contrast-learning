#!/usr/bin/env python3
"""
Отдельный прогон генерации для молекул RSAPred с pKd в AptaBench.

Выбирает уникальные SMILES из source=RSAPred + pKd_value, предпочитая
компактные молекулы (ближе к калибровочной зоне RSAPred), и генерирует
non-interacting аптамеры. Результаты пишутся в rsapred_* файлы.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

from Embeddings import aptamer_encode
from GRU import (
    ConditionalGRUDecoder,
    build_generation_target_from_dataframe,
    compute_molecule_baseline_sims,
    generate_ranked_aptamers_for_molecule,
    select_diverse_output_candidates,
    select_top_candidates_for_tools,
)
from Model import MicroContrastiveModel
from decoder import get_negative_cluster_embeddings


DATA_FILE = "aptabench_with_embeddings_v2.csv"
MODEL_CHECKPOINT = "final_micro_model.pth"
DECODER_CHECKPOINT = "best_conditional_decoder.pth"

# Калибровочные мишени RSAPred — всегда включаются, если есть в датасете с pKd.
PRIORITY_RSAPRED_SMILES = [
    "Cn1c(=O)c2[nH]cnc2n(C)c1=O",   # caffeine / theophylline
    "O=c1nc[nH]c2nc[nH]c12",        # xanthine
    "Nc1ccnc(N)n1",                  # 2-aminopyrimidine (pKd~3 neg calib)
]

GEN_CONFIG = {
    "sequence_sim_filter": True,
    "max_latent_sim_for_decode": -0.10,
    "allow_relaxed_fallback": False,
    "n_latent_points": 128,
    "samples_per_latent": 8,
    "latent_jitter_copies": 2,
    "latent_jitter_std": 0.10,
    "diversity_threshold": 0.88,
    "temperature": 1.0,
    "top_k": 8,
    "max_latent_similarity": 0.15,
    "n_keep": 50,
    "output_top_k_for_tools": 3,
    "max_motif_penalty_for_tools": 0.05,
    "min_seq_len": 18,
    "max_seq_len": 50,
    "motif_penalty_weight": 0.15,
    "max_kmer_repeat": 2,
    "motif_kmer_sizes": (5, 6),
    "max_homopolymer_motif": 5,
    "reject_high_motif_repeat": True,
    "max_same_prefix": 2,
    "motif_prefix_len": 8,
}


def heavy_atom_count(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    return int(Descriptors.HeavyAtomCount(mol)) if mol else None


def resolve_smiles_col(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.lower() in {"canonical_smiles", "smiles"}:
            return col
    return "canonical_smiles"


def select_rsapred_pkd_molecules(
    df: pd.DataFrame,
    max_targets: int,
    max_heavy_atoms: int,
    require_positive_baseline: bool,
    priority_smiles: list | None = None,
) -> pd.DataFrame:
    """Уникальные молекулы RSAPred с pKd, отсортированные для адекватного сравнения."""
    smiles_col = resolve_smiles_col(df)
    rs = df[
        df["source"].astype(str).str.contains("RSAPred", case=False, na=False)
        & df["pKd_value"].notna()
    ].copy()
    if rs.empty:
        raise ValueError("No RSAPred rows with pKd_value found in dataset.")

    grouped = (
        rs.groupby(smiles_col, as_index=False)
        .agg(
            pKd_mean=("pKd_value", "mean"),
            pKd_min=("pKd_value", "min"),
            pKd_max=("pKd_value", "max"),
            n_pairs=("pKd_value", "count"),
            n_pos=("label", lambda s: int((s == 1).sum())),
            n_neg=("label", lambda s: int((s == 0).sum())),
        )
    )
    grouped["heavy_atoms"] = grouped[smiles_col].map(heavy_atom_count)
    grouped = grouped.dropna(subset=["heavy_atoms"])
    grouped = grouped[grouped["heavy_atoms"] <= max_heavy_atoms]

    if require_positive_baseline:
        grouped = grouped[grouped["n_pos"] > 0]

    grouped = grouped.sort_values(
        ["heavy_atoms", "pKd_mean", "n_pairs"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    priority_smiles = priority_smiles or []
    picked = []
    used = set()
    for smiles in priority_smiles:
        match = grouped[grouped[smiles_col].astype(str) == smiles]
        if not match.empty:
            picked.append(match.iloc[0])
            used.add(smiles)

    for _, row in grouped.iterrows():
        if row[smiles_col] in used:
            continue
        picked.append(row)
        if max_targets > 0 and len(picked) >= max_targets:
            break

    result = pd.DataFrame(picked).reset_index(drop=True)
    result.insert(0, "molecule_index", np.arange(len(result), dtype=int))
    return result


def load_contrastive_model(path: str, device: torch.device) -> MicroContrastiveModel:
    model = MicroContrastiveModel(
        input_dim_apt=768,
        input_dim_mol=768,
        latent_dim=768,
        projection_dim=768,
    )
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_decoder(path: str, device: torch.device) -> ConditionalGRUDecoder:
    decoder = ConditionalGRUDecoder(
        mol_dim=768,
        latent_dim=768,
        hidden_dim=512,
        max_len=50,
        min_len=20,
        num_layers=2,
        dropout=0.3,
    )
    decoder.load_state_dict(torch.load(path, map_location=device))
    decoder.to(device)
    decoder.eval()
    return decoder


def load_global_negative_pool() -> np.ndarray:
    emb_path = Path("cluster_embeddings_768d.npy")
    labels_path = Path("cluster_labels_768d.npy")
    types_path = Path("cluster_types.npy")
    if not all(p.exists() for p in (emb_path, labels_path, types_path)):
        raise FileNotFoundError(
            "Cluster files not found. Run Contrast Learning.py once to build clusters."
        )
    embeddings = np.load(emb_path)
    labels = np.load(labels_path)
    types = np.load(types_path)
    negative_embeddings, _, _ = get_negative_cluster_embeddings(embeddings, labels, types)
    return negative_embeddings.astype(np.float32)


def build_generation_targets(
    molecule_table: pd.DataFrame,
    df: pd.DataFrame,
    model: MicroContrastiveModel,
    device: torch.device,
    global_negative_latents: np.ndarray,
) -> list:
    smiles_col = resolve_smiles_col(df)
    targets = []

    for _, row in molecule_table.iterrows():
        smiles = str(row[smiles_col])
        built = build_generation_target_from_dataframe(
            df=df,
            target_smiles=smiles,
            contrastive_model=model,
            device=device,
            global_negative_latents=global_negative_latents,
            smiles_col=smiles_col,
        )
        if built is None:
            print(f"[SKIP] cannot build target: {smiles[:70]}")
            continue

        baseline = compute_molecule_baseline_sims(
            model,
            df,
            smiles,
            device,
            smiles_col=smiles_col,
            global_negative_latents=built.get("local_negative_latents"),
        )
        if baseline:
            built.update(baseline)
        else:
            built["contrastive_separation"] = np.nan
            built["positive_mean"] = np.nan
            built["negative_mean"] = np.nan

        built["rsapred_pkd_mean"] = float(row["pKd_mean"])
        built["rsapred_pkd_min"] = float(row["pKd_min"])
        built["rsapred_pkd_max"] = float(row["pKd_max"])
        built["rsapred_n_pairs"] = int(row["n_pairs"])
        built["rsapred_n_pos"] = int(row["n_pos"])
        built["rsapred_n_neg"] = int(row["n_neg"])
        built["heavy_atoms"] = int(row["heavy_atoms"])
        built["molecule_index"] = int(row["molecule_index"])
        targets.append(built)

    return targets


def run_generation(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    prefix = args.output_prefix

    print("=" * 70)
    print("RSAPred pKd GENERATION RUN")
    print("=" * 70)
    print(f"Device: {device}")

    df = pd.read_csv(args.data, low_memory=False)
    molecule_table = select_rsapred_pkd_molecules(
        df,
        max_targets=args.max_targets,
        max_heavy_atoms=args.max_heavy_atoms,
        require_positive_baseline=not args.allow_no_positive,
        priority_smiles=PRIORITY_RSAPRED_SMILES,
    )
    targets_path = f"{prefix}_target_molecules.csv"
    molecule_table.to_csv(targets_path, index=False, encoding="utf-8-sig")
    print(f"\nSelected {len(molecule_table)} RSAPred molecules -> {targets_path}")
    print(molecule_table.to_string(index=False))

    model = load_contrastive_model(args.model_path, device)
    decoder = load_decoder(args.decoder_path, device)
    global_negative_pool = load_global_negative_pool()
    print(f"Global negative pool: {len(global_negative_pool)} latents")

    generation_targets = build_generation_targets(
        molecule_table, df, model, device, global_negative_pool
    )
    if not generation_targets:
        raise RuntimeError("No generation targets could be built.")

    print(f"\nReady targets: {len(generation_targets)}")
    for t in generation_targets:
        sep = t.get("contrastive_separation", float("nan"))
        print(
            f"  #{t['molecule_index']} atoms={t['heavy_atoms']} "
            f"pKd={t['rsapred_pkd_mean']:.2f} sep={sep:.4f} "
            f"smiles={str(t['smiles'])[:65]}"
        )

    print("\nPreloading GENA-LM...")
    aptamer_encode(["ACGTACGTACGTACGTACGTACGT"], batch_size=1, device=str(device))

    pair_rows = []
    top_tool_rows = []
    generation_stats = []
    txt_path = f"{prefix}_generated_pairs.txt"

    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write("=" * 80 + "\n")
        handle.write("RSAPred pKd MOLECULES — GRU GENERATED NON-INTERACTING APTAMERS\n")
        handle.write("=" * 80 + "\n\n")

        for target in generation_targets:
            mol_idx = target["molecule_index"]
            mol_emb = target["mol_embedding"]
            mol_smiles = target["smiles"]
            local_neg = target["local_negative_latents"]

            handle.write(f"\n{'=' * 70}\n")
            handle.write(f"MOLECULE #{mol_idx}\n")
            handle.write(f"SMILES: {mol_smiles}\n")
            handle.write(f"RSAPred pKd mean: {target['rsapred_pkd_mean']:.3f}\n")
            handle.write(f"heavy_atoms: {target['heavy_atoms']}\n")
            handle.write(f"{'=' * 70}\n")

            print(f"\nMolecule #{mol_idx} pKd={target['rsapred_pkd_mean']:.2f}: {mol_smiles[:70]}...")

            candidates = generate_ranked_aptamers_for_molecule(
                decoder=decoder,
                mol_embedding=mol_emb,
                local_negative_points=local_neg,
                global_negative_points=global_negative_pool,
                device=device,
                n_latent_points=GEN_CONFIG["n_latent_points"],
                samples_per_latent=GEN_CONFIG["samples_per_latent"],
                temperature=GEN_CONFIG["temperature"],
                top_k=GEN_CONFIG["top_k"],
                max_similarity=GEN_CONFIG["max_latent_similarity"],
                diversity_threshold=GEN_CONFIG["diversity_threshold"],
                latent_jitter_copies=GEN_CONFIG["latent_jitter_copies"],
                latent_jitter_std=GEN_CONFIG["latent_jitter_std"],
                min_seq_len=GEN_CONFIG["min_seq_len"],
                max_seq_len=GEN_CONFIG["max_seq_len"],
                contrastive_model=model,
                aptamer_encode_fn=aptamer_encode,
                raw_smi_embedding=target.get("raw_smi_embedding"),
                baseline_positive_mean=target.get("positive_mean"),
                use_sequence_sim_filter=GEN_CONFIG["sequence_sim_filter"],
                max_latent_sim_for_decode=GEN_CONFIG["max_latent_sim_for_decode"],
                motif_penalty_weight=GEN_CONFIG["motif_penalty_weight"],
                max_kmer_repeat=GEN_CONFIG["max_kmer_repeat"],
                motif_kmer_sizes=GEN_CONFIG["motif_kmer_sizes"],
                max_homopolymer_motif=GEN_CONFIG["max_homopolymer_motif"],
                reject_high_motif_repeat=GEN_CONFIG["reject_high_motif_repeat"],
            )

            strict_candidates = [
                c for c in candidates
                if c.get("latent_sim", c.get("sequence_sim", 1.0)) <= GEN_CONFIG["max_latent_sim_for_decode"]
            ]
            output_candidates = select_diverse_output_candidates(
                strict_candidates if strict_candidates else candidates,
                n_keep=GEN_CONFIG["n_keep"],
                max_same_prefix=GEN_CONFIG["max_same_prefix"],
                prefix_len=GEN_CONFIG["motif_prefix_len"],
            )

            decoded_sims = [
                c.get("latent_sim", c.get("sequence_sim"))
                for c in output_candidates
                if c.get("latent_sim") is not None or c.get("sequence_sim") is not None
            ]
            if decoded_sims:
                generation_stats.append({
                    "molecule_index": mol_idx,
                    "contrastive_separation": target.get("contrastive_separation"),
                    "rsapred_pkd_mean": target["rsapred_pkd_mean"],
                    "heavy_atoms": target["heavy_atoms"],
                    "sequence_sim_min": float(min(decoded_sims)),
                    "sequence_sim_mean": float(np.mean(decoded_sims)),
                    "sequence_sim_max": float(max(decoded_sims)),
                    "n_saved": len(output_candidates),
                    "n_candidates": len(candidates),
                })

            tool_candidates = select_top_candidates_for_tools(
                strict_candidates if strict_candidates else candidates,
                top_k=GEN_CONFIG["output_top_k_for_tools"],
                max_motif_penalty=GEN_CONFIG["max_motif_penalty_for_tools"],
            )
            handle.write(f"\n  TOP {GEN_CONFIG['output_top_k_for_tools']} FOR RSAPred:\n")
            for tool_rank, tool_cand in enumerate(tool_candidates, start=1):
                tool_decoded = tool_cand.get("latent_sim", tool_cand.get("sequence_sim"))
                decoded_val = float(tool_decoded) if tool_decoded is not None else float("nan")
                handle.write(
                    f"    #{tool_rank} DNA={tool_cand['sequence']} | "
                    f"RNA={tool_cand['sequence'].replace('T', 'U')} | "
                    f"decoded_sim={decoded_val:.4f} | "
                    f"motif_penalty={tool_cand.get('motif_penalty', 0.0):.3f}\n"
                )
                top_tool_rows.append({
                    "pair_id": f"mol{mol_idx}_diverse_{tool_rank}",
                    "molecule_index": mol_idx,
                    "tool_rank": tool_rank,
                    "canonical_smiles": mol_smiles,
                    "rsapred_pkd_mean": target["rsapred_pkd_mean"],
                    "heavy_atoms": target["heavy_atoms"],
                    "contrastive_separation": target.get("contrastive_separation"),
                    "positive_mean": target.get("positive_mean"),
                    "negative_mean": target.get("negative_mean"),
                    "dna_sequence": tool_cand["sequence"],
                    "rna_sequence": tool_cand["sequence"].replace("T", "U"),
                    "decoded_sim": decoded_val,
                    "motif_penalty": tool_cand.get("motif_penalty", 0.0),
                    "composite_score": tool_cand.get("composite_score"),
                    "seed_sim": tool_cand.get("latent_similarity"),
                })

            for i, candidate in enumerate(output_candidates):
                seq = candidate["sequence"]
                decoded_sim = candidate.get("latent_sim", candidate.get("sequence_sim"))
                handle.write(f"\n  Aptamer #{i + 1}:\n  {seq}\n")
                handle.write(
                    f"  seed_sim={candidate['latent_similarity']:.4f}, "
                    f"decoded_sim={decoded_sim:.4f}, "
                    f"motif_penalty={candidate.get('motif_penalty', 0.0):.3f}\n"
                )
                pair_rows.append({
                    "molecule_index": mol_idx,
                    "canonical_smiles": mol_smiles,
                    "rsapred_pkd_mean": target["rsapred_pkd_mean"],
                    "heavy_atoms": target["heavy_atoms"],
                    "contrastive_separation": target.get("contrastive_separation"),
                    "positive_mean": target.get("positive_mean"),
                    "negative_mean": target.get("negative_mean"),
                    "aptamer_rank": i + 1,
                    "dna_sequence": seq,
                    "rna_sequence": seq.replace("T", "U"),
                    "seed_sim": candidate["latent_similarity"],
                    "decoded_sim": decoded_sim,
                    "motif_penalty": candidate.get("motif_penalty", 0.0),
                    "length": candidate["length"],
                    "gc_content": candidate["gc"],
                })

    summary_path = f"{prefix}_generation_summary.csv"
    tools_path = f"{prefix}_top_candidates_for_tools.csv"
    pd.DataFrame(pair_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(top_tool_rows).to_csv(tools_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(f"Saved: {txt_path}")
    print(f"Saved: {summary_path} ({len(pair_rows)} rows)")
    print(f"Saved: {tools_path} ({len(top_tool_rows)} rows)")
    if generation_stats:
        print("\nDecoded_sim by molecule:")
        for row in generation_stats:
            print(
                f"  #{row['molecule_index']} pKd={row['rsapred_pkd_mean']:.2f} "
                f"atoms={row['heavy_atoms']} "
                f"decoded min/mean/max="
                f"{row['sequence_sim_min']:.4f}/"
                f"{row['sequence_sim_mean']:.4f}/"
                f"{row['sequence_sim_max']:.4f}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate aptamers for RSAPred pKd molecules.")
    parser.add_argument("--data", default=DATA_FILE)
    parser.add_argument("--model-path", default=MODEL_CHECKPOINT)
    parser.add_argument("--decoder-path", default=DECODER_CHECKPOINT)
    parser.add_argument("--max-targets", type=int, default=20)
    parser.add_argument("--max-heavy-atoms", type=int, default=40)
    parser.add_argument(
        "--allow-no-positive",
        action="store_true",
        help="Also include RSAPred molecules without label=1 pairs in CSV.",
    )
    parser.add_argument("--output-prefix", default="rsapred")
    parser.add_argument("--device", default=None)
    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    run_generation(cli_args)
