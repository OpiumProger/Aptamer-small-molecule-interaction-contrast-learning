#!/usr/bin/env python3
"""
Contrastive rerank for generated aptamers from generated_pairs_molecule_aptamer.txt.

For one generated molecule block:
  1. Parse generated aptamer sequences.
  2. Encode sequences with the same GENA-LM encoder used for seq_emb_*.
  3. Load the trained MicroContrastiveModel.
  4. Compute sequence_sim = cosine(encode_molecule(SMILES), encode_aptamer(sequence)).
  5. Save a ranked CSV.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

# Windows/conda can load both libomp.dll and libiomp5md.dll via torch,
# numpy, transformers, or sklearn dependencies. This keeps the rerank
# utility from aborting after embeddings are computed.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from Embeddings import aptamer_encode, smiles_encode
from Model import MicroContrastiveModel
from load_data_and_visual_data import load_data
from DataPrepare import FinalContrastiveDataset, molecule_disjoint_split
from GRU import compute_motif_penalty


DATA_FILE = "aptabench_with_embeddings_v2.csv"


def parse_generated_file(path: str) -> List[Dict]:
    """Parse molecule blocks and generated aptamers from the text output file."""
    blocks = []
    current = None
    pending_aptamer_number = None

    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]

    for line in lines:
        molecule_match = re.match(r"MOLECULE #(\d+)", line.strip())
        if molecule_match:
            if current is not None:
                blocks.append(current)
            current = {
                "molecule_index": int(molecule_match.group(1)),
                "smiles": None,
                "aptamers": [],
            }
            pending_aptamer_number = None
            continue

        if current is None:
            continue

        if line.startswith("SMILES:"):
            current["smiles"] = line.split("SMILES:", 1)[1].strip()
            continue

        aptamer_match = re.match(r"\s*Aptamer #(\d+):", line)
        if aptamer_match:
            pending_aptamer_number = int(aptamer_match.group(1))
            continue

        if pending_aptamer_number is not None:
            seq = line.strip().upper()
            if seq and re.fullmatch(r"[ACGTU]+", seq):
                current["aptamers"].append(
                    {
                        "aptamer_number": pending_aptamer_number,
                        "sequence": seq.replace("U", "T"),
                    }
                )
                pending_aptamer_number = None
                continue

        if current["aptamers"] and (
            "seed_sim=" in line
            or "decoded_sim=" in line
            or "latent_similarity=" in line
        ):
            apt = current["aptamers"][-1]
            for field, pattern in (
                ("seed_sim", r"seed_sim=([-+]?\d*\.?\d+)"),
                ("decoded_sim", r"decoded_sim=([-+]?\d*\.?\d+)"),
                ("latent_similarity", r"latent_similarity=([-+]?\d*\.?\d+)"),
                ("motif_penalty", r"motif_penalty=([-+]?\d*\.?\d+)"),
                ("length", r"length=(\d+)"),
                ("gc", r"GC=([-+]?\d*\.?\d+)"),
            ):
                match = re.search(pattern, line)
                if match:
                    apt[field] = float(match.group(1)) if field in {"seed_sim", "decoded_sim", "latent_similarity", "motif_penalty", "gc"} else int(match.group(1))

            match = re.search(r"relaxed=(True|False)", line)
            if match:
                apt["relaxed"] = match.group(1) == "True"

    if current is not None:
        blocks.append(current)

    return blocks


def apply_ranking(result: pd.DataFrame, rank_by: str, motif_penalty_weight: float) -> pd.DataFrame:
    """Add composite_score and rank columns."""
    result = result.copy()
    motif_vals = []
    for _, row in result.iterrows():
        penalty = row.get("motif_penalty")
        if pd.notna(penalty):
            motif_vals.append(float(penalty))
        else:
            p, _, _ = compute_motif_penalty(str(row["sequence"]))
            motif_vals.append(float(p))
    result["motif_penalty"] = motif_vals
    result["composite_score"] = result["sequence_sim"] + motif_penalty_weight * result["motif_penalty"]

    result["rank_by_sequence_sim"] = (
        result.groupby("molecule_index")["sequence_sim"].rank(method="first", ascending=True).astype(int)
        if "molecule_index" in result.columns and result["molecule_index"].nunique() > 1
        else pd.Series(np.arange(1, len(result) + 1), index=result.index)
    )
    result["rank_by_composite"] = (
        result.groupby("molecule_index")["composite_score"].rank(method="first", ascending=True).astype(int)
        if "molecule_index" in result.columns and result["molecule_index"].nunique() > 1
        else pd.Series(np.arange(1, len(result) + 1), index=result.index)
    )

    rank_col = "rank_by_composite" if rank_by == "composite" else "rank_by_sequence_sim"
    result = result.sort_values(
        ["molecule_index", rank_col] if "molecule_index" in result.columns else [rank_col]
    ).reset_index(drop=True)
    result.insert(0, "rank", result[rank_col].astype(int))
    return result


def add_diverse_ranking(result: pd.DataFrame, max_motif_penalty: Optional[float]) -> pd.DataFrame:
    """Rank non-template candidates by sequence_sim within each molecule."""
    result = result.copy()
    if max_motif_penalty is None:
        result["passes_motif_filter"] = True
        result["rank_diverse"] = np.nan
        return result

    result["passes_motif_filter"] = result["motif_penalty"] <= max_motif_penalty
    diverse = result[result["passes_motif_filter"]].copy()
    if diverse.empty:
        result["rank_diverse"] = np.nan
        return result

    diverse["rank_diverse"] = (
        diverse.groupby("molecule_index")["sequence_sim"].rank(method="first", ascending=True).astype(int)
        if "molecule_index" in diverse.columns and diverse["molecule_index"].nunique() > 1
        else pd.Series(np.arange(1, len(diverse) + 1), index=diverse.index)
    )
    result["rank_diverse"] = np.nan
    result.loc[diverse.index, "rank_diverse"] = diverse["rank_diverse"]
    return result


def print_diverse_top(result: pd.DataFrame, max_motif_penalty: float, print_top: int) -> None:
    diverse = result[result["passes_motif_filter"]].copy()
    if diverse.empty:
        print(f"\nNo candidates with motif_penalty <= {max_motif_penalty}")
        return

    diverse = diverse.sort_values(
        ["molecule_index", "sequence_sim"] if "molecule_index" in diverse.columns else ["sequence_sim"]
    )
    print(
        f"\nTop diverse candidates (motif_penalty <= {max_motif_penalty}, "
        f"ranked by sequence_sim): {len(diverse)} pass"
    )
    cols = ["molecule_index", "rank_diverse", "sequence_sim", "motif_penalty", "sequence"]
    cols = [c for c in cols if c in diverse.columns]
    print(diverse[cols].head(print_top).to_string(index=False))


def load_pairs_from_summary(path: str) -> List[Dict]:
    """Load molecule/aptamer blocks from generation_summary.csv."""
    df = pd.read_csv(path)
    required = {"molecule_index", "canonical_smiles", "dna_sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"generation_summary.csv missing columns: {sorted(missing)}")

    blocks = []
    for mol_idx, group in df.groupby("molecule_index", sort=True):
        smiles = str(group.iloc[0]["canonical_smiles"])
        aptamers = []
        for _, row in group.iterrows():
            aptamers.append(
                {
                    "aptamer_number": int(row.get("aptamer_rank", len(aptamers) + 1)),
                    "sequence": str(row["dna_sequence"]).upper().replace("U", "T"),
                    "seed_sim": row.get("seed_sim"),
                    "decoded_sim": row.get("decoded_sim"),
                    "motif_penalty": row.get("motif_penalty"),
                    "length": row.get("length"),
                    "gc": row.get("gc_content", row.get("gc")),
                    "relaxed": row.get("relaxed"),
                }
            )
        blocks.append(
            {
                "molecule_index": int(mol_idx),
                "smiles": smiles,
                "contrastive_separation": group.iloc[0].get("contrastive_separation"),
                "aptamers": aptamers,
            }
        )
    return blocks


def select_blocks(
    blocks: List[Dict],
    molecule_index: Optional[int],
    smiles_contains: Optional[str],
    all_molecules: bool,
) -> List[Dict]:
    if all_molecules:
        if not blocks:
            raise ValueError("No molecule blocks found in input.")
        return blocks

    return [select_block(blocks, molecule_index, smiles_contains)]


def select_block(blocks: List[Dict], molecule_index: Optional[int], smiles_contains: Optional[str]) -> Dict:
    if smiles_contains:
        matches = [block for block in blocks if block["smiles"] and smiles_contains in block["smiles"]]
        if not matches:
            raise ValueError(f"No molecule block contains SMILES substring: {smiles_contains}")
        return matches[0]

    if molecule_index is None:
        molecule_index = 0

    for block in blocks:
        if block["molecule_index"] == molecule_index:
            return block

    raise ValueError(f"Molecule #{molecule_index} was not found in generated file.")


def embedding_key(values: np.ndarray, ndigits: int = 6) -> tuple:
    return tuple(np.asarray(values, dtype=np.float32).round(ndigits).tolist())


def build_generation_smiles_map(data_path: str) -> Dict[int, str]:
    """Map MOLECULE #N from generated_pairs file to canonical SMILES (same order as Contrast Learning.py)."""
    apt_pos, smi_pos, apt_neg, smi_neg, _, _, mol_keys_pos, mol_keys_neg = load_data(data_path)
    df = pd.read_csv(data_path, low_memory=False)
    (
        _train_pos_idx, _val_pos_idx, test_pos_idx,
        _train_neg_idx, _val_neg_idx, test_neg_idx,
        _,
    ) = molecule_disjoint_split(df, seed=42)

    test_dataset = FinalContrastiveDataset(
        apt_pos[test_pos_idx],
        smi_pos[test_pos_idx],
        apt_neg[test_neg_idx],
        smi_neg[test_neg_idx],
        mol_keys_pos[test_pos_idx],
        mol_keys_neg[test_neg_idx],
    )

    df = pd.read_csv(data_path, low_memory=False)
    smi_cols = [col for col in df.columns if col.startswith("smi_emb_")]
    smiles_col = next(
        (c for c in df.columns if c.lower() in {"canonical_smiles", "smiles"}),
        "canonical_smiles",
    )
    smi_to_smiles = {
        embedding_key(row[smi_cols].values): str(row[smiles_col])
        for _, row in df.iterrows()
    }

    smiles_map: Dict[int, str] = {}
    mol_counter = 0
    for mol_idx, mol_key in enumerate(test_dataset.smis):
        if mol_keys_pos is not None:
            smiles_map[mol_counter] = str(mol_key)
        else:
            raw_mol = test_dataset.key_to_smi[mol_key]
            smiles_map[mol_counter] = smi_to_smiles.get(
                embedding_key(raw_mol), f"Unknown_SMILES_{mol_idx}"
            )
        mol_counter += 1
    return smiles_map


def resolve_target_smiles(
    block: Dict,
    data_path: str,
    target_smiles: Optional[str],
) -> str:
    if target_smiles:
        return target_smiles.strip()

    smiles = (block.get("smiles") or "").strip()
    if smiles and not smiles.startswith("Unknown_SMILES_"):
        return smiles

    resolved = build_generation_smiles_map(data_path).get(block["molecule_index"])
    if resolved and not resolved.startswith("Unknown_SMILES_"):
        print(
            f"[INFO] Resolved SMILES for molecule #{block['molecule_index']} "
            f"from {data_path}: {resolved[:80]}..."
        )
        return resolved

    raise ValueError(
        f"Could not resolve SMILES for molecule #{block['molecule_index']}. "
        f"Generated file has '{smiles}'. Re-run Contrast Learning.py with v2 CSV "
        "or pass --target-smiles / --smiles-contains."
    )


def molecule_baseline_sequence_sims(
    model: MicroContrastiveModel,
    smiles: str,
    data_path: str,
    device: torch.device,
) -> Dict[str, float]:
    """Contrastive cosine sims for known positive/negative pairs of the target molecule."""
    df = pd.read_csv(data_path, low_memory=False)
    seq_cols = [col for col in df.columns if col.startswith("seq_emb_")]
    smi_cols = [col for col in df.columns if col.startswith("smi_emb_")]
    smiles_col = next(
        (c for c in df.columns if c.lower() in {"canonical_smiles", "smiles"}),
        "canonical_smiles",
    )

    subset = df[df[smiles_col].astype(str) == smiles]
    if subset.empty or not seq_cols or not smi_cols:
        return {}

    smi_embedding = subset.iloc[0][smi_cols].to_numpy(dtype=np.float32)
    pos_rows = subset[subset["label"] == 1]
    neg_rows = subset[subset["label"] == 0]
    if pos_rows.empty or neg_rows.empty:
        return {}

    with torch.no_grad():
        smi_t = torch.tensor(smi_embedding, dtype=torch.float32, device=device).unsqueeze(0)
        smi_z = model.encode_molecule(smi_t)

        pos_t = torch.tensor(pos_rows[seq_cols].values.astype(np.float32), device=device)
        neg_t = torch.tensor(neg_rows[seq_cols].values.astype(np.float32), device=device)
        pos_sims = F.cosine_similarity(smi_z.expand(len(pos_rows), -1), model.encode_aptamer(pos_t), dim=-1)
        neg_sims = F.cosine_similarity(smi_z.expand(len(neg_rows), -1), model.encode_aptamer(neg_t), dim=-1)

    return {
        "positive_mean": float(pos_sims.mean().cpu()),
        "positive_min": float(pos_sims.min().cpu()),
        "negative_mean": float(neg_sims.mean().cpu()),
        "negative_min": float(neg_sims.min().cpu()),
        "negative_max": float(neg_sims.max().cpu()),
    }


def load_target_smiles_embedding(smiles: str, data_path: str) -> np.ndarray:
    """Use stored smi_emb_* when possible, otherwise encode SMILES from scratch."""
    df = pd.read_csv(data_path, low_memory=False)
    smi_cols = [col for col in df.columns if col.startswith("smi_emb_")]
    smiles_col = next((c for c in df.columns if c.lower() in {"canonical_smiles", "smiles"}), None)

    if smi_cols and smiles_col:
        exact = df[df[smiles_col].astype(str) == smiles]
        if not exact.empty:
            return exact.iloc[0][smi_cols].to_numpy(dtype=np.float32)

    print("[WARN] Target SMILES not found in stored embeddings; encoding SMILES from scratch.")
    return smiles_encode([smiles])[0].astype(np.float32)


def load_contrastive_model(model_path: str, device: torch.device) -> MicroContrastiveModel:
    model = MicroContrastiveModel(input_dim_apt=768, input_dim_mol=768, latent_dim=768, projection_dim=768)
    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Run Contrast Learning.py first or pass --model-path."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def nearest_reference_scores(generated_z: np.ndarray, ref_path: str) -> Optional[np.ndarray]:
    path = Path(ref_path)
    if not path.exists():
        return None
    refs = np.load(path).astype(np.float32)
    refs = refs / np.clip(np.linalg.norm(refs, axis=1, keepdims=True), 1e-8, None)
    generated_z = generated_z / np.clip(np.linalg.norm(generated_z, axis=1, keepdims=True), 1e-8, None)
    return generated_z @ refs.T


def vector_diagnostics(name: str, values: np.ndarray) -> None:
    """Print simple variance and pairwise cosine diagnostics."""
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return
    per_dim_std = values.std(axis=0)
    norms = np.linalg.norm(values, axis=1)
    normalized = values / np.clip(norms[:, None], 1e-8, None)
    pairwise = normalized @ normalized.T
    off_diag = pairwise[~np.eye(len(values), dtype=bool)] if len(values) > 1 else np.array([])

    print(f"\n{name} diagnostics:")
    print(f"  shape={values.shape}")
    print(f"  norm mean={norms.mean():.6f}, std={norms.std():.6f}")
    print(f"  per-dim std mean={per_dim_std.mean():.8f}, max={per_dim_std.max():.8f}")
    if len(off_diag):
        print(
            "  pairwise cosine off-diagonal: "
            f"min={off_diag.min():.6f}, mean={off_diag.mean():.6f}, max={off_diag.max():.6f}"
        )


def rerank_block(
    block: Dict,
    args: argparse.Namespace,
    model: MicroContrastiveModel,
    device: torch.device,
    show_diagnostics: bool,
) -> pd.DataFrame:
    aptamers = block["aptamers"]
    if not aptamers:
        raise ValueError(f"Molecule #{block['molecule_index']} has no generated aptamers.")

    smiles = resolve_target_smiles(block, args.data, args.target_smiles)
    sequences = [item["sequence"] for item in aptamers]

    print(f"\nMolecule #{block['molecule_index']}: {smiles[:80]}...")
    print(f"Generated aptamers: {len(sequences)}")

    print("Encoding generated aptamers with GENA-LM...")
    seq_embeddings = aptamer_encode(sequences).astype(np.float32)
    smi_embedding = load_target_smiles_embedding(smiles, args.data).astype(np.float32)
    if show_diagnostics:
        vector_diagnostics("Raw GENA seq embeddings", seq_embeddings)

    with torch.no_grad():
        apt_t = torch.tensor(seq_embeddings, dtype=torch.float32, device=device)
        smi_t = torch.tensor(smi_embedding, dtype=torch.float32, device=device).unsqueeze(0)
        apt_z = model.encode_aptamer(apt_t)
        smi_z = model.encode_molecule(smi_t).expand(apt_z.size(0), -1)
        sequence_sim = F.cosine_similarity(smi_z, apt_z, dim=-1).cpu().numpy()
        apt_z_np = apt_z.cpu().numpy()

    baseline = molecule_baseline_sequence_sims(model, smiles, args.data, device)
    if baseline and show_diagnostics:
        print(
            "\nDataset baseline for this molecule (stored seq_emb, contrastive space):"
            f"\n  positive: mean={baseline['positive_mean']:.4f}, min={baseline['positive_min']:.4f}"
            f"\n  negative: mean={baseline['negative_mean']:.4f}, "
            f"min={baseline['negative_min']:.4f}, max={baseline['negative_max']:.4f}"
        )

    if show_diagnostics:
        vector_diagnostics("Contrastive aptamer embeddings", apt_z_np)
        print(
            "\nsequence_sim diagnostics: "
            f"min={sequence_sim.min():.8f}, mean={sequence_sim.mean():.8f}, "
            f"max={sequence_sim.max():.8f}, std={sequence_sim.std():.8f}"
        )

    pos_mean = baseline.get("positive_mean")
    neg_mean = baseline.get("negative_mean")

    rows = []
    for item, sim in zip(aptamers, sequence_sim):
        passes_absolute = bool(sim <= args.max_sequence_similarity)
        passes_relative = sim < pos_mean if pos_mean is not None else None
        rows.append(
            {
                "molecule_index": block["molecule_index"],
                "smiles": smiles,
                "contrastive_separation": block.get("contrastive_separation"),
                "aptamer_number": item.get("aptamer_number"),
                "sequence": item["sequence"],
                "length": item.get("length", len(item["sequence"])),
                "gc": item.get("gc"),
                "seed_sim": item.get("seed_sim"),
                "decoded_sim": item.get("decoded_sim"),
                "motif_penalty": item.get("motif_penalty"),
                "latent_similarity": item.get("latent_similarity", item.get("seed_sim")),
                "sequence_sim": float(sim),
                "baseline_positive_mean": pos_mean,
                "baseline_negative_mean": neg_mean,
                "delta_from_positive_mean": (
                    float(sim - pos_mean) if pos_mean is not None else None
                ),
                "passes_absolute_filter": passes_absolute,
                "passes_relative_filter": passes_relative,
                "passes_contrastive_filter": passes_relative if passes_relative is not None else passes_absolute,
                "relaxed": item.get("relaxed"),
            }
        )

    result = pd.DataFrame(rows)

    pos_scores = nearest_reference_scores(apt_z_np, args.apt_pos_reference)
    neg_scores = nearest_reference_scores(apt_z_np, args.apt_neg_reference)
    if pos_scores is not None:
        result["nearest_positive_apt_sim"] = pos_scores.max(axis=1)
    if neg_scores is not None:
        result["nearest_negative_apt_sim"] = neg_scores.max(axis=1)
    if pos_scores is not None and neg_scores is not None:
        result["negative_minus_positive_neighbor"] = (
            result["nearest_negative_apt_sim"] - result["nearest_positive_apt_sim"]
        )

    return apply_ranking(result, args.rank_by, args.motif_penalty_weight)


def load_input_blocks(args: argparse.Namespace) -> List[Dict]:
    summary_path = Path(args.summary)
    if args.use_summary or (args.all_molecules and summary_path.exists()):
        print(f"Loading pairs from summary CSV: {summary_path}")
        return load_pairs_from_summary(str(summary_path))
    print(f"Parsing generated file: {args.generated}")
    return parse_generated_file(args.generated)


def rerank_molecules(args: argparse.Namespace) -> pd.DataFrame:
    blocks = load_input_blocks(args)
    selected = select_blocks(
        blocks,
        molecule_index=args.molecule_index,
        smiles_contains=args.smiles_contains,
        all_molecules=args.all_molecules,
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_contrastive_model(args.model_path, device)

    print(f"Reranking {len(selected)} molecule block(s)...")
    results = []
    show_diagnostics = args.diagnostics and len(selected) == 1

    for block in selected:
        results.append(rerank_block(block, args, model, device, show_diagnostics=show_diagnostics))

    result = pd.concat(results, ignore_index=True)
    max_motif_penalty = args.max_motif_penalty if args.max_motif_penalty >= 0 else None
    result = add_diverse_ranking(result, max_motif_penalty)

    output_path = Path(args.output)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved rerank CSV: {output_path} ({len(result)} rows, {len(selected)} molecules)")
    print(
        f"Pass absolute threshold sequence_sim <= {args.max_sequence_similarity}: "
        f"{int(result['passes_absolute_filter'].sum())}/{len(result)}"
    )
    if result["passes_relative_filter"].notna().any():
        print(
            "Pass relative filter sequence_sim < baseline_positive_mean: "
            f"{int(result['passes_relative_filter'].fillna(False).sum())}/{len(result)}"
        )

    if args.all_molecules:
        rank_col = "rank_by_composite" if args.rank_by == "composite" else "rank_by_sequence_sim"
        best = (
            result.sort_values(["molecule_index", rank_col])
            .groupby("molecule_index", as_index=False)
            .first()
        )
        print(f"\nBest candidate per molecule ({len(best)} rows, rank_by={args.rank_by}):")
        print(
            best[["molecule_index", "rank", "sequence_sim", "composite_score", "motif_penalty", "sequence"]]
            .head(args.print_top)
            .to_string(index=False)
        )
    else:
        print(f"\nTop candidates (rank_by={args.rank_by}):")
        print(
            result[["rank", "aptamer_number", "sequence_sim", "composite_score", "motif_penalty", "sequence"]]
            .head(args.print_top)
            .to_string(index=False)
        )
        alt_col = "rank_by_sequence_sim" if args.rank_by == "composite" else "rank_by_composite"
        print(f"\nTop by alternate ranking ({'sequence_sim' if alt_col == 'rank_by_sequence_sim' else 'composite'}):")
        print(
            result.sort_values(alt_col)
            [["rank", alt_col, "sequence_sim", "composite_score", "motif_penalty", "sequence"]]
            .head(min(5, args.print_top))
            .to_string(index=False)
        )

    if args.max_motif_penalty is not None and args.max_motif_penalty >= 0:
        print_diverse_top(result, args.max_motif_penalty, args.print_top)

    return result


def rerank_one_molecule(args: argparse.Namespace) -> pd.DataFrame:
    return rerank_molecules(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rerank generated aptamers with the contrastive model.")
    parser.add_argument("--generated", default="generated_pairs_molecule_aptamer.txt")
    parser.add_argument(
        "--summary",
        default="generation_summary.csv",
        help="CSV from Contrast Learning.py with all molecule/aptamer pairs.",
    )
    parser.add_argument(
        "--use-summary",
        action="store_true",
        help="Read pairs from generation_summary.csv instead of the text file.",
    )
    parser.add_argument(
        "--all-molecules",
        action="store_true",
        help="Rerank all molecules (uses generation_summary.csv if present).",
    )
    parser.add_argument("--data", default=DATA_FILE)
    parser.add_argument("--model-path", default="final_micro_model.pth")
    parser.add_argument("--molecule-index", type=int, default=0)
    parser.add_argument("--smiles-contains", default=None)
    parser.add_argument(
        "--target-smiles",
        default=None,
        help="Override SMILES when generated file contains Unknown_SMILES_* placeholders.",
    )
    parser.add_argument("--max-sequence-similarity", type=float, default=0.15)
    parser.add_argument("--output", default="rerank_molecule_0.csv")
    parser.add_argument("--apt-pos-reference", default="apt_pos_final.npy")
    parser.add_argument("--apt-neg-reference", default="apt_neg_final.npy")
    parser.add_argument("--device", default=None, help="cuda, cpu, or empty for auto")
    parser.add_argument("--print-top", type=int, default=10)
    parser.add_argument(
        "--rank-by",
        choices=("composite", "sequence_sim"),
        default="composite",
        help="composite = sequence_sim + motif_penalty_weight * motif_penalty (recommended).",
    )
    parser.add_argument("--motif-penalty-weight", type=float, default=0.15)
    parser.add_argument(
        "--max-motif-penalty",
        type=float,
        default=0.05,
        help="Also rank candidates with motif_penalty <= this value by sequence_sim. "
        "Use 0.05 to skip CTTACGAC/GGGACGAC templates. Pass a negative value to disable.",
    )
    parser.add_argument("--diagnostics", action="store_true", help="Print embedding variance diagnostics.")
    return parser


if __name__ == "__main__":
    rerank_molecules(build_parser().parse_args())
