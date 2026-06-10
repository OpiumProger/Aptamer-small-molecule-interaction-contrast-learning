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
from DataPrepare import FinalContrastiveDataset


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

        if "latent_similarity=" in line and current["aptamers"]:
            match = re.search(r"latent_similarity=([-+]?\d*\.?\d+)", line)
            if match:
                current["aptamers"][-1]["latent_similarity"] = float(match.group(1))

            match = re.search(r"length=(\d+)", line)
            if match:
                current["aptamers"][-1]["length"] = int(match.group(1))

            match = re.search(r"GC=([-+]?\d*\.?\d+)", line)
            if match:
                current["aptamers"][-1]["gc"] = float(match.group(1))

            match = re.search(r"relaxed=(True|False)", line)
            if match:
                current["aptamers"][-1]["relaxed"] = match.group(1) == "True"

    if current is not None:
        blocks.append(current)

    return blocks


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
    apt_pos, smi_pos, apt_neg, smi_neg, _, _ = load_data(data_path)
    np.random.seed(42)

    n_pos = len(smi_pos)
    pos_indices = np.random.permutation(n_pos)
    train_pos_size = int(0.7 * n_pos)
    val_pos_size = int(0.15 * n_pos)
    test_pos_idx = pos_indices[train_pos_size + val_pos_size :]

    n_neg = len(smi_neg)
    neg_indices = np.random.permutation(n_neg)
    train_neg_size = int(0.7 * n_neg)
    val_neg_size = int(0.15 * n_neg)
    test_neg_idx = neg_indices[train_neg_size + val_neg_size :]

    test_dataset = FinalContrastiveDataset(
        apt_pos[test_pos_idx],
        smi_pos[test_pos_idx],
        apt_neg[test_neg_idx],
        smi_neg[test_neg_idx],
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
    for mol_idx, smi_tuple in enumerate(test_dataset.smis):
        raw_mol = np.asarray(smi_tuple, dtype=np.float32)
        if len(test_dataset.smi_to_neg.get(smi_tuple, [])) == 0:
            continue
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


def rerank_one_molecule(args: argparse.Namespace) -> pd.DataFrame:
    blocks = parse_generated_file(args.generated)
    block = select_block(blocks, args.molecule_index, args.smiles_contains)
    aptamers = block["aptamers"]
    if not aptamers:
        raise ValueError(f"Molecule #{block['molecule_index']} has no generated aptamers.")

    smiles = resolve_target_smiles(block, args.data, args.target_smiles)
    sequences = [item["sequence"] for item in aptamers]

    print(f"Selected molecule #{block['molecule_index']}: {smiles}")
    print(f"Generated aptamers: {len(sequences)}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_contrastive_model(args.model_path, device)

    print("Encoding generated aptamers with GENA-LM...")
    seq_embeddings = aptamer_encode(sequences).astype(np.float32)
    smi_embedding = load_target_smiles_embedding(smiles, args.data).astype(np.float32)
    if args.diagnostics:
        vector_diagnostics("Raw GENA seq embeddings", seq_embeddings)

    with torch.no_grad():
        apt_t = torch.tensor(seq_embeddings, dtype=torch.float32, device=device)
        smi_t = torch.tensor(smi_embedding, dtype=torch.float32, device=device).unsqueeze(0)
        apt_z = model.encode_aptamer(apt_t)
        smi_z = model.encode_molecule(smi_t).expand(apt_z.size(0), -1)
        sequence_sim = F.cosine_similarity(smi_z, apt_z, dim=-1).cpu().numpy()
        apt_z_np = apt_z.cpu().numpy()

    baseline = molecule_baseline_sequence_sims(model, smiles, args.data, device)
    if baseline:
        print(
            "\nDataset baseline for this molecule (stored seq_emb, contrastive space):"
            f"\n  positive: mean={baseline['positive_mean']:.4f}, min={baseline['positive_min']:.4f}"
            f"\n  negative: mean={baseline['negative_mean']:.4f}, "
            f"min={baseline['negative_min']:.4f}, max={baseline['negative_max']:.4f}"
        )
        print(
            "  Note: absolute sequence_sim is often ~0.93-0.96 even for known negatives; "
            "use relative ranking vs positive_mean, not threshold <= 0.15 alone."
        )

    if args.diagnostics:
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
        passes_relative = (
            sim < pos_mean if pos_mean is not None else None
        )
        rows.append(
            {
                "molecule_index": block["molecule_index"],
                "smiles": smiles,
                "aptamer_number": item.get("aptamer_number"),
                "sequence": item["sequence"],
                "length": item.get("length", len(item["sequence"])),
                "gc": item.get("gc"),
                "latent_similarity": item.get("latent_similarity"),
                "sequence_sim": float(sim),
                "baseline_positive_mean": pos_mean,
                "baseline_negative_mean": neg_mean,
                "delta_from_positive_mean": (
                    float(sim - pos_mean) if pos_mean is not None else None
                ),
                "passes_absolute_filter": passes_absolute,
                "passes_relative_filter": passes_relative,
                "passes_contrastive_filter": passes_relative if passes_relative is not None else passes_absolute,
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

    result = result.sort_values("sequence_sim", ascending=True).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))

    output_path = Path(args.output)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved rerank CSV: {output_path}")
    print(
        f"Pass absolute threshold sequence_sim <= {args.max_sequence_similarity}: "
        f"{int(result['passes_absolute_filter'].sum())}/{len(result)}"
    )
    if result["passes_relative_filter"].notna().any():
        print(
            "Pass relative filter sequence_sim < baseline_positive_mean: "
            f"{int(result['passes_relative_filter'].fillna(False).sum())}/{len(result)}"
        )
    print("\nTop candidates:")
    print(result[["rank", "aptamer_number", "sequence_sim", "latent_similarity", "sequence"]].head(args.print_top).to_string(index=False))

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rerank generated aptamers with the contrastive model.")
    parser.add_argument("--generated", default="generated_pairs_molecule_aptamer.txt")
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
    parser.add_argument("--diagnostics", action="store_true", help="Print embedding variance diagnostics.")
    return parser


if __name__ == "__main__":
    rerank_one_molecule(build_parser().parse_args())
