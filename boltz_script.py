#!/usr/bin/env python3
"""
Boltz-2 affinity utilities for aptamer-small molecule complexes.

Main use case:
    1. Sample N positive and N negative pairs from AptaBench.
    2. Run Boltz-2 affinity prediction for every pair.
    3. Save CSV/JSON results and affinity distribution plots.

Examples:
    python boltz_script.py aptabench --data aptabench_with_embeddings.csv --n-positive 100 --n-negative 100
    python boltz_script.py aptabench --data aptabench_with_embeddings.csv --n-positive 5 --n-negative 5 --dry-run
    python boltz_script.py plot --results boltz_aptabench_affinity/run_YYYYMMDD_HHMMSS/affinity_results.csv
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


AFFINITY_COLUMNS = [
    "affinity_pred_value",
    "affinity_pred_value1",
    "affinity_pred_value2",
]

PROBABILITY_COLUMNS = [
    "affinity_probability_binary",
    "affinity_probability_binary1",
    "affinity_probability_binary2",
]

# Boltz returns 3 affinity estimates per pair (suffix "", "1", "2").
BOLTZ_RUN_VARIANTS = [
    (0, "affinity_pred_value", "affinity_probability_binary"),
    (1, "affinity_pred_value1", "affinity_probability_binary1"),
    (2, "affinity_pred_value2", "affinity_probability_binary2"),
]

BEST_RUN_AFFINITY_COL = "affinity_pred_value_best"
BEST_RUN_PROBABILITY_COL = "affinity_probability_binary_best"


def sanitize_name(value: object, max_len: int = 80) -> str:
    text = str(value) if value is not None else "unknown"
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in text)
    safe = safe.strip("_") or "unknown"
    return safe[:max_len]


def normalize_molecule_type(value: object) -> str:
    mol_type = str(value).strip().lower()
    if mol_type not in {"dna", "rna"}:
        return "dna"
    return mol_type


def normalize_sequence(sequence: object, molecule_type: str) -> str:
    seq = str(sequence).strip().upper()
    if molecule_type == "dna":
        return seq.replace("U", "T")
    return seq.replace("T", "U")


class BoltzAffinityPredictor:
    """Run Boltz-2 affinity prediction and organize all outputs."""

    def __init__(
        self,
        output_base_dir: str = "boltz_aptabench_affinity",
        use_msa_server: bool = True,
        affinity_mw_correction: bool = True,
        timeout_seconds: int = 3600,
    ):
        self.output_base_dir = Path(output_base_dir)
        self.use_msa_server = use_msa_server
        self.affinity_mw_correction = affinity_mw_correction
        self.timeout_seconds = timeout_seconds

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_base_dir / f"run_{self.timestamp}"
        self.configs_dir = self.run_dir / "configs"
        self.results_dir = self.run_dir / "results"
        self.logs_dir = self.run_dir / "logs"
        self.plots_dir = self.run_dir / "plots"

        for dir_path in [self.configs_dir, self.results_dir, self.logs_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def create_config(
        self,
        aptamer_sequence: str,
        target_smiles: str,
        molecule_type: str = "dna",
        complex_name: str = "complex",
    ) -> Path:
        """Create a Boltz YAML config for one aptamer-ligand pair."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to write Boltz YAML configs.") from exc

        molecule_type = normalize_molecule_type(molecule_type)
        aptamer_sequence = normalize_sequence(aptamer_sequence, molecule_type)
        complex_name = sanitize_name(complex_name)

        config = {
            "version": 1,
            "sequences": [
                {
                    molecule_type: {
                        "id": "A",
                        "sequence": aptamer_sequence,
                    }
                },
                {
                    "ligand": {
                        "id": "B",
                        "smiles": str(target_smiles).strip(),
                    }
                },
            ],
            "properties": [
                {
                    "affinity": {
                        "binder": "B",
                    }
                }
            ],
        }

        config_path = self.configs_dir / f"{complex_name}.yaml"
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.dump(config, handle, default_flow_style=False, indent=2, allow_unicode=True)

        return config_path

    def run_prediction(self, config_path: Path, dry_run: bool = False) -> Dict:
        """Run Boltz for one config and return parsed affinity data."""
        complex_name = config_path.stem
        output_dir = self.results_dir / complex_name
        log_path = self.logs_dir / f"{complex_name}.log"

        cmd = f'boltz predict "{config_path}"'
        if self.use_msa_server:
            cmd += " --use_msa_server"
        if self.affinity_mw_correction:
            cmd += " --affinity_mw_correction"
        cmd += f' --out_dir "{output_dir}"'

        if dry_run:
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("DRY RUN\n")
                handle.write(cmd + "\n")
            return {"error": "dry_run", "command": cmd}

        print(f"[RUN] {complex_name}")
        print(f"      {cmd}")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                encoding="utf-8",
                cwd=str(Path.cwd()),
            )
        except subprocess.TimeoutExpired:
            return {"error": "timeout", "command": cmd}
        except Exception as exc:
            return {"error": str(exc), "command": cmd}

        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("=== COMMAND ===\n")
            handle.write(cmd)
            handle.write("\n=== STDOUT ===\n")
            handle.write(result.stdout or "")
            handle.write("\n=== STDERR ===\n")
            handle.write(result.stderr or "")
            handle.write(f"\n=== RETURN CODE ===\n{result.returncode}")

        if result.returncode != 0:
            return {"error": f"return_code_{result.returncode}", "command": cmd}

        affinity_data = self._parse_results(output_dir)
        if affinity_data:
            print(
                f"[OK] {complex_name}: "
                f"{affinity_data.get('affinity_pred_value', 'N/A')}"
            )
            return affinity_data

        return {"error": "no_results_found", "command": cmd}

    def _parse_results(self, output_dir: Path) -> Optional[Dict]:
        """Find and parse Boltz affinity JSON outputs below output_dir."""
        if not output_dir.exists():
            return None

        json_files = []
        for pattern in ["**/affinity*.json", "**/*.json"]:
            json_files.extend(output_dir.glob(pattern))

        seen = set()
        unique_json_files = []
        for path in json_files:
            resolved = str(path.resolve())
            if resolved not in seen:
                unique_json_files.append(path)
                seen.add(resolved)

        for json_file in unique_json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception:
                continue

            if "affinity_pred_value" not in data:
                continue

            result = {col: data.get(col) for col in AFFINITY_COLUMNS + PROBABILITY_COLUMNS}
            result["source_file"] = str(json_file)
            return result

        return None

    def run_pairs(self, pairs: List[Dict], dry_run: bool = False) -> pd.DataFrame:
        """Run Boltz on an explicit list of pair dictionaries."""
        results = []
        print(f"\nRunning {len(pairs)} aptamer-ligand pairs")
        print(f"Run directory: {self.run_dir}\n")

        for idx, pair in enumerate(pairs, 1):
            pair_id = pair["pair_id"]
            print(f"[{idx}/{len(pairs)}] {pair_id} label={pair.get('label')}")

            config_path = self.create_config(
                aptamer_sequence=pair["sequence"],
                target_smiles=pair["canonical_smiles"],
                molecule_type=pair.get("molecule_type", "dna"),
                complex_name=pair_id,
            )

            affinity_data = self.run_prediction(config_path, dry_run=dry_run)
            result = {
                **pair,
                **affinity_data,
                "config_path": str(config_path),
            }
            results.append(result)
            self.save_intermediate_results(results)

        results_df = pd.DataFrame(results)
        self.save_final_results(results_df)
        return results_df

    def save_intermediate_results(self, results: List[Dict]) -> None:
        temp_path = self.run_dir / "intermediate_results.csv"
        pd.DataFrame(results).to_csv(temp_path, index=False, encoding="utf-8-sig")

    def save_final_results(self, df: pd.DataFrame) -> None:
        csv_path = self.run_dir / "affinity_results.csv"
        json_path = self.run_dir / "affinity_results.json"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        df.to_json(json_path, orient="records", indent=2, force_ascii=False)
        print(f"\nSaved results:")
        print(f"  CSV:  {csv_path}")
        print(f"  JSON: {json_path}")


def load_aptabench_pairs(
    data_path: str,
    n_positive: int = 100,
    n_negative: int = 100,
    random_state: int = 42,
    max_sequence_length: Optional[int] = None,
    dna_only: bool = False,
) -> List[Dict]:
    """Sample positive and negative aptamer-ligand pairs from AptaBench."""
    required_cols = ["type", "sequence", "canonical_smiles", "label"]
    optional_cols = ["pKd_value", "origin", "source"]
    header = pd.read_csv(data_path, nrows=0)
    usecols = required_cols + [c for c in optional_cols if c in header.columns]

    df = pd.read_csv(data_path, usecols=usecols, low_memory=False)
    df = df.dropna(subset=["type", "sequence", "canonical_smiles", "label"]).copy()
    df["aptabench_row_index"] = df.index
    df["label"] = df["label"].astype(int)
    df["molecule_type"] = df["type"].apply(normalize_molecule_type)
    df["sequence"] = [
        normalize_sequence(seq, mol_type)
        for seq, mol_type in zip(df["sequence"], df["molecule_type"])
    ]
    df["sequence_length"] = df["sequence"].str.len()

    if max_sequence_length is not None:
        df = df[df["sequence_length"] <= max_sequence_length].copy()
    if dna_only:
        df = df[df["molecule_type"] == "dna"].copy()

    samples = []
    for label, n in [(1, n_positive), (0, n_negative)]:
        subset = df[df["label"] == label]
        if subset.empty:
            raise ValueError(f"No rows found for label={label}")
        take_n = min(n, len(subset))
        if take_n < n:
            print(f"[WARN] Requested {n} rows for label={label}, only {take_n} available.")
        samples.append(subset.sample(n=take_n, random_state=random_state + label))

    sampled = pd.concat(samples, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    pairs = []
    for idx, row in sampled.iterrows():
        label_name = "pos" if int(row["label"]) == 1 else "neg"
        pair_id = f"aptabench_{idx:04d}_{label_name}_{normalize_molecule_type(row['molecule_type'])}"
        pair = {
            "pair_id": pair_id,
            "dataset_row_index": int(row["aptabench_row_index"]),
            "label": int(row["label"]),
            "label_name": "positive" if int(row["label"]) == 1 else "negative",
            "molecule_type": normalize_molecule_type(row["molecule_type"]),
            "sequence": row["sequence"],
            "sequence_length": int(row["sequence_length"]),
            "canonical_smiles": row["canonical_smiles"],
        }
        for col in optional_cols:
            if col in row:
                pair[col] = row[col]
        pairs.append(pair)

    return pairs


def load_selected_pairs(selected_pairs_path: str) -> List[Dict]:
    """Load a previously saved selected_aptabench_pairs.csv file."""
    df = pd.read_csv(selected_pairs_path, low_memory=False)
    required = ["pair_id", "sequence", "canonical_smiles", "molecule_type"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Selected pairs file is missing columns: {missing}")

    pairs = []
    for _, row in df.iterrows():
        mol_type = normalize_molecule_type(row["molecule_type"])
        pair = row.to_dict()
        pair["molecule_type"] = mol_type
        pair["sequence"] = normalize_sequence(row["sequence"], mol_type)
        pair["canonical_smiles"] = str(row["canonical_smiles"]).strip()
        pairs.append(pair)
    return pairs


def collapse_to_most_confident_run(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each pair, pick the Boltz run with the highest binary probability
    among affinity_pred_value / value1 / value2.
    """
    df = results_df.copy()
    variants = [
        (run_idx, aff_col, prob_col)
        for run_idx, aff_col, prob_col in BOLTZ_RUN_VARIANTS
        if aff_col in df.columns and prob_col in df.columns
    ]
    if not variants:
        return df

    aff_cols = [aff_col for _, aff_col, _ in variants]
    prob_cols = [prob_col for _, _, prob_col in variants]
    run_indices = [run_idx for run_idx, _, _ in variants]

    aff_matrix = df[aff_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    prob_matrix = df[prob_cols].apply(pd.to_numeric, errors="coerce").to_numpy()

    with np.errstate(all="ignore"):
        best_col_idx = np.nanargmax(prob_matrix, axis=1)

    row_idx = np.arange(len(df))
    df["boltz_best_run_index"] = [run_indices[i] for i in best_col_idx]
    df[BEST_RUN_AFFINITY_COL] = aff_matrix[row_idx, best_col_idx]
    df[BEST_RUN_PROBABILITY_COL] = prob_matrix[row_idx, best_col_idx]
    return df


def summarize_affinity_results(
    results_df: pd.DataFrame,
    output_dir: Path,
    metric_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Save grouped summary statistics for affinity and probability columns."""
    if metric_columns is not None:
        numeric_cols = [col for col in metric_columns if col in results_df.columns]
    else:
        numeric_cols = [col for col in AFFINITY_COLUMNS + PROBABILITY_COLUMNS if col in results_df.columns]
    for col in numeric_cols:
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

    summary_parts = []
    for col in numeric_cols:
        stats = results_df.groupby("label_name")[col].agg(["count", "mean", "std", "median", "min", "max"])
        stats.insert(0, "metric", col)
        summary_parts.append(stats.reset_index())

    if not summary_parts:
        return pd.DataFrame()

    summary_df = pd.concat(summary_parts, ignore_index=True)
    summary_path = output_dir / "affinity_summary_by_label.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"  Summary: {summary_path}")
    return summary_df


def classify_boltz_affinity(value: float) -> str:
    """Map Boltz affinity_pred_value to a simple class."""
    if pd.isna(value):
        return "missing"
    if value > -3.0:
        return "good_negative"
    if value > -5.0:
        return "borderline"
    return "potential_positive"


def plot_affinity_distributions(
    results_df: pd.DataFrame,
    output_dir: Path,
    plot_all_runs: bool = False,
) -> bool:
    """Plot positive vs negative affinity distributions. Returns True if plots were created."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is not installed. Skipping plots.")
        print("Install with: pip install matplotlib")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    if "label_name" not in results_df.columns:
        if "label" not in results_df.columns:
            raise ValueError("Results must contain either label_name or label.")
        results_df["label_name"] = results_df["label"].map({1: "positive", 0: "negative"})

    if plot_all_runs:
        available_affinity = [col for col in AFFINITY_COLUMNS if col in results_df.columns]
        available_probs = [col for col in PROBABILITY_COLUMNS if col in results_df.columns]
        summary_metrics = available_affinity + available_probs
    else:
        results_df = collapse_to_most_confident_run(results_df)
        print(
            "Plots use the most confident Boltz run per pair "
            "(max affinity_probability_binary among 3 runs)."
        )
        collapsed_path = output_dir / "affinity_most_confident.csv"
        save_cols = [
            c
            for c in [
                "pair_id",
                "label",
                "label_name",
                "boltz_best_run_index",
                BEST_RUN_AFFINITY_COL,
                BEST_RUN_PROBABILITY_COL,
            ]
            if c in results_df.columns
        ]
        results_df[save_cols].to_csv(collapsed_path, index=False, encoding="utf-8-sig")
        print(f"  Table: {collapsed_path}")
        available_affinity = [BEST_RUN_AFFINITY_COL]
        available_probs = [BEST_RUN_PROBABILITY_COL]
        summary_metrics = available_affinity + available_probs

    for col in available_affinity + available_probs:
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

    for col in available_affinity + available_probs:
        data = results_df.dropna(subset=[col])
        if data.empty:
            continue

        plt.figure(figsize=(10, 6))
        for label_name, color in [("positive", "green"), ("negative", "red")]:
            values = data.loc[data["label_name"] == label_name, col].dropna()
            if values.empty:
                continue
            plt.hist(
                values,
                bins=30,
                alpha=0.55,
                density=True,
                label=f"{label_name} (n={len(values)}, mean={values.mean():.3f})",
                color=color,
            )
            plt.axvline(values.mean(), color=color, linestyle="--", linewidth=1)

        title_col = col.replace("_best", " (most confident run)")
        plt.title(f"Boltz distribution: {title_col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        out_path = output_dir / f"{col}_distribution.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Plot: {out_path}")

    if available_affinity:
        fig, axes = plt.subplots(1, len(available_affinity), figsize=(6 * len(available_affinity), 5))
        if len(available_affinity) == 1:
            axes = [axes]
        for ax, col in zip(axes, available_affinity):
            data = results_df.dropna(subset=[col])
            groups = [
                data.loc[data["label_name"] == "positive", col].dropna(),
                data.loc[data["label_name"] == "negative", col].dropna(),
            ]
            ax.boxplot(groups, labels=["positive", "negative"])
            ax.set_title(col)
            ax.set_ylabel("Boltz affinity")
            ax.grid(alpha=0.25)
        plt.tight_layout()
        out_path = output_dir / "affinity_boxplots.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Plot: {out_path}")

    summarize_affinity_results(results_df, output_dir, metric_columns=summary_metrics)
    return True


def run_aptabench_mode(args: argparse.Namespace) -> pd.DataFrame:
    if args.from_selected_pairs:
        pairs = load_selected_pairs(args.from_selected_pairs)
    else:
        pairs = load_aptabench_pairs(
            data_path=args.data,
            n_positive=args.n_positive,
            n_negative=args.n_negative,
            random_state=args.seed,
            max_sequence_length=args.max_sequence_length,
            dna_only=args.dna_only,
        )

    predictor = BoltzAffinityPredictor(
        output_base_dir=args.output_dir,
        use_msa_server=not args.no_msa_server,
        affinity_mw_correction=not args.no_mw_correction,
        timeout_seconds=args.timeout,
    )

    selected_pairs_path = predictor.run_dir / "selected_aptabench_pairs.csv"
    pd.DataFrame(pairs).to_csv(selected_pairs_path, index=False, encoding="utf-8-sig")
    print(f"Selected pairs saved to: {selected_pairs_path}")

    results_df = predictor.run_pairs(pairs, dry_run=args.dry_run)
    if not args.dry_run:
        if not args.skip_plots:
            plot_affinity_distributions(
                results_df,
                predictor.plots_dir,
                plot_all_runs=args.plot_all_runs,
            )
        print_boltz_threshold_legend(results_df, use_most_confident=not args.plot_all_runs)
    return results_df


def print_boltz_threshold_legend(
    results_df: pd.DataFrame,
    use_most_confident: bool = True,
) -> None:
    """Print a short interpretation guide for Boltz affinity_pred_value."""
    df = results_df.copy()
    if use_most_confident:
        df = collapse_to_most_confident_run(df)
        affinity_col = BEST_RUN_AFFINITY_COL
        print("\nClassification uses the most confident Boltz run per pair.")
    else:
        affinity_col = "affinity_pred_value"

    if affinity_col not in df.columns:
        return

    print("\nBoltz affinity interpretation (script heuristic):")
    print("  affinity_pred_value > -3.0  -> good_negative (weaker binding)")
    print("  affinity_pred_value > -5.0  -> borderline")
    print("  affinity_pred_value <= -5.0 -> potential_positive (stronger binding)")
    df["boltz_class"] = df[affinity_col].apply(classify_boltz_affinity)
    print("\nCounts by class:")
    print(df["boltz_class"].value_counts().to_string())


def interactive_input() -> Optional[pd.DataFrame]:
    """Interactive mode compatible with the original script."""
    print("\nBoltz-2 Affinity Predictor - Interactive Mode")
    use_msa = input("Use MSA server? (y/n, default y): ").lower().strip() != "n"
    use_mw_corr = input("Use affinity MW correction? (y/n, default y): ").lower().strip() != "n"

    predictor = BoltzAffinityPredictor(
        output_base_dir="boltz_aptamer_screening",
        use_msa_server=use_msa,
        affinity_mw_correction=use_mw_corr,
    )

    target_name = input("Target name: ").strip() or "target"
    target_smiles = input("SMILES: ").strip()
    while not target_smiles:
        target_smiles = input("SMILES cannot be empty. SMILES: ").strip()

    aptamers = []
    while True:
        name = input("Aptamer name (or 'stop'): ").strip()
        if name.lower() == "stop":
            break
        mol_type = normalize_molecule_type(input("Type (dna/rna, default dna): ").strip() or "dna")
        sequence = normalize_sequence(input("Sequence: ").strip(), mol_type)
        if not sequence:
            print("Sequence cannot be empty.")
            continue
        aptamers.append(
            {
                "pair_id": f"{sanitize_name(name)}_{sanitize_name(target_name)}",
                "label": None,
                "label_name": "manual",
                "molecule_type": mol_type,
                "sequence": sequence,
                "sequence_length": len(sequence),
                "canonical_smiles": target_smiles,
                "target_name": target_name,
            }
        )
        if input("Add another? (y/n, default n): ").strip().lower() != "y":
            break

    if not aptamers:
        print("No aptamers provided.")
        return None

    return predictor.run_pairs(aptamers, dry_run=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Boltz-2 affinity analysis utilities.")
    subparsers = parser.add_subparsers(dest="command")

    apta = subparsers.add_parser("aptabench", help="Sample AptaBench pairs and run Boltz.")
    apta.add_argument("--data", default="aptabench_with_embeddings.csv", help="AptaBench CSV path.")
    apta.add_argument("--n-positive", type=int, default=100, help="Number of positive pairs.")
    apta.add_argument("--n-negative", type=int, default=100, help="Number of negative pairs.")
    apta.add_argument("--seed", type=int, default=42, help="Random seed.")
    apta.add_argument("--output-dir", default="boltz_aptabench_affinity", help="Output base directory.")
    apta.add_argument("--timeout", type=int, default=3600, help="Timeout per Boltz prediction in seconds.")
    apta.add_argument("--max-sequence-length", type=int, default=None, help="Optional max aptamer length.")
    apta.add_argument("--dna-only", action="store_true", help="Use only DNA aptamers.")
    apta.add_argument("--no-msa-server", action="store_true", help="Disable --use_msa_server.")
    apta.add_argument("--no-mw-correction", action="store_true", help="Disable --affinity_mw_correction.")
    apta.add_argument("--dry-run", action="store_true", help="Create configs and selected pairs only.")
    apta.add_argument("--skip-plots", action="store_true", help="Skip distribution plots even if matplotlib is available.")
    apta.add_argument(
        "--plot-all-runs",
        action="store_true",
        help="Plot all 3 Boltz runs separately instead of only the most confident run per pair.",
    )
    apta.add_argument(
        "--from-selected-pairs",
        default=None,
        help="Run an existing selected_aptabench_pairs.csv instead of sampling again.",
    )

    plot = subparsers.add_parser("plot", help="Plot distributions from an existing results CSV.")
    plot.add_argument("--results", required=True, help="Path to affinity_results.csv.")
    plot.add_argument("--output-dir", default=None, help="Directory for plots.")
    plot.add_argument(
        "--plot-all-runs",
        action="store_true",
        help="Plot all 3 Boltz runs separately instead of only the most confident run per pair.",
    )

    subparsers.add_parser("interactive", help="Manual interactive mode.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "aptabench":
        run_aptabench_mode(args)
    elif args.command == "plot":
        results_path = Path(args.results)
        out_dir = Path(args.output_dir) if args.output_dir else results_path.parent / "plots"
        df = pd.read_csv(results_path)
        plot_affinity_distributions(df, out_dir, plot_all_runs=args.plot_all_runs)
    else:
        interactive_input()


if __name__ == "__main__":
    main()
