#!/usr/bin/env python3
"""Cross-model item rank concordance: 1PL vs 2PL vs 3PL.

For each benchmark slice (MGSM language, MMLU language×domain), loads item
difficulty parameters from the 1PL, 2PL, and 3PL fits and computes pairwise
Spearman rank correlation between them.  Results are aggregated across all
slices per benchmark (no language/domain breakdown).

Outputs (in --output_dir):
  concordance_by_slice.csv   per-slice Spearman ρ for each of the 3 pairs
  concordance_summary.csv    mean / min / max ρ per pair per benchmark
  concordance_boxplot.png    box+strip plot of ρ distributions

Usage:
    python scripts/analysis/cross_model_concordance.py
    python scripts/analysis/cross_model_concordance.py \
        --results_root results --output_dir results/analysis/cross_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MGSM_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
MMLU_LANGUAGES = ["de", "en", "es", "fr", "ja", "sw", "zh"]
MMLU_DOMAINS = ["business", "humanities", "medical", "other", "social_sciences", "stem"]

PAIRS = [("1PL", "2PL"), ("1PL", "3PL"), ("2PL", "3PL")]
PAIR_KEYS = [f"{a}_{b}" for a, b in PAIRS]
PAIR_LABELS = {"1PL_2PL": "1PL vs 2PL", "1PL_3PL": "1PL vs 3PL", "2PL_3PL": "2PL vs 3PL"}

BENCHMARK_COLORS = {"MGSM": "steelblue", "MMLU": "darkorange"}


def resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_diff(results_root: Path, pl: str, stem: str) -> pd.Series | None:
    path = results_root / pl / f"{stem}_{pl.lower()}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    n = len(data["diff"])
    item_ids = [data["item_ids"][str(i)] for i in range(n)]
    return pd.Series(data["diff"], index=item_ids)


def slice_concordance(results_root: Path, stem: str) -> dict[str, float] | None:
    """Return pairwise Spearman ρ of difficulty rankings for one slice."""
    series: dict[str, pd.Series] = {}
    for pl in ["1PL", "2PL", "3PL"]:
        s = load_diff(results_root, pl, stem)
        if s is None:
            return None
        series[pl] = s

    common = series["1PL"].index
    for pl in ["2PL", "3PL"]:
        common = common.intersection(series[pl].index)
    if len(common) < 3:
        return None

    result: dict[str, float] = {}
    for a, b in PAIRS:
        rho, _ = spearmanr(series[a].loc[common], series[b].loc[common])
        result[f"{a}_{b}"] = float(rho)
    return result


def collect_rows(results_root: Path) -> list[dict]:
    rows: list[dict] = []

    for lang in MGSM_LANGUAGES:
        stem = f"mgsm_{lang}"
        r = slice_concordance(results_root, stem)
        if r is not None:
            rows.append({"benchmark": "MGSM", "slice": stem, **r})
        else:
            print(f"  skipped (missing files): {stem}")

    for lang in MMLU_LANGUAGES:
        for domain in MMLU_DOMAINS:
            stem = f"mmlu_{lang}_{domain}"
            r = slice_concordance(results_root, stem)
            if r is not None:
                rows.append({"benchmark": "MMLU", "slice": stem, **r})
            else:
                print(f"  skipped (missing files): {stem}")

    return rows


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for benchmark in ["MGSM", "MMLU"]:
        sub = df[df["benchmark"] == benchmark]
        for key in PAIR_KEYS:
            vals = sub[key].dropna()
            records.append({
                "benchmark": benchmark,
                "pair": PAIR_LABELS[key],
                "n_slices": len(vals),
                "mean_rho": vals.mean(),
                "min_rho": vals.min(),
                "max_rho": vals.max(),
                "std_rho": vals.std(),
            })
    return pd.DataFrame(records)


def plot_concordance(df: pd.DataFrame, output_path: Path) -> None:
    benchmarks = ["MGSM", "MMLU"]
    n_pairs = len(PAIR_KEYS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    fig.suptitle(
        "Item difficulty rank concordance across IRT models\n(Spearman ρ of difficulty rankings, per benchmark slice)",
        fontsize=11,
    )

    for ax, benchmark in zip(axes, benchmarks):
        sub = df[df["benchmark"] == benchmark]
        color = BENCHMARK_COLORS[benchmark]

        positions = np.arange(n_pairs)
        data_per_pair = [sub[key].dropna().values for key in PAIR_KEYS]

        bp = ax.boxplot(
            data_per_pair,
            positions=positions,
            widths=0.4,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            boxprops=dict(facecolor=color, alpha=0.5),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            flierprops=dict(marker="o", color=color, markersize=4, alpha=0.5),
        )

        # Overlay individual slice points with jitter
        rng = np.random.default_rng(42)
        for i, vals in enumerate(data_per_pair):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                positions[i] + jitter, vals,
                color=color, alpha=0.6, s=18, zorder=3,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels([PAIR_LABELS[k] for k in PAIR_KEYS], fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Spearman ρ" if benchmark == "MGSM" else "", fontsize=10)
        ax.set_title(f"{benchmark}  (n={len(sub)} slices)", fontsize=10)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(0.9, color="orange", linestyle="--", linewidth=0.8, label="ρ = 0.9")
        ax.axhline(0.7, color="red", linestyle="--", linewidth=0.8, label="ρ = 0.7")
        ax.legend(fontsize=8, loc="lower right")

        # Annotate medians
        for i, vals in enumerate(data_per_pair):
            med = np.median(vals)
            ax.text(
                positions[i], med + 0.02, f"{med:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/analysis/cross_model"))
    args = parser.parse_args()
    args.results_root = resolve_path(args.results_root)
    args.output_dir = resolve_path(args.output_dir)
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading slices...")
    rows = collect_rows(args.results_root)
    if not rows:
        raise RuntimeError("No slices loaded — check --results_root path.")

    df = pd.DataFrame(rows)

    df.to_csv(args.output_dir / "concordance_by_slice.csv", index=False)
    print(f"Saved concordance_by_slice.csv  ({len(df)} slices)")

    summary = build_summary(df)
    summary.to_csv(args.output_dir / "concordance_summary.csv", index=False)
    print("Saved concordance_summary.csv")

    plot_concordance(df, args.output_dir / "concordance_boxplot.png")

    print("\n=== Cross-model item rank concordance (Spearman ρ of difficulty) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
