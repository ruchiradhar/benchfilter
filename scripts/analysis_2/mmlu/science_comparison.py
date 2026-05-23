#!/usr/bin/env python3
"""MMLU science vs non-science item difficulty rank consistency, 1PL / 2PL / 3PL.

Produces a 2×3 Spearman heatmap grid:
  rows  — Science (STEM + Medical) vs Non-science (Business + Humanities + Other + Social Sciences)
  cols  — 1PL, 2PL, 3PL

Each cell is the element-wise average Spearman ρ matrix across the constituent domains.
Annotated with the average Kendall's W across those domains.

Examples:
    python scripts/analysis/mmlu/science_comparison.py
    python scripts/analysis/mmlu/science_comparison.py \\
        --results_1pl results/1PL --results_2pl results/2PL --results_3pl results/3PL \\
        --output_dir results/analysis_2/mmlu_science_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LANGUAGES = ["de", "en", "es", "fr", "ja", "sw", "zh"]
LANG_LABELS = {
    "de": "German", "en": "English", "es": "Spanish", "fr": "French",
    "ja": "Japanese", "sw": "Swahili", "zh": "Chinese",
}

SCIENCE_DOMAINS     = ["stem", "medical"]
NONSCIENCE_DOMAINS  = ["business", "humanities", "other", "social_sciences"]

CMAP_RYG = LinearSegmentedColormap.from_list(
    "red_yellow_green",
    [(0.00, "#CC0000"), (0.50, "#FFFF00"), (1.00, "#007700")],
)

MODEL_CONFIGS = {
    "1PL": {"suffix": "1pl", "param": "diff"},
    "2PL": {"suffix": "2pl", "param": "diff"},
    "3PL": {"suffix": "3pl", "param": "diff"},
}


def resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_domain_df(results_dir: Path, suffix: str, param: str, domain: str) -> pd.DataFrame:
    cols = {}
    for lang in LANGUAGES:
        path = results_dir / f"mmlu_{lang}_{domain}_{suffix}.json"
        data = json.loads(path.read_text())
        item_ids = [data["item_ids"][str(i)] for i in range(len(data[param]))]
        cols[lang] = pd.Series(data[param], index=item_ids)
    return pd.DataFrame(cols)


def spearman_matrix(df: pd.DataFrame) -> np.ndarray:
    n = len(LANGUAGES)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(df.iloc[:, i], df.iloc[:, j])
            mat[i, j] = mat[j, i] = rho
    return mat


def kendalls_w(df: pd.DataFrame) -> float:
    ranks = df.rank(axis=0, method="average").values
    n, m = ranks.shape
    col_sums = ranks.sum(axis=1)
    s = np.sum((col_sums - col_sums.mean()) ** 2)
    return float(12 * s / (m ** 2 * (n ** 3 - n)))


def group_avg_spearman(results_dir: Path, suffix: str, param: str,
                        domains: list[str]) -> tuple[np.ndarray, float]:
    """Return (avg Spearman matrix, avg Kendall's W) over the given domains."""
    corr_stack, w_vals = [], []
    for domain in domains:
        df = load_domain_df(results_dir, suffix, param, domain)
        corr_stack.append(spearman_matrix(df))
        w_vals.append(kendalls_w(df))
    return np.mean(corr_stack, axis=0), float(np.mean(w_vals))


def plot_heatmap(ax: plt.Axes, mat: np.ndarray, title: str, w: float) -> object:
    labels = [LANG_LABELS[l] for l in LANGUAGES]
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap=CMAP_RYG, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "black" if abs(mat[i, j]) < 0.5 else "white"
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color=color)
    return im


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_1pl", type=Path, default=Path("results/1PL"))
    parser.add_argument("--results_2pl", type=Path, default=Path("results/2PL_map"))
    parser.add_argument("--results_3pl", type=Path, default=Path("results/3PL_map"))
    parser.add_argument("--output_dir",  type=Path, default=Path("results/analysis_2/mmlu_science_comparison"))
    args = parser.parse_args()
    args.results_1pl = resolve_path(args.results_1pl)
    args.results_2pl = resolve_path(args.results_2pl)
    args.results_3pl = resolve_path(args.results_3pl)
    args.output_dir  = resolve_path(args.output_dir)
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = {
        "1PL": args.results_1pl,
        "2PL": args.results_2pl,
        "3PL": args.results_3pl,
    }

    # Compute grouped matrices for each model
    science_mat, science_w = {}, {}
    nonsci_mat,  nonsci_w  = {}, {}
    for model, cfg in MODEL_CONFIGS.items():
        rdir   = model_dirs[model]
        suffix = cfg["suffix"]
        param  = cfg["param"]
        science_mat[model], science_w[model] = group_avg_spearman(
            rdir, suffix, param, SCIENCE_DOMAINS)
        nonsci_mat[model],  nonsci_w[model]  = group_avg_spearman(
            rdir, suffix, param, NONSCIENCE_DOMAINS)

    # --- 2×3 figure: rows = Science / Non-science, cols = 1PL / 2PL / 3PL ---
    models = ["1PL", "2PL", "3PL"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    last_im = None
    row_labels = [
        ("Science", "STEM + Medical", science_mat, science_w),
        ("Non-science", "Business + Humanities + Other + Social Sciences", nonsci_mat, nonsci_w),
    ]
    for row_idx, (group_name, group_desc, mat_dict, w_dict) in enumerate(row_labels):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            title = f"{model}  |  {group_name}"
            im = plot_heatmap(ax, mat_dict[model], title, w_dict[model])
            last_im = im
        # Row label on the left
        axes[row_idx, 0].set_ylabel(f"{group_name}\n({group_desc})", fontsize=8, labelpad=6)

    fig.colorbar(last_im, ax=axes, label="Spearman ρ", shrink=0.55, pad=0.02)
    fig.tight_layout()
    out_path = args.output_dir / "science_vs_nonscience_heatmaps.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # Print summary stats
    print("\n=== Avg Kendall's W by group and model ===")
    print(f"  {'model':6s}  {'science W':>12}  {'non-science W':>14}")
    for model in models:
        print(f"  {model:6s}  {science_w[model]:12.4f}  {nonsci_w[model]:14.4f}")

    print("\n=== Mean pairwise Spearman ρ by group and model ===")
    n = len(LANGUAGES)
    idx = np.triu_indices(n, k=1)
    print(f"  {'model':6s}  {'science mean ρ':>15}  {'non-science mean ρ':>19}")
    for model in models:
        s_rho  = science_mat[model][idx].mean()
        ns_rho = nonsci_mat[model][idx].mean()
        print(f"  {model:6s}  {s_rho:15.4f}  {ns_rho:19.4f}")


if __name__ == "__main__":
    main()
