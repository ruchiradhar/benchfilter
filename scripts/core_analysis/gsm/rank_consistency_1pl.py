#!/usr/bin/env python3
"""Cross-language item rank consistency for MGSM 1PL IRT results.

For each language, loads 1PL difficulty parameters and computes:
  - Spearman rank correlation matrix across languages
  - Kendall's W (coefficient of concordance)
  - Per-item rank stability (std dev of rank across languages)
  - Hierarchical clustering of languages based on difficulty profiles

Examples:
    python scripts/analysis/gsm_1pl/rank_consistency_1pl.py
    python scripts/analysis/gsm_1pl/rank_consistency_1pl.py --results_dir results/1PL --output_dir results/core_analysis/gsm_1pl
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
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ALL_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]

LANG_LABELS = {
    "bn": "Bengali", "de": "German", "en": "English", "es": "Spanish",
    "fr": "French", "ja": "Japanese", "ru": "Russian", "sw": "Swahili",
    "te": "Telugu", "th": "Thai", "zh": "Chinese",
}

CMAP_RYG = LinearSegmentedColormap.from_list(
    "red_yellow_green",
    [
        (0.00, "#CC0000"),
        (0.50, "#FFFF00"),
        (1.00, "#007700"),
    ],
)


def resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def available_languages(results_dir: Path) -> list[str]:
    return [
        lang for lang in ALL_LANGUAGES
        if (results_dir / f"mgsm_{lang}_1pl.json").exists()
    ]


def load_language_matrix(results_dir: Path, langs: list[str]) -> pd.DataFrame:
    """Return DataFrame (items × languages) of difficulty parameters."""
    cols = {}
    for lang in langs:
        path = results_dir / f"mgsm_{lang}_1pl.json"
        data = json.loads(path.read_text())
        item_ids = [data["item_ids"][str(i)] for i in range(len(data["diff"]))]
        cols[lang] = pd.Series(data["diff"], index=item_ids)
    return pd.DataFrame(cols)


def kendalls_w(rank_matrix: np.ndarray) -> float:
    n, m = rank_matrix.shape
    col_sums = rank_matrix.sum(axis=1)
    grand_mean = col_sums.mean()
    s = np.sum((col_sums - grand_mean) ** 2)
    return float(12 * s / (m ** 2 * (n ** 3 - n)))


def spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
    langs = df.columns.tolist()
    n = len(langs)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(df.iloc[:, i], df.iloc[:, j])
            mat[i, j] = mat[j, i] = rho
    return pd.DataFrame(mat, index=langs, columns=langs)


def rank_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=0, method="average")


def plot_spearman_heatmap(corr_df: pd.DataFrame, w: float, langs: list[str], ax: plt.Axes) -> object:
    labels = [LANG_LABELS[l] for l in langs]
    data = corr_df.values
    im = ax.imshow(data, vmin=-1, vmax=1, cmap=CMAP_RYG, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "black" if abs(data[i, j]) < 0.5 else "white"
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)
    return im


def plot_item_stability(rank_std: pd.Series, ax: plt.Axes, ylim: float | None = None) -> None:
    sorted_std = rank_std.sort_values()
    ax.bar(range(len(sorted_std)), sorted_std.values, color="steelblue", width=1.0, linewidth=0)
    ax.set_xlabel("Item (sorted by stability)", fontsize=9)
    ax.set_ylabel("Rank std dev across languages", fontsize=9)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    if ylim is not None:
        ax.set_ylim(0, ylim)


def plot_language_dendrogram(diff_df: pd.DataFrame, langs: list[str], ax: plt.Axes) -> None:
    rk = diff_df.rank(axis=0, method="average")
    profile_matrix = np.array([rk[l].values for l in langs])
    dist = ssd.pdist(profile_matrix, metric="correlation")
    linkage = sch.linkage(dist, method="average")
    labels = [LANG_LABELS[l] for l in langs]
    sch.dendrogram(linkage, labels=labels, ax=ax, leaf_rotation=45, leaf_font_size=9,
                   color_threshold=0.3 * max(linkage[:, 2]))
    ax.set_title("Language clustering\n(by item difficulty rank profile)", fontsize=10)
    ax.set_ylabel("Distance (1 − Spearman ρ)", fontsize=9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("results/1PL"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/core_analysis/gsm_1pl"))
    args = parser.parse_args()
    args.results_dir = resolve_path(args.results_dir)
    args.output_dir = resolve_path(args.output_dir)
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    langs = available_languages(args.results_dir)
    missing = [l for l in ALL_LANGUAGES if l not in langs]
    if missing:
        print(f"Missing languages (skipped): {missing}")
    print(f"Languages: {langs}")

    diff_df = load_language_matrix(args.results_dir, langs)
    rk = rank_df(diff_df)
    corr = spearman_matrix(diff_df)
    w = kendalls_w(rk.values)
    rank_std = rk.std(axis=1)

    # --- Figure 1: Spearman heatmap ---
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    im = plot_spearman_heatmap(corr, w, langs, ax1)
    fig1.colorbar(im, ax=ax1, label="Spearman ρ", shrink=0.8)
    fig1.tight_layout()
    fig1.savefig(args.output_dir / "spearman_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("Saved spearman_heatmap.png")

    # --- Figure 2: Kendall's W bar chart ---
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    bars = ax2.bar(["Difficulty"], [w], color="steelblue", edgecolor="white", width=0.4)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Kendall's W")
    for bar, val in zip(bars, [w]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(args.output_dir / "kendalls_w.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved kendalls_w.png")

    # --- Figure 3: Item rank stability ---
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    shared_ylim = rank_std.max() * 1.05
    plot_item_stability(rank_std, ax2, ylim=shared_ylim)
    fig2.tight_layout()
    fig2.savefig(args.output_dir / "item_rank_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved item_rank_stability.png")

    # --- Figure 4: Language dendrogram ---
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    plot_language_dendrogram(diff_df, langs, ax4)
    fig4.tight_layout()
    fig4.savefig(args.output_dir / "language_dendrogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print("Saved language_dendrogram.png")

    # --- CSVs ---
    pd.DataFrame({
        "n_languages": [len(langs)],
        "missing_languages": [",".join(missing)],
        "kendalls_w": [w],
        "n_items": [len(diff_df)],
    }).to_csv(args.output_dir / "kendalls_w_summary.csv", index=False)
    print("Saved kendalls_w_summary.csv")

    rows = [{"item_id": item_id, "rank_std": val} for item_id, val in rank_std.items()]
    pd.DataFrame(rows).to_csv(args.output_dir / "item_rank_stability.csv", index=False)
    print("Saved item_rank_stability.csv")

    print(f"\nKendall's W = {w:.4f}  (n_items={len(diff_df)}, n_langs={len(langs)})")
    print("\n=== Mean pairwise Spearman ρ ===")
    upper = corr.values[np.triu_indices(len(langs), k=1)]
    print(f"  mean ρ = {upper.mean():.4f}  min ρ = {upper.min():.4f}  max ρ = {upper.max():.4f}")


if __name__ == "__main__":
    main()
