#!/usr/bin/env python3
"""Cross-language item rank consistency for MGSM 2PL IRT results.

Analyses both difficulty (diff) and discrimination (disc) parameters.
Computes:
  - Spearman rank correlation matrix across languages (diff and disc separately)
  - Kendall's W for both parameters
  - Per-item rank stability (std dev of rank across languages)
  - Hierarchical clustering of languages based on combined diff + disc profiles

Examples:
    python scripts/analysis/gsm_2pl/rank_consistency_2pl.py
    python scripts/analysis/gsm_2pl/rank_consistency_2pl.py --results_dir results/2PL --output_dir results/analysis/gsm_2pl
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
        if (results_dir / f"mgsm_{lang}_2pl.json").exists()
    ]


def load_language_matrix(results_dir: Path, langs: list[str], param: str) -> pd.DataFrame:
    cols = {}
    for lang in langs:
        path = results_dir / f"mgsm_{lang}_2pl.json"
        data = json.loads(path.read_text())
        item_ids = [data["item_ids"][str(i)] for i in range(len(data[param]))]
        cols[lang] = pd.Series(data[param], index=item_ids)
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


def plot_spearman_heatmap(corr_df: pd.DataFrame, w: float, langs: list[str],
                          param_label: str, ax: plt.Axes) -> object:
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
    ax.set_title(f"MGSM 2PL — Item {param_label} rank correlation\nKendall's W = {w:.3f}", fontsize=10)
    return im


def plot_item_stability(rank_std: pd.Series, param_label: str, ax: plt.Axes) -> None:
    sorted_std = rank_std.sort_values()
    ax.bar(range(len(sorted_std)), sorted_std.values, color="steelblue", width=1.0, linewidth=0)
    ax.set_xlabel("Item (sorted by stability)", fontsize=9)
    ax.set_ylabel("Rank std dev", fontsize=9)
    ax.set_title(f"Item rank stability [{param_label}]", fontsize=10)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)


def plot_language_dendrogram(diff_df: pd.DataFrame, disc_df: pd.DataFrame,
                              langs: list[str], ax: plt.Axes) -> None:
    rk_diff = diff_df.rank(axis=0, method="average")
    rk_disc = disc_df.rank(axis=0, method="average")
    profiles = {l: list(rk_diff[l].values) + list(rk_disc[l].values) for l in langs}
    profile_matrix = np.array([profiles[l] for l in langs])
    dist = ssd.pdist(profile_matrix, metric="correlation")
    linkage = sch.linkage(dist, method="average")
    labels = [LANG_LABELS[l] for l in langs]
    sch.dendrogram(linkage, labels=labels, ax=ax, leaf_rotation=45, leaf_font_size=9,
                   color_threshold=0.3 * max(linkage[:, 2]))
    ax.set_title("Language clustering\n(diff + disc rank profile)", fontsize=10)
    ax.set_ylabel("Distance (1 − Spearman ρ)", fontsize=9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("results/2PL"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/analysis/gsm_2pl"))
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

    diff_df = load_language_matrix(args.results_dir, langs, "diff")
    disc_df = load_language_matrix(args.results_dir, langs, "disc")

    corr_diff = spearman_matrix(diff_df)
    corr_disc = spearman_matrix(disc_df)
    w_diff = kendalls_w(rank_df(diff_df).values)
    w_disc = kendalls_w(rank_df(disc_df).values)
    stab_diff = rank_df(diff_df).std(axis=1)
    stab_disc = rank_df(disc_df).std(axis=1)

    # --- Figure 1: Spearman heatmap — difficulty ---
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    im1 = plot_spearman_heatmap(corr_diff, w_diff, langs, "difficulty", ax1)
    fig1.colorbar(im1, ax=ax1, label="Spearman ρ", shrink=0.8)
    fig1.tight_layout()
    fig1.savefig(args.output_dir / "spearman_heatmap_diff.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("Saved spearman_heatmap_diff.png")

    # --- Figure 2: Spearman heatmap — discrimination ---
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    im2 = plot_spearman_heatmap(corr_disc, w_disc, langs, "discrimination", ax2)
    fig2.colorbar(im2, ax=ax2, label="Spearman ρ", shrink=0.8)
    fig2.tight_layout()
    fig2.savefig(args.output_dir / "spearman_heatmap_disc.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved spearman_heatmap_disc.png")

    # --- Figure 3: Kendall's W bar chart ---
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    param_labels = ["Difficulty", "Discrimination"]
    w_values = [w_diff, w_disc]
    bars = ax3.bar(param_labels, w_values, color="steelblue", edgecolor="white", width=0.4)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Kendall's W")
    ax3.set_title("MGSM 2PL — Cross-language item rank concordance (Kendall's W)")
    ax3.axhline(0.7, color="orange", linestyle="--", linewidth=1, label="W = 0.7 (strong)")
    ax3.axhline(0.5, color="red", linestyle="--", linewidth=1, label="W = 0.5 (moderate)")
    for bar, val in zip(bars, w_values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax3.legend(fontsize=8)
    fig3.tight_layout()
    fig3.savefig(args.output_dir / "kendalls_w.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("Saved kendalls_w.png")

    # --- Figure 4: Item rank stability (diff top, disc bottom) ---
    fig4a, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(10, 7))
    fig4a.suptitle("MGSM 2PL — Item rank stability across languages (lower = more consistent)", fontsize=11)
    plot_item_stability(stab_diff, "difficulty", ax4a)
    plot_item_stability(stab_disc, "discrimination", ax4b)
    fig4a.tight_layout()
    fig4a.savefig(args.output_dir / "item_rank_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig4a)
    print("Saved item_rank_stability.png")

    # --- Figure 5: Language dendrogram ---
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    plot_language_dendrogram(diff_df, disc_df, langs, ax5)
    fig5.tight_layout()
    fig5.savefig(args.output_dir / "language_dendrogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig5)
    print("Saved language_dendrogram.png")

    # --- CSVs ---
    pd.DataFrame({
        "n_languages": [len(langs)],
        "missing_languages": [",".join(missing)],
        "kendalls_w_diff": [w_diff],
        "kendalls_w_disc": [w_disc],
        "n_items": [len(diff_df)],
    }).to_csv(args.output_dir / "kendalls_w_summary.csv", index=False)
    print("Saved kendalls_w_summary.csv")

    rows = [
        {"item_id": item_id, "rank_std_diff": stab_diff[item_id], "rank_std_disc": stab_disc[item_id]}
        for item_id in stab_diff.index
    ]
    pd.DataFrame(rows).to_csv(args.output_dir / "item_rank_stability.csv", index=False)
    print("Saved item_rank_stability.csv")

    print(f"\nKendall's W — diff: {w_diff:.4f}  disc: {w_disc:.4f}  (n_items={len(diff_df)}, n_langs={len(langs)})")
    print("\n=== Mean pairwise Spearman ρ ===")
    n = len(langs)
    upper_diff = corr_diff.values[np.triu_indices(n, k=1)]
    upper_disc = corr_disc.values[np.triu_indices(n, k=1)]
    print(f"  diff — mean ρ = {upper_diff.mean():.4f}  min ρ = {upper_diff.min():.4f}")
    print(f"  disc — mean ρ = {upper_disc.mean():.4f}  min ρ = {upper_disc.min():.4f}")


if __name__ == "__main__":
    main()
