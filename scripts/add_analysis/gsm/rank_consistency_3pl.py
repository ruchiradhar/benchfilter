#!/usr/bin/env python3
"""Cross-language item rank consistency for MGSM 3PL IRT results.

Analyses difficulty (diff), discrimination (disc), and guessing (lambdas).
Missing languages are skipped automatically.

Examples:
    python scripts/analysis/gsm/rank_consistency_3pl.py
    python scripts/analysis/gsm/rank_consistency_3pl.py --results_dir results/3PL --output_dir results/add_analysis/gsm_3pl
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
        if (results_dir / f"mgsm_{lang}_3pl.json").exists()
    ]


def load_language_matrix(results_dir: Path, langs: list[str], param: str) -> pd.DataFrame:
    cols = {}
    for lang in langs:
        path = results_dir / f"mgsm_{lang}_3pl.json"
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
    missing = [l for l in ALL_LANGUAGES if l not in langs]
    note = f"\n(excl. {', '.join(LANG_LABELS[l] for l in missing)})" if missing else ""
    return im


def plot_item_stability(rank_std: pd.Series, param_label: str, ax: plt.Axes, ylim: float | None = None) -> None:
    sorted_std = rank_std.sort_values()
    ax.bar(range(len(sorted_std)), sorted_std.values, color="steelblue", width=1.0, linewidth=0)
    ax.set_xlabel("Item (sorted by stability)", fontsize=9)
    ax.set_ylabel("Rank std dev", fontsize=9)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    if ylim is not None:
        ax.set_ylim(0, ylim)


def plot_language_dendrogram(diff_df: pd.DataFrame, disc_df: pd.DataFrame,
                              guess_df: pd.DataFrame, langs: list[str], ax: plt.Axes) -> None:
    rk_diff  = diff_df.rank(axis=0, method="average")
    rk_disc  = disc_df.rank(axis=0, method="average")
    rk_guess = guess_df.rank(axis=0, method="average")
    profiles = {
        l: list(rk_diff[l].values) + list(rk_disc[l].values) + list(rk_guess[l].values)
        for l in langs
    }
    profile_matrix = np.array([profiles[l] for l in langs])
    dist = ssd.pdist(profile_matrix, metric="correlation")
    linkage = sch.linkage(dist, method="average")
    labels = [LANG_LABELS[l] for l in langs]
    sch.dendrogram(linkage, labels=labels, ax=ax, leaf_rotation=45, leaf_font_size=9,
                   color_threshold=0.3 * max(linkage[:, 2]))
    missing = [l for l in ALL_LANGUAGES if l not in langs]
    note = f"\n(excl. {', '.join(LANG_LABELS[l] for l in missing)})" if missing else ""
    ax.set_title(f"Language clustering (diff + disc + guessing rank profile, 3PL){note}", fontsize=9)
    ax.set_ylabel("Distance (1 − Spearman ρ)", fontsize=9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("results/3PL"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/add_analysis/gsm_3pl"))
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

    diff_df  = load_language_matrix(args.results_dir, langs, "diff")
    disc_df  = load_language_matrix(args.results_dir, langs, "disc")
    guess_df = load_language_matrix(args.results_dir, langs, "lambdas")

    corr_diff  = spearman_matrix(diff_df)
    corr_disc  = spearman_matrix(disc_df)
    corr_guess = spearman_matrix(guess_df)
    w_diff  = kendalls_w(rank_df(diff_df).values)
    w_disc  = kendalls_w(rank_df(disc_df).values)
    w_guess = kendalls_w(rank_df(guess_df).values)
    stab_diff  = rank_df(diff_df).std(axis=1)
    stab_disc  = rank_df(disc_df).std(axis=1)
    stab_guess = rank_df(guess_df).std(axis=1)

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

    # --- Figure 3: Spearman heatmap — guessing ---
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    im3 = plot_spearman_heatmap(corr_guess, w_guess, langs, "guessing", ax3)
    fig3.colorbar(im3, ax=ax3, label="Spearman ρ", shrink=0.8)
    fig3.tight_layout()
    fig3.savefig(args.output_dir / "spearman_heatmap_guess.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("Saved spearman_heatmap_guess.png")

    # --- Figure 4: Kendall's W bar chart ---
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    param_labels = ["Difficulty", "Discrimination", "Guessing"]
    w_values = [w_diff, w_disc, w_guess]
    bars = ax4.bar(param_labels, w_values, color="steelblue", edgecolor="white", width=0.4)
    ax4.set_ylim(0, 1)
    ax4.set_ylabel("Kendall's W")
    ax4.axhline(0.7, color="orange", linestyle="--", linewidth=1, label="W = 0.7 (strong)")
    ax4.axhline(0.5, color="red", linestyle="--", linewidth=1, label="W = 0.5 (moderate)")
    for bar, val in zip(bars, w_values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax4.legend(fontsize=8)
    fig4.tight_layout()
    fig4.savefig(args.output_dir / "kendalls_w.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print("Saved kendalls_w.png")

    # --- Figure 5: Item rank stability (3 rows) ---
    fig5, (ax5a, ax5b, ax5c) = plt.subplots(3, 1, figsize=(10, 10))
    shared_ylim = max(stab_diff.max(), stab_disc.max(), stab_guess.max()) * 1.05
    plot_item_stability(stab_diff,  "difficulty",      ax5a, ylim=shared_ylim)
    plot_item_stability(stab_disc,  "discrimination",  ax5b, ylim=shared_ylim)
    plot_item_stability(stab_guess, "guessing",        ax5c, ylim=shared_ylim)
    fig5.tight_layout()
    fig5.savefig(args.output_dir / "item_rank_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig5)
    print("Saved item_rank_stability.png")

    # --- Figure 6: Language dendrogram ---
    fig6, ax6 = plt.subplots(figsize=(10, 5))
    plot_language_dendrogram(diff_df, disc_df, guess_df, langs, ax6)
    fig6.tight_layout()
    fig6.savefig(args.output_dir / "language_dendrogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig6)
    print("Saved language_dendrogram.png")

    # --- CSVs ---
    pd.DataFrame({
        "n_languages":      [len(langs)],
        "missing_languages":[",".join(missing)],
        "kendalls_w_diff":  [w_diff],
        "kendalls_w_disc":  [w_disc],
        "kendalls_w_guess": [w_guess],
        "n_items":          [len(diff_df)],
    }).to_csv(args.output_dir / "kendalls_w_summary.csv", index=False)
    print("Saved kendalls_w_summary.csv")

    rows = [
        {
            "item_id":        item_id,
            "rank_std_diff":  stab_diff[item_id],
            "rank_std_disc":  stab_disc[item_id],
            "rank_std_guess": stab_guess[item_id],
        }
        for item_id in stab_diff.index
    ]
    pd.DataFrame(rows).to_csv(args.output_dir / "item_rank_stability.csv", index=False)
    print("Saved item_rank_stability.csv")

    print(f"\nKendall's W — diff: {w_diff:.4f}  disc: {w_disc:.4f}  guessing: {w_guess:.4f}"
          f"  (n_items={len(diff_df)}, n_langs={len(langs)})")
    print("\n=== Mean pairwise Spearman ρ ===")
    n = len(langs)
    upper_diff  = corr_diff.values[np.triu_indices(n, k=1)]
    upper_disc  = corr_disc.values[np.triu_indices(n, k=1)]
    upper_guess = corr_guess.values[np.triu_indices(n, k=1)]
    print(f"  diff    — mean ρ = {upper_diff.mean():.4f}  min ρ = {upper_diff.min():.4f}")
    print(f"  disc    — mean ρ = {upper_disc.mean():.4f}  min ρ = {upper_disc.min():.4f}")
    print(f"  guessing— mean ρ = {upper_guess.mean():.4f}  min ρ = {upper_guess.min():.4f}")


if __name__ == "__main__":
    main()
