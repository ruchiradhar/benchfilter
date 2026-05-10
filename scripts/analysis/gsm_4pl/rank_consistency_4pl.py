#!/usr/bin/env python3
"""Cross-language item rank consistency for MGSM 4PL IRT results.

Analyses difficulty (diff), discrimination (disc), and upper-asymptote (lambdas).
Missing languages are skipped automatically.

Examples:
    python scripts/analysis/gsm_4pl/rank_consistency_4pl.py
    python scripts/analysis/gsm_4pl/rank_consistency_4pl.py --results_dir results/4PL --output_dir results/analysis/gsm_4pl
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

ALL_LANGUAGES = ["bn", "de", "en", "es", "es_spanish_bench", "fr", "ja", "ru", "sw", "te", "th", "zh"]

LANG_LABELS = {
    "bn": "Bengali", "de": "German", "en": "English", "es": "Spanish",
    "es_spanish_bench": "Spanish (bench)", "fr": "French", "ja": "Japanese",
    "ru": "Russian", "sw": "Swahili", "te": "Telugu", "th": "Thai", "zh": "Chinese",
}

CMAP_RY = LinearSegmentedColormap.from_list(
    "red_yellow",
    [
        (0.00, "#8B0000"),
        (0.20, "#CC2200"),
        (0.40, "#DD6600"),
        (0.60, "#EE9900"),
        (0.80, "#FFCC00"),
        (1.00, "#FFE800"),
    ],
)


def resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def available_languages(results_dir: Path) -> list[str]:
    return [
        lang for lang in ALL_LANGUAGES
        if (results_dir / f"mgsm_{lang}_4pl.json").exists()
    ]


def load_language_matrix(results_dir: Path, langs: list[str], param: str) -> pd.DataFrame:
    cols = {}
    for lang in langs:
        path = results_dir / f"mgsm_{lang}_4pl.json"
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
    im = ax.imshow(data, vmin=0, vmax=1, cmap=CMAP_RY, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "black" if data[i, j] > 0.5 else "white"
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=color)
    missing = [l for l in ALL_LANGUAGES if l not in langs]
    note = f"\n(excl. {', '.join(LANG_LABELS[l] for l in missing)})" if missing else ""
    ax.set_title(f"MGSM 4PL — Item {param_label} rank correlation{note}\nKendall's W = {w:.3f}", fontsize=9)
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
    missing = [l for l in ALL_LANGUAGES if l not in langs]
    note = f"\n(excl. {', '.join(LANG_LABELS[l] for l in missing)})" if missing else ""
    ax.set_title(f"Language clustering (diff + disc rank profile, 4PL){note}", fontsize=9)
    ax.set_ylabel("Distance (1 − Spearman ρ)", fontsize=9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("results/4PL"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/analysis/gsm_4pl"))
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
    lam_df  = load_language_matrix(args.results_dir, langs, "lambdas")

    corr_diff = spearman_matrix(diff_df)
    corr_disc = spearman_matrix(disc_df)
    corr_lam  = spearman_matrix(lam_df)
    w_diff = kendalls_w(rank_df(diff_df).values)
    w_disc = kendalls_w(rank_df(disc_df).values)
    w_lam  = kendalls_w(rank_df(lam_df).values)
    stab_diff = rank_df(diff_df).std(axis=1)
    stab_disc = rank_df(disc_df).std(axis=1)
    stab_lam  = rank_df(lam_df).std(axis=1)

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

    # --- Figure 3: Spearman heatmap — upper asymptote (λ) ---
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    im3 = plot_spearman_heatmap(corr_lam, w_lam, langs, "upper asymptote (λ)", ax3)
    fig3.colorbar(im3, ax=ax3, label="Spearman ρ", shrink=0.8)
    fig3.tight_layout()
    fig3.savefig(args.output_dir / "spearman_heatmap_lambda.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("Saved spearman_heatmap_lambda.png")

    # --- Figure 4: Item rank stability (3 rows: diff, disc, lambda) ---
    fig4, (ax4a, ax4b, ax4c) = plt.subplots(3, 1, figsize=(10, 10))
    fig4.suptitle("MGSM 4PL — Item rank stability across languages (lower = more consistent)", fontsize=11)
    plot_item_stability(stab_diff, "difficulty", ax4a)
    plot_item_stability(stab_disc, "discrimination", ax4b)
    plot_item_stability(stab_lam, "λ", ax4c)
    fig4.tight_layout()
    fig4.savefig(args.output_dir / "item_rank_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
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
        "kendalls_w_lambda": [w_lam],
        "n_items": [len(diff_df)],
    }).to_csv(args.output_dir / "kendalls_w_summary.csv", index=False)
    print("Saved kendalls_w_summary.csv")

    rows = [
        {
            "item_id": item_id,
            "rank_std_diff": stab_diff[item_id],
            "rank_std_disc": stab_disc[item_id],
            "rank_std_lambda": stab_lam[item_id],
        }
        for item_id in stab_diff.index
    ]
    pd.DataFrame(rows).to_csv(args.output_dir / "item_rank_stability.csv", index=False)
    print("Saved item_rank_stability.csv")

    print(f"\nKendall's W — diff: {w_diff:.4f}  disc: {w_disc:.4f}  λ: {w_lam:.4f}  (n_items={len(diff_df)}, n_langs={len(langs)})")
    print("\n=== Mean pairwise Spearman ρ ===")
    n = len(langs)
    upper_diff = corr_diff.values[np.triu_indices(n, k=1)]
    upper_disc = corr_disc.values[np.triu_indices(n, k=1)]
    upper_lam  = corr_lam.values[np.triu_indices(n, k=1)]
    print(f"  diff — mean ρ = {upper_diff.mean():.4f}  min ρ = {upper_diff.min():.4f}")
    print(f"  disc — mean ρ = {upper_disc.mean():.4f}  min ρ = {upper_disc.min():.4f}")
    print(f"  λ    — mean ρ = {upper_lam.mean():.4f}  min ρ = {upper_lam.min():.4f}")


if __name__ == "__main__":
    main()
