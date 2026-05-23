#!/usr/bin/env python3
"""Cross-language item rank consistency for MMLU 3PL IRT results.

Analyses difficulty (diff), discrimination (disc), and guessing (lambdas).
For each MMLU domain, computes:
  - Spearman rank correlation matrix across languages (per parameter)
  - Kendall's W per parameter
  - Per-item rank stability (std dev of rank across languages)
  - Hierarchical clustering of languages based on combined profiles
  - Average Spearman heatmap across all domains (per parameter)

Examples:
    python scripts/analysis/mmlu/rank_consistency_3pl.py
    python scripts/analysis/mmlu/rank_consistency_3pl.py --results_dir results/3PL --output_dir results/analysis_2/mmlu_3pl
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

LANGUAGES = ["de", "en", "es", "fr", "ja", "sw", "zh"]
DOMAINS = ["business", "humanities", "medical", "other", "social_sciences", "stem"]

LANG_LABELS = {
    "de": "German", "en": "English", "es": "Spanish", "fr": "French",
    "ja": "Japanese", "sw": "Swahili", "zh": "Chinese",
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


def load_domain_matrix(results_dir: Path, domain: str, param: str) -> pd.DataFrame:
    cols = {}
    for lang in LANGUAGES:
        path = results_dir / f"mmlu_{lang}_{domain}_3pl.json"
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


def plot_spearman_heatmap(corr_df: pd.DataFrame, domain: str, w: float,
                          ax: plt.Axes, param_label: str) -> object:
    labels = [LANG_LABELS[l] for l in corr_df.columns]
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
    ax.set_title(f"{domain.replace('_', ' ').title()} [{param_label}]\nW = {w:.3f}", fontsize=9)
    return im


def plot_item_stability(rank_std: pd.Series, domain: str, ax: plt.Axes, param_label: str, ylim: float | None = None) -> None:
    sorted_std = rank_std.sort_values()
    ax.bar(range(len(sorted_std)), sorted_std.values, color="steelblue", width=1.0, linewidth=0)
    ax.set_xlabel("Item (sorted)", fontsize=8)
    ax.set_ylabel("Rank std", fontsize=8)
    ax.set_title(f"{domain.replace('_', ' ').title()} [{param_label}]", fontsize=9)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    if ylim is not None:
        ax.set_ylim(0, ylim)


def plot_language_dendrogram(diff_matrices: dict, disc_matrices: dict,
                              guess_matrices: dict, ax: plt.Axes) -> None:
    lang_profiles: dict[str, list[float]] = {lang: [] for lang in LANGUAGES}
    for domain in DOMAINS:
        for matrices in [diff_matrices, disc_matrices, guess_matrices]:
            rk = matrices[domain].rank(axis=0, method="average")
            for lang in LANGUAGES:
                lang_profiles[lang].extend(rk[lang].tolist())
    profile_matrix = np.array([lang_profiles[l] for l in LANGUAGES])
    dist = ssd.pdist(profile_matrix, metric="correlation")
    linkage = sch.linkage(dist, method="average")
    labels = [LANG_LABELS[l] for l in LANGUAGES]
    sch.dendrogram(linkage, labels=labels, ax=ax, leaf_rotation=45, leaf_font_size=9,
                   color_threshold=0.3 * max(linkage[:, 2]))
    ax.set_title("Language clustering\n(diff + disc + guessing rank profile, all domains)", fontsize=9)
    ax.set_ylabel("Distance (1 − Spearman ρ)", fontsize=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("results/3PL_map"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/analysis_2/mmlu_3pl"))
    args = parser.parse_args()
    args.results_dir = resolve_path(args.results_dir)
    args.output_dir = resolve_path(args.output_dir)
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diff_matrices, disc_matrices, guess_matrices = {}, {}, {}
    corr_diff, corr_disc, corr_guess = {}, {}, {}
    w_diff, w_disc, w_guess = {}, {}, {}
    stab_diff, stab_disc, stab_guess = {}, {}, {}

    for domain in DOMAINS:
        diff_df  = load_domain_matrix(args.results_dir, domain, "diff")
        disc_df  = load_domain_matrix(args.results_dir, domain, "disc")
        guess_df = load_domain_matrix(args.results_dir, domain, "lambdas")

        diff_matrices[domain]  = diff_df
        disc_matrices[domain]  = disc_df
        guess_matrices[domain] = guess_df

        rk_diff  = rank_df(diff_df)
        rk_disc  = rank_df(disc_df)
        rk_guess = rank_df(guess_df)

        corr_diff[domain]  = spearman_matrix(diff_df)
        corr_disc[domain]  = spearman_matrix(disc_df)
        corr_guess[domain] = spearman_matrix(guess_df)
        w_diff[domain]  = kendalls_w(rk_diff.values)
        w_disc[domain]  = kendalls_w(rk_disc.values)
        w_guess[domain] = kendalls_w(rk_guess.values)
        stab_diff[domain]  = rk_diff.std(axis=1)
        stab_disc[domain]  = rk_disc.std(axis=1)
        stab_guess[domain] = rk_guess.std(axis=1)

    # --- Figure 1: Spearman heatmaps — difficulty ---
    fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
    last_im = None
    for ax, domain in zip(axes1.flat, DOMAINS):
        im = plot_spearman_heatmap(corr_diff[domain], domain, w_diff[domain], ax, "difficulty")
        last_im = im
    fig1.colorbar(last_im, ax=axes1, label="Spearman ρ", shrink=0.6, pad=0.02)
    fig1.tight_layout()
    fig1.savefig(args.output_dir / "spearman_heatmaps_diff.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("Saved spearman_heatmaps_diff.png")

    # --- Figure 2: Spearman heatmaps — discrimination ---
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
    last_im = None
    for ax, domain in zip(axes2.flat, DOMAINS):
        im = plot_spearman_heatmap(corr_disc[domain], domain, w_disc[domain], ax, "discrimination")
        last_im = im
    fig2.colorbar(last_im, ax=axes2, label="Spearman ρ", shrink=0.6, pad=0.02)
    fig2.tight_layout()
    fig2.savefig(args.output_dir / "spearman_heatmaps_disc.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved spearman_heatmaps_disc.png")

    # --- Figure 3: Spearman heatmaps — guessing ---
    fig3, axes3 = plt.subplots(2, 3, figsize=(16, 10))
    last_im = None
    for ax, domain in zip(axes3.flat, DOMAINS):
        im = plot_spearman_heatmap(corr_guess[domain], domain, w_guess[domain], ax, "guessing")
        last_im = im
    fig3.colorbar(last_im, ax=axes3, label="Spearman ρ", shrink=0.6, pad=0.02)
    fig3.tight_layout()
    fig3.savefig(args.output_dir / "spearman_heatmaps_guess.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("Saved spearman_heatmaps_guess.png")

    # --- Figures 1b/2b/3b: Average heatmaps across all domains ---
    labels_avg = [LANG_LABELS[l] for l in LANGUAGES]
    for param_label, corr_dict, w_dict, fname in [
        ("difficulty",     corr_diff,  w_diff,  "spearman_heatmap_avg_diff.png"),
        ("discrimination", corr_disc,  w_disc,  "spearman_heatmap_avg_disc.png"),
        ("guessing",       corr_guess, w_guess, "spearman_heatmap_avg_guess.png"),
    ]:
        avg_data  = np.mean([corr_dict[d].values for d in DOMAINS], axis=0)
        avg_w_val = float(np.mean([w_dict[d] for d in DOMAINS]))
        figA, axA = plt.subplots(figsize=(8, 7))
        imA = axA.imshow(avg_data, vmin=-1, vmax=1, cmap=CMAP_RYG, aspect="auto")
        axA.set_xticks(range(len(labels_avg)))
        axA.set_yticks(range(len(labels_avg)))
        axA.set_xticklabels(labels_avg, rotation=45, ha="right", fontsize=10)
        axA.set_yticklabels(labels_avg, fontsize=10)
        for i in range(len(labels_avg)):
            for j in range(len(labels_avg)):
                tc = "black" if abs(avg_data[i, j]) < 0.5 else "white"
                axA.text(j, i, f"{avg_data[i, j]:.2f}", ha="center", va="center", fontsize=7, color=tc)
        figA.colorbar(imA, ax=axA, label="Spearman ρ", shrink=0.8)
        figA.tight_layout()
        figA.savefig(args.output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(figA)
        print(f"Saved {fname}")

    # --- Figure 4: Kendall's W grouped bar chart ---
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    domain_labels = [d.replace("_", " ").title() for d in DOMAINS]
    x = np.arange(len(DOMAINS))
    width = 0.25
    bars1 = ax4.bar(x - width, [w_diff[d]  for d in DOMAINS], width, label="Difficulty",      color="steelblue")
    bars2 = ax4.bar(x,         [w_disc[d]  for d in DOMAINS], width, label="Discrimination",  color="darkorange")
    bars3 = ax4.bar(x + width, [w_guess[d] for d in DOMAINS], width, label="Guessing",        color="forestgreen")
    ax4.set_xticks(x)
    ax4.set_xticklabels(domain_labels)
    ax4.set_ylim(0, 1)
    ax4.set_ylabel("Kendall's W")
    for bar in list(bars1) + list(bars2) + list(bars3):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    fig4.tight_layout()
    fig4.savefig(args.output_dir / "kendalls_w.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print("Saved kendalls_w.png")

    # --- Figure 5: Item rank stability (3 rows × 6 cols) ---
    fig5, axes5 = plt.subplots(3, 6, figsize=(22, 9), sharey=False)
    shared_ylim = max(
        max(stab_diff[d].max() for d in DOMAINS),
        max(stab_disc[d].max() for d in DOMAINS),
        max(stab_guess[d].max() for d in DOMAINS),
    ) * 1.05
    for col, domain in enumerate(DOMAINS):
        plot_item_stability(stab_diff[domain],  domain, axes5[0, col], "diff",     ylim=shared_ylim)
        plot_item_stability(stab_disc[domain],  domain, axes5[1, col], "disc",     ylim=shared_ylim)
        plot_item_stability(stab_guess[domain], domain, axes5[2, col], "guessing", ylim=shared_ylim)
    axes5[0, 0].set_ylabel("Rank std (difficulty)",      fontsize=8)
    axes5[1, 0].set_ylabel("Rank std (discrimination)",  fontsize=8)
    axes5[2, 0].set_ylabel("Rank std (guessing)",        fontsize=8)
    fig5.tight_layout()
    fig5.savefig(args.output_dir / "item_rank_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig5)
    print("Saved item_rank_stability.png")

    # --- Figure 6: Language dendrogram ---
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    plot_language_dendrogram(diff_matrices, disc_matrices, guess_matrices, ax6)
    fig6.tight_layout()
    fig6.savefig(args.output_dir / "language_dendrogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig6)
    print("Saved language_dendrogram.png")

    # --- CSVs ---
    pd.DataFrame({
        "domain":            DOMAINS,
        "kendalls_w_diff":   [w_diff[d]  for d in DOMAINS],
        "kendalls_w_disc":   [w_disc[d]  for d in DOMAINS],
        "kendalls_w_guess":  [w_guess[d] for d in DOMAINS],
        "n_items":           [len(diff_matrices[d]) for d in DOMAINS],
    }).to_csv(args.output_dir / "kendalls_w_summary.csv", index=False)
    print("Saved kendalls_w_summary.csv")

    rows = []
    for domain in DOMAINS:
        for item_id in stab_diff[domain].index:
            rows.append({
                "domain":          domain,
                "item_id":         item_id,
                "rank_std_diff":   stab_diff[domain][item_id],
                "rank_std_disc":   stab_disc[domain][item_id],
                "rank_std_guess":  stab_guess[domain][item_id],
            })
    pd.DataFrame(rows).to_csv(args.output_dir / "item_rank_stability.csv", index=False)
    print("Saved item_rank_stability.csv")

    print("\n=== Kendall's W summary ===")
    print(f"  {'domain':20s}  {'W_diff':>8}  {'W_disc':>8}  {'W_guess':>8}")
    for d in DOMAINS:
        print(f"  {d:20s}  {w_diff[d]:8.4f}  {w_disc[d]:8.4f}  {w_guess[d]:8.4f}")

    print("\n=== Mean pairwise Spearman ρ ===")
    print(f"  {'domain':20s}  {'mean_diff':>10}  {'mean_disc':>10}  {'mean_guess':>10}")
    for d in DOMAINS:
        upper_diff  = corr_diff[d].values[np.triu_indices(len(LANGUAGES), k=1)]
        upper_disc  = corr_disc[d].values[np.triu_indices(len(LANGUAGES), k=1)]
        upper_guess = corr_guess[d].values[np.triu_indices(len(LANGUAGES), k=1)]
        print(f"  {d:20s}  {upper_diff.mean():10.4f}  {upper_disc.mean():10.4f}  {upper_guess.mean():10.4f}")


if __name__ == "__main__":
    main()
