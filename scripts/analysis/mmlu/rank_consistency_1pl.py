#!/usr/bin/env python3
"""Cross-language item rank consistency for MMLU 1PL IRT results.

For each MMLU domain, loads per-language 1PL difficulty parameters and computes:
  - Spearman rank correlation matrix across languages
  - Kendall's W (coefficient of concordance)
  - Per-item rank stability (std dev of rank across languages)
  - Hierarchical clustering of languages based on difficulty profiles

All outputs are written to --output_dir.

Examples:
    python scripts/analysis/mmlu/rank_consistency_1pl.py
    python scripts/analysis/mmlu/rank_consistency_1pl.py --results_dir results/1PL --output_dir results/analysis/mmlu_1pl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from scipy.stats import spearmanr

CMAP_RYG = LinearSegmentedColormap.from_list(
    "red_yellow_green",
    [
        (0.00, "#CC0000"),
        (0.50, "#FFFF00"),
        (1.00, "#007700"),
    ],
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LANGUAGES = ["de", "en", "es", "fr", "ja", "sw", "zh"]
DOMAINS = ["business", "humanities", "medical", "other", "social_sciences", "stem"]

LANG_LABELS = {
    "bn": "Bengali", "de": "German", "en": "English", "es": "Spanish",
    "fr": "French", "ja": "Japanese", "sw": "Swahili", "zh": "Chinese",
}


def resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_domain_matrix(results_dir: Path, domain: str) -> pd.DataFrame:
    """Return DataFrame (items × languages) of difficulty parameters."""
    cols = {}
    for lang in LANGUAGES:
        path = results_dir / f"mmlu_{lang}_{domain}_1pl.json"
        data = json.loads(path.read_text())
        item_ids = [data["item_ids"][str(i)] for i in range(len(data["diff"]))]
        cols[lang] = pd.Series(data["diff"], index=item_ids)
    return pd.DataFrame(cols)


def kendalls_w(rank_matrix: np.ndarray) -> float:
    """Kendall's W from a (items × raters) rank matrix."""
    n, m = rank_matrix.shape
    col_sums = rank_matrix.sum(axis=1)
    grand_mean = col_sums.mean()
    s = np.sum((col_sums - grand_mean) ** 2)
    w = 12 * s / (m ** 2 * (n ** 3 - n))
    return float(w)


def spearman_matrix(diff_df: pd.DataFrame) -> pd.DataFrame:
    """8×8 Spearman ρ matrix across languages."""
    langs = diff_df.columns.tolist()
    n = len(langs)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(diff_df.iloc[:, i], diff_df.iloc[:, j])
            mat[i, j] = mat[j, i] = rho
    return pd.DataFrame(mat, index=langs, columns=langs)


def rank_matrix(diff_df: pd.DataFrame) -> pd.DataFrame:
    """Rank items within each language (1 = easiest)."""
    return diff_df.rank(axis=0, method="average")


def plot_spearman_heatmap(corr_df: pd.DataFrame, domain: str, w: float, ax: plt.Axes) -> None:
    labels = [LANG_LABELS[l] for l in corr_df.columns]
    data = corr_df.values
    im = ax.imshow(data, vmin=-1, vmax=1, cmap=CMAP_RYG, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            text_color = "black" if abs(data[i, j]) < 0.5 else "white"
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color=text_color)
    ax.set_title(f"{domain.replace('_', ' ').title()}\nKendall's W = {w:.3f}", fontsize=9)
    return im


def plot_item_stability(rank_std: pd.Series, domain: str, ax: plt.Axes, ylim: float | None = None) -> None:
    sorted_std = rank_std.sort_values()
    ax.bar(range(len(sorted_std)), sorted_std.values, color="steelblue", width=1.0, linewidth=0)
    ax.set_xlabel("Item (sorted by stability)", fontsize=8)
    ax.set_ylabel("Rank std dev across languages", fontsize=8)
    ax.set_title(f"{domain.replace('_', ' ').title()}", fontsize=9)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    if ylim is not None:
        ax.set_ylim(0, ylim)


def plot_language_dendrogram(diff_matrices: dict[str, pd.DataFrame], ax: plt.Axes) -> None:
    """Cluster languages by concatenated difficulty rank profiles across all domains."""
    lang_profiles: dict[str, list[float]] = {lang: [] for lang in LANGUAGES}
    for domain, df in diff_matrices.items():
        rk = df.rank(axis=0, method="average")
        for lang in LANGUAGES:
            lang_profiles[lang].extend(rk[lang].tolist())

    profile_matrix = np.array([lang_profiles[l] for l in LANGUAGES])
    dist = ssd.pdist(profile_matrix, metric="correlation")
    linkage = sch.linkage(dist, method="average")
    labels = [LANG_LABELS[l] for l in LANGUAGES]
    sch.dendrogram(linkage, labels=labels, ax=ax, leaf_rotation=45, leaf_font_size=9,
                   color_threshold=0.3 * max(linkage[:, 2]))
    ax.set_title("Language clustering\n(by item difficulty rank profile)", fontsize=9)
    ax.set_ylabel("Distance (1 − Spearman ρ)", fontsize=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("results/1PL"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/analysis/mmlu_1pl"))
    args = parser.parse_args()
    args.results_dir = resolve_path(args.results_dir)
    args.output_dir = resolve_path(args.output_dir)
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    corr_matrices: dict[str, pd.DataFrame] = {}
    diff_matrices: dict[str, pd.DataFrame] = {}
    kendall_ws: dict[str, float] = {}
    stability: dict[str, pd.Series] = {}

    for domain in DOMAINS:
        diff_df = load_domain_matrix(args.results_dir, domain)
        diff_matrices[domain] = diff_df

        rk = rank_matrix(diff_df)
        corr = spearman_matrix(diff_df)
        w = kendalls_w(rk.values)

        corr_matrices[domain] = corr
        kendall_ws[domain] = w
        stability[domain] = rk.std(axis=1)

    # --- Figure 1: Spearman heatmaps (one per domain, 2×3 grid) ---
    fig1, axes = plt.subplots(2, 3, figsize=(16, 10))
    last_im = None
    for ax, domain in zip(axes.flat, DOMAINS):
        im = plot_spearman_heatmap(corr_matrices[domain], domain, kendall_ws[domain], ax)
        last_im = im
    fig1.colorbar(last_im, ax=axes, label="Spearman ρ", shrink=0.6, pad=0.02)
    fig1.tight_layout()
    fig1.savefig(args.output_dir / "spearman_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("Saved spearman_heatmaps.png")

    # --- Figure 1b: Average Spearman heatmap across all domains ---
    avg_corr_data = np.mean([corr_matrices[d].values for d in DOMAINS], axis=0)
    avg_corr = pd.DataFrame(avg_corr_data, index=LANGUAGES, columns=LANGUAGES)
    avg_w = float(np.mean([kendall_ws[d] for d in DOMAINS]))
    fig1b, ax1b = plt.subplots(figsize=(8, 7))
    labels_avg = [LANG_LABELS[l] for l in LANGUAGES]
    data_avg = avg_corr.values
    im1b = ax1b.imshow(data_avg, vmin=-1, vmax=1, cmap=CMAP_RYG, aspect="auto")
    ax1b.set_xticks(range(len(labels_avg)))
    ax1b.set_yticks(range(len(labels_avg)))
    ax1b.set_xticklabels(labels_avg, rotation=45, ha="right", fontsize=10)
    ax1b.set_yticklabels(labels_avg, fontsize=10)
    for i in range(len(labels_avg)):
        for j in range(len(labels_avg)):
            tc = "black" if abs(data_avg[i, j]) < 0.5 else "white"
            ax1b.text(j, i, f"{data_avg[i, j]:.2f}", ha="center", va="center", fontsize=7, color=tc)
    fig1b.colorbar(im1b, ax=ax1b, label="Spearman ρ", shrink=0.8)
    fig1b.tight_layout()
    fig1b.savefig(args.output_dir / "spearman_heatmap_avg.png", dpi=150, bbox_inches="tight")
    plt.close(fig1b)
    print("Saved spearman_heatmap_avg.png")

    # --- Figure 2: Kendall's W bar chart ---
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    domain_labels = [d.replace("_", " ").title() for d in DOMAINS]
    w_values = [kendall_ws[d] for d in DOMAINS]
    bars = ax2.bar(domain_labels, w_values, color="steelblue", edgecolor="white")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Kendall's W")
    ax2.axhline(0.7, color="orange", linestyle="--", linewidth=1, label="W = 0.7 (strong)")
    ax2.axhline(0.5, color="red", linestyle="--", linewidth=1, label="W = 0.5 (moderate)")
    for bar, w in zip(bars, w_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{w:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(args.output_dir / "kendalls_w.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved kendalls_w.png")

    # --- Figure 3: Item rank stability per domain (2×3 grid) ---
    fig3, axes3 = plt.subplots(2, 3, figsize=(16, 7))
    shared_ylim = max(stability[d].max() for d in DOMAINS) * 1.05
    for ax, domain in zip(axes3.flat, DOMAINS):
        plot_item_stability(stability[domain], domain, ax, ylim=shared_ylim)
    fig3.tight_layout()
    fig3.savefig(args.output_dir / "item_rank_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("Saved item_rank_stability.png")

    # --- Figure 4: Language dendrogram ---
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    plot_language_dendrogram(diff_matrices, ax4)
    fig4.tight_layout()
    fig4.savefig(args.output_dir / "language_dendrogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print("Saved language_dendrogram.png")

    # --- CSV: Kendall's W summary ---
    summary_df = pd.DataFrame({
        "domain": DOMAINS,
        "kendalls_w": [kendall_ws[d] for d in DOMAINS],
        "n_items": [len(diff_matrices[d]) for d in DOMAINS],
    })
    summary_df.to_csv(args.output_dir / "kendalls_w_summary.csv", index=False)
    print("Saved kendalls_w_summary.csv")

    # --- CSV: per-item rank std dev (all domains concatenated) ---
    rows = []
    for domain in DOMAINS:
        for item_id, std_val in stability[domain].items():
            rows.append({"domain": domain, "item_id": item_id, "rank_std": std_val})
    pd.DataFrame(rows).to_csv(args.output_dir / "item_rank_stability.csv", index=False)
    print("Saved item_rank_stability.csv")

    # --- Print summary to stdout ---
    print("\n=== Kendall's W summary ===")
    for d in DOMAINS:
        print(f"  {d:20s}  W = {kendall_ws[d]:.4f}  (n_items={len(diff_matrices[d])})")

    print("\n=== Mean pairwise Spearman ρ per domain ===")
    for d in DOMAINS:
        mat = corr_matrices[d].values
        upper = mat[np.triu_indices_from(mat, k=1)]
        print(f"  {d:20s}  mean ρ = {upper.mean():.4f}  min ρ = {upper.min():.4f}")


if __name__ == "__main__":
    main()
