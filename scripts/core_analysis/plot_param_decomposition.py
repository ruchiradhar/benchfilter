#!/usr/bin/env python3
"""Figure: parameter decomposition line plot.

Reads results/core_analysis/param_decomposition/summary.csv and plots median
Spearman rho across the 6 IIF selection strategies for MGSM and MMLU,
split by in-sample vs held-out. Two panels (MGSM | MMLU), two lines each.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH  = PROJECT_ROOT / "results" / "core_analysis" / "param_decomposition" / "summary.csv"
OUT_PATH  = PROJECT_ROOT / "images_2" / "param_decomposition.png"

STRATEGY_ORDER = [
    "1PL (diff)",
    "2PL diff-only",
    "2PL full (+ disc)",
    "3PL diff-only",
    "3PL diff+disc",
    "3PL full (+ guess)",
]

XLABELS = [
    "1PL\n(diff)",
    "2PL\ndiff only",
    "2PL\n+disc",
    "3PL\ndiff only",
    "3PL\n+disc",
    "3PL\n+guess",
]

COLORS = {"insample": "steelblue", "heldout": "darkorange"}
STYLES = {"insample": "-", "heldout": "--"}
LABELS = {"insample": "In-sample", "heldout": "Held-out"}
MARKERS = {"insample": "o", "heldout": "s"}


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    pivot = (
        df.groupby(["benchmark", "split", "strategy"])["spearman_rho"]
        .median()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, benchmark in zip(axes, ["MGSM", "MMLU"]):
        sub = pivot[pivot["benchmark"] == benchmark]
        for split in ["insample", "heldout"]:
            vals = (
                sub[sub["split"] == split]
                .set_index("strategy")
                .reindex(STRATEGY_ORDER)["spearman_rho"]
                .values
            )
            x = range(len(STRATEGY_ORDER))
            ax.plot(
                x, vals,
                color=COLORS[split],
                linestyle=STYLES[split],
                marker=MARKERS[split],
                markersize=6,
                linewidth=1.8,
                label=LABELS[split],
            )
            for xi, v in enumerate(vals):
                ax.annotate(
                    f"{v:.3f}", (xi, v),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center", fontsize=7, color=COLORS[split],
                )

        ax.set_xticks(range(len(STRATEGY_ORDER)))
        ax.set_xticklabels(XLABELS, fontsize=8)
        ax.set_title(benchmark, fontsize=12, fontweight="bold")
        ax.set_ylabel("Median Spearman $\\rho$", fontsize=9)
        ax.set_ylim(
            min(pivot[pivot["benchmark"] == benchmark]["spearman_rho"]) - 0.05,
            1.02,
        )
        ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # shade the regions to show parameter additions
        ax.axvspan(-0.5, 0.5, alpha=0.04, color="steelblue")   # diff only
        ax.axvspan(0.5,  2.5, alpha=0.04, color="darkorange")  # + disc
        ax.axvspan(2.5,  5.5, alpha=0.04, color="seagreen")    # 3PL

        for xpos, label in [(0, "diff"), (1.5, "+disc"), (4, "+guess")]:
            ax.text(xpos, ax.get_ylim()[0] + 0.005, label,
                    ha="center", fontsize=7, color="gray", style="italic")

    fig.suptitle(
        "Ranking fidelity by IIF selection strategy",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
