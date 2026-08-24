#!/usr/bin/env python3
"""Figure: DIF rate and mean anchor-corrected difficulty shift by language.

Reads results/core_analysis/dif_anchor/dif_anchor_summary.csv and plots, for MGSM and MMLU
(languages averaged across domains), the percentage of items flagged as
showing significant DIF (bars, left axis) alongside the mean anchor-corrected
|delta_b| shift (line+markers, right axis), sorted ascending by % DIF.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "results" / "core_analysis" / "dif_anchor" / "dif_anchor_summary.csv"
OUT_PATH = PROJECT_ROOT / "images_2" / "dif_summary.png"

BAR_COLOR = "steelblue"
LINE_COLOR = "darkorange"


def panel(ax, sub: pd.DataFrame, title: str, mean_pct: float):
    sub = sub.sort_values("pct_dif")
    x = range(len(sub))

    bars = ax.bar(x, sub["pct_dif"], color=BAR_COLOR, alpha=0.85, width=0.6, zorder=2)
    ax.axhline(mean_pct, color=BAR_COLOR, linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(len(sub) - 0.5, mean_pct + 1.5, f"mean = {mean_pct:.1f}%",
            ha="right", fontsize=7, color=BAR_COLOR)

    for xi, v in zip(x, sub["pct_dif"]):
        ax.annotate(f"{v:.0f}", (xi, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7, color=BAR_COLOR)

    ax.set_xticks(list(x))
    ax.set_xticklabels(sub["focal"], fontsize=8)
    ax.set_ylabel("% items with significant DIF", fontsize=9, color=BAR_COLOR)
    ax.tick_params(axis="y", labelcolor=BAR_COLOR)
    ax.set_ylim(0, max(sub["pct_dif"].max(), mean_pct) * 1.25)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(x, sub["mean_abs_delta"], color=LINE_COLOR, marker="o",
              markersize=5, linewidth=1.6, zorder=3)
    ax2.set_ylabel("mean $|\\Delta b|$ (logits)", fontsize=9, color=LINE_COLOR)
    ax2.tick_params(axis="y", labelcolor=LINE_COLOR)
    ax2.set_ylim(0, sub["mean_abs_delta"].max() * 1.3)
    ax2.spines["top"].set_visible(False)


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    mgsm = df[df["benchmark"] == "MGSM"].copy()
    mmlu_lang = (
        df[df["benchmark"] == "MMLU"]
        .groupby("focal")
        .agg(pct_dif=("pct_dif", "mean"), mean_abs_delta=("mean_abs_delta", "mean"))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    panel(axes[0], mgsm, "MGSM", mgsm["pct_dif"].mean())
    panel(axes[1], mmlu_lang, "MMLU (avg. over domains)", mmlu_lang["pct_dif"].mean())

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
