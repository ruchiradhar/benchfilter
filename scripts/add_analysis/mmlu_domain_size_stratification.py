#!/usr/bin/env python3
"""
Stratify MMLU cross-lingual concordance (Kendall's W) by domain size.

Splits domains into:
  large  (>=100 items): humanities (102), social_sciences (102)
  small  (<100 items):  business (58), other (56), stem (46), medical (36)

For each size group and IRT model variant (1PL, 2PL, 3PL), computes
Kendall's W and mean pairwise Spearman rho for difficulty (all models),
discrimination (2PL, 3PL), and guessing (3PL).

The key test: if discrimination instability persists in large domains,
it reflects genuine language-specificity, not estimation noise.

Outputs:
  stdout:                     W and rho table
  results/add_analysis/mmlu_domain_size_stratification/domain_size_stratification.png — bar chart of W by size group

Usage:
    python scripts/add_analysis/mmlu_domain_size_stratification.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
OUT_DIR = PROJECT_ROOT / "results" / "add_analysis" / "mmlu_domain_size_stratification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = ["de", "en", "es", "fr", "ja", "sw", "zh"]

LARGE_DOMAINS = ["humanities", "social_sciences"]          # 102 items each
SMALL_DOMAINS = ["business", "other", "stem", "medical"]   # 36–58 items

ITEM_COUNTS = {
    "humanities": 102, "social_sciences": 102,
    "business": 58, "other": 56, "stem": 46, "medical": 36,
}

MODEL_DIRS = {"1PL": "1PL", "2PL": "2PL_map", "3PL": "3PL_map"}
PARAMS = {"1PL": ["diff"], "2PL": ["diff", "disc"], "3PL": ["diff", "disc", "guess"]}
PARAM_KEY = {"diff": "diff", "disc": "disc", "guess": "lambdas"}


def load_param(model: str, lang: str, domain: str, param: str) -> pd.Series | None:
    suffix = model.lower()
    path = RESULTS / MODEL_DIRS[model] / f"mmlu_{lang}_{domain}_{suffix}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    key = PARAM_KEY[param]
    if key not in d:
        return None
    item_ids = [d["item_ids"][str(i)] for i in range(len(d[key]))]
    return pd.Series(d[key], index=item_ids)


def kendalls_w(rank_matrix: np.ndarray) -> float:
    n, m = rank_matrix.shape   # n items, m languages
    col_sums = rank_matrix.sum(axis=1)
    grand_mean = col_sums.mean()
    s = np.sum((col_sums - grand_mean) ** 2)
    return float(12 * s / (m ** 2 * (n ** 3 - n)))


def compute_concordance(domains: list[str], model: str, param: str) -> tuple[float, float, int]:
    """Return (mean Kendall's W, mean pairwise Spearman rho, n_domains) across given domains."""
    w_vals, rho_vals, n_ok = [], [], 0
    for domain in domains:
        series = {}
        for lang in LANGUAGES:
            s = load_param(model, lang, domain, param)
            if s is not None:
                series[lang] = s
        if len(series) < 4:
            continue
        # align on common items
        common = set.intersection(*[set(s.index) for s in series.values()])
        if len(common) < 5:
            continue
        mat = np.array([series[l].loc[sorted(common)].rank().values for l in series])
        # mat shape: (n_langs, n_items) — kendalls_w expects (n_items, n_langs)
        w = kendalls_w(mat.T)
        w_vals.append(w)
        # mean pairwise spearman
        langs = list(series.keys())
        rhos = []
        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                r, _ = spearmanr(
                    series[langs[i]].loc[sorted(common)],
                    series[langs[j]].loc[sorted(common)],
                )
                rhos.append(r)
        rho_vals.append(np.mean(rhos))
        n_ok += 1
    if not w_vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(w_vals)), float(np.mean(rho_vals)), n_ok


def main() -> None:
    rows = []
    for model in ["1PL", "2PL", "3PL"]:
        for param in PARAMS[model]:
            for group_name, domains in [("large (≥100)", LARGE_DOMAINS),
                                         ("small (<100)",  SMALL_DOMAINS)]:
                w, rho, n = compute_concordance(domains, model, param)
                rows.append({
                    "model": model, "param": param,
                    "group": group_name, "n_domains": n,
                    "W": w, "mean_rho": rho,
                })

    df = pd.DataFrame(rows)

    print("\n=== Kendall's W by domain size group ===")
    print(f"{'Model':>5}  {'Param':>5}  {'Group':>15}  {'n_domains':>9}  {'W':>6}  {'mean_rho':>9}")
    for _, r in df.iterrows():
        print(f"{r['model']:>5}  {r['param']:>5}  {r['group']:>15}  "
              f"{r['n_domains']:>9}  {r['W']:>6.3f}  {r['mean_rho']:>9.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    # One panel per (model, param), comparing large vs small
    combos = [(m, p) for m in ["1PL", "2PL", "3PL"] for p in PARAMS[m]]
    n_panels = len(combos)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.5 * n_panels, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]

    colors = {"large (≥100)": "#4c72b0", "small (<100)": "#dd8452"}

    for ax, (model, param) in zip(axes, combos):
        sub = df[(df["model"] == model) & (df["param"] == param)]
        groups = ["large (≥100)", "small (<100)"]
        ws = [sub.loc[sub["group"] == g, "W"].values[0] for g in groups]
        bars = ax.bar([0, 1], ws, color=[colors[g] for g in groups],
                      width=0.55, edgecolor="white")
        for bar, w_val in zip(bars, ws):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    w_val + 0.01, f"{w_val:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["large\n(≥100)", "small\n(<100)"], fontsize=8)
        ax.set_title(f"{model}\n{param}", fontsize=9)
        ax.set_ylim(0, 0.85)
        ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.7)

    axes[0].set_ylabel("Kendall's $W$", fontsize=10)
    fig.suptitle(
        "Cross-lingual concordance (Kendall's $W$) by MMLU domain size\n"
        "large = humanities + social sciences (102 items each); "
        "small = business, other, stem, medical (36–58 items)",
        fontsize=9,
    )

    # legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[g], label=g) for g in groups]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=8, bbox_to_anchor=(0.5, -0.05), frameon=False)

    fig.tight_layout()
    out_path = OUT_DIR / "domain_size_stratification.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")


if __name__ == "__main__":
    main()
