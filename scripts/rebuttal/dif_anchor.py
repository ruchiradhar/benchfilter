#!/usr/bin/env python3
"""Anchor-item DIF analysis for MGSM and MMLU using fitted 1PL MAP parameters.

For each benchmark language vs English (reference), computes anchor-corrected
difficulty differences and tests significance using approximate Fisher-information
SEs. Anchor items = lowest-rank-SD quartile (most cross-linguistically stable
items from the existing rank-consistency analysis).

No re-fitting required: uses already-fitted MAP parameters from results/1PL_map/.
The anchor set handles ability-scale equating, so no equal-ability assumption
is needed (addresses IRT-LR DIF reviewer concern).

Method (approximation to IRT-LR anchor-item DIF):
  1. Anchor set A = bottom 25% of items by cross-lingual rank SD (per domain for MMLU)
  2. For each language pair (en, focal):
       raw_diff_j   = b_j_focal - b_j_en
       scale_corr   = mean(raw_diff_j for j in A)   [anchor equating]
       delta_j      = raw_diff_j - scale_corr        [DIF statistic]
       SE(b_j)      = 1 / sqrt(I(b_j))               [Fisher info]
                      where I(b_j) = sum_i sigmoid(theta_i - b_j)(1 - sigmoid(...))
       SE(delta_j)  = sqrt(SE_en_j^2 + SE_focal_j^2)
       z_j          = delta_j / SE(delta_j)
  3. Two-sided p-values, BH-FDR at q=0.05

Usage:
    python scripts/rebuttal/dif_anchor.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MGSM_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
MGSM_FOCAL = [l for l in MGSM_LANGUAGES if l != "en"]

MMLU_LANGUAGES = ["de", "en", "es", "fr", "ja", "sw", "zh"]
MMLU_FOCAL = [l for l in MMLU_LANGUAGES if l != "en"]
MMLU_DOMAINS = ["business", "humanities", "medical", "other", "social_sciences", "stem"]

ANCHOR_QUANTILE = 0.25


def load_params(path: Path) -> dict:
    data = json.loads(path.read_text())
    item_ids = [data["item_ids"][str(i)] for i in range(len(data["diff"]))]
    return {
        "theta": np.array(data["ability"]),
        "b": pd.Series(data["diff"], index=item_ids),
    }


def fisher_se(theta: np.ndarray, b: pd.Series) -> pd.Series:
    se_vals = {}
    for item, b_j in b.items():
        p = expit(theta - b_j)
        info = np.sum(p * (1 - p))
        se_vals[item] = 1.0 / np.sqrt(info) if info > 0 else np.nan
    return pd.Series(se_vals)


def bh_fdr(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    n = len(p_values)
    order = np.argsort(p_values)
    ranks = np.argsort(order) + 1
    threshold = (ranks / n) * q
    reject = p_values <= threshold
    if reject.any():
        last = np.max(ranks[reject])
        reject = ranks <= last
    return reject


def run_dif(ref_params: dict, foc_params: dict, anchor_items: set) -> dict:
    common = ref_params["b"].index.intersection(foc_params["b"].index)
    b_ref = ref_params["b"].loc[common]
    b_foc = foc_params["b"].loc[common]
    se_ref = fisher_se(ref_params["theta"], b_ref)
    se_foc = fisher_se(foc_params["theta"], b_foc)

    raw_diff = b_foc - b_ref
    anchor_common = [i for i in anchor_items if i in common]
    scale_corr = raw_diff.loc[anchor_common].mean()
    delta = raw_diff - scale_corr

    non_anchor = [i for i in common if i not in anchor_items]
    delta_na = delta.loc[non_anchor]
    se_j = np.sqrt(se_ref.loc[non_anchor].values ** 2 + se_foc.loc[non_anchor].values ** 2)
    z = delta_na.values / se_j
    p = 2 * norm.sf(np.abs(z))
    reject = bh_fdr(p, q=0.05)

    return {
        "n_anchor": len(anchor_common),
        "n_tested": len(non_anchor),
        "n_dif": int(reject.sum()),
        "pct_dif": 100 * reject.sum() / len(non_anchor),
        "mean_abs_delta": float(np.abs(delta_na.values).mean()),
        "mean_abs_delta_dif": float(np.abs(delta_na.values[reject]).mean()) if reject.any() else np.nan,
        "scale_correction": float(scale_corr),
    }


def main() -> None:
    output_dir = PROJECT_ROOT / "results" / "rebuttal"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    # ── MGSM ──────────────────────────────────────────────────────────────────
    print("=== MGSM ===")
    stab_mgsm = pd.read_csv(
        PROJECT_ROOT / "results" / "analysis_2" / "gsm_1pl" / "item_rank_stability.csv",
        index_col="item_id",
    )
    thresh_mgsm = stab_mgsm["rank_std"].quantile(ANCHOR_QUANTILE)
    anchor_mgsm = set(stab_mgsm.index[stab_mgsm["rank_std"] <= thresh_mgsm])
    print(f"Anchor items (rank_std ≤ {thresh_mgsm:.2f}): {len(anchor_mgsm)}")

    ref_mgsm = load_params(PROJECT_ROOT / "results" / "1PL_map" / "mgsm_en_1pl.json")

    for lang in MGSM_FOCAL:
        foc = load_params(PROJECT_ROOT / "results" / "1PL_map" / f"mgsm_{lang}_1pl.json")
        r = run_dif(ref_mgsm, foc, anchor_mgsm)
        print(f"  en vs {lang}: n_tested={r['n_tested']}, DIF={r['n_dif']} ({r['pct_dif']:.1f}%), "
              f"mean|Δb|={r['mean_abs_delta']:.3f}, mean|Δb|_DIF={r['mean_abs_delta_dif']:.3f}")
        summary_rows.append({"benchmark": "MGSM", "domain": "-", "focal": lang, **r})

    # ── MMLU ──────────────────────────────────────────────────────────────────
    print("\n=== MMLU ===")
    stab_mmlu = pd.read_csv(
        PROJECT_ROOT / "results" / "analysis_2" / "mmlu_1pl" / "item_rank_stability.csv",
    )

    for domain in MMLU_DOMAINS:
        stab_d = stab_mmlu[stab_mmlu["domain"] == domain].set_index("item_id")
        thresh_d = stab_d["rank_std"].quantile(ANCHOR_QUANTILE)
        anchor_d = set(stab_d.index[stab_d["rank_std"] <= thresh_d])

        ref_path = PROJECT_ROOT / "results" / "1PL_map" / f"mmlu_en_{domain}_1pl.json"
        if not ref_path.exists():
            print(f"  skipping {domain} (missing en reference)")
            continue
        ref_mmlu = load_params(ref_path)

        for lang in MMLU_FOCAL:
            foc_path = PROJECT_ROOT / "results" / "1PL_map" / f"mmlu_{lang}_{domain}_1pl.json"
            if not foc_path.exists():
                continue
            foc = load_params(foc_path)
            r = run_dif(ref_mmlu, foc, anchor_d)
            print(f"  en vs {lang} [{domain}]: n_tested={r['n_tested']}, DIF={r['n_dif']} ({r['pct_dif']:.1f}%), "
                  f"mean|Δb|={r['mean_abs_delta']:.3f}")
            summary_rows.append({"benchmark": "MMLU", "domain": domain, "focal": lang, **r})

    df = pd.DataFrame(summary_rows)
    df.to_csv(output_dir / "dif_anchor_summary.csv", index=False)
    print(f"\nSaved dif_anchor_summary.csv to {output_dir}")

    print("\n=== Summary by benchmark ===")
    for bench in ["MGSM", "MMLU"]:
        sub = df[df["benchmark"] == bench]
        print(f"{bench}: mean % DIF = {sub['pct_dif'].mean():.1f}%  "
              f"mean |Δb| = {sub['mean_abs_delta'].mean():.3f}  "
              f"(n={len(sub)} language pairs)")

    print("\n=== MMLU summary by domain ===")
    mmlu = df[df["benchmark"] == "MMLU"]
    for domain in MMLU_DOMAINS:
        sub = mmlu[mmlu["domain"] == domain]
        if len(sub):
            print(f"  {domain}: mean % DIF = {sub['pct_dif'].mean():.1f}%  "
                  f"mean |Δb| = {sub['mean_abs_delta'].mean():.3f}")


if __name__ == "__main__":
    main()
