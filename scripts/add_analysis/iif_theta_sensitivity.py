#!/usr/bin/env python3
"""Sensitivity of IIF-based item selection to choice of theta.

Evaluates IIF at theta in {-1, 0, 1} for each IRT model variant and measures:
  (a) Jaccard similarity of selected subsets across theta values (within each model)
  (b) Whether the cross-model Jaccard ordering changes across theta values

Usage:
    python scripts/analysis/iif_theta_sensitivity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MGSM_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
MMLU_LANGUAGES = ["de", "en", "es", "fr", "ja", "sw", "zh"]
MMLU_DOMAINS = ["business", "humanities", "medical", "other", "social_sciences", "stem"]
THETAS = [-1.0, 0.0, 1.0]
TOP_FRAC = 0.5
EPS = 1e-8


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def iif_1pl(diff: np.ndarray, theta: float) -> np.ndarray:
    p = sigmoid(theta - diff)
    return p * (1.0 - p)


def iif_2pl(diff: np.ndarray, disc: np.ndarray, theta: float) -> np.ndarray:
    p = sigmoid(disc * (theta - diff))
    return disc ** 2 * p * (1.0 - p)


def iif_3pl(diff: np.ndarray, disc: np.ndarray, lam: np.ndarray, theta: float) -> np.ndarray:
    p_star = sigmoid(disc * (theta - diff))
    p = lam + (1.0 - lam) * p_star
    p = np.clip(p, EPS, 1.0 - EPS)
    return disc ** 2 * (p - lam) ** 2 * (1.0 - p) / ((1.0 - lam) ** 2 * p)


def load_params(results_root: Path, pl: str, stem: str) -> dict | None:
    path = results_root / pl / f"{stem}_{pl.lower()}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    n = len(data["diff"])
    return {
        "item_ids": [data["item_ids"][str(i)] for i in range(n)],
        "diff": np.array(data["diff"], dtype=float),
        "disc": np.array(data.get("disc", np.ones(n)), dtype=float),
        "lambdas": np.array(data.get("lambdas", np.zeros(n)), dtype=float),
    }


def top_set(info: np.ndarray, item_ids: list[str], frac: float) -> set[str]:
    k = max(1, int(round(len(item_ids) * frac)))
    return {item_ids[i] for i in np.argsort(info)[::-1][:k]}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def analyse_slice(results_root: Path, stem: str) -> dict | None:
    p1 = load_params(results_root, "1PL", stem)
    p2 = load_params(results_root, "2PL", stem)
    p3 = load_params(results_root, "3PL", stem)
    if p1 is None or p2 is None or p3 is None:
        return None

    # For each model, compute subset at each theta and measure cross-theta Jaccard
    rows = {}
    for label, params, iif_fn in [
        ("1PL", p1, lambda t: iif_1pl(p1["diff"], t)),
        ("2PL", p2, lambda t: iif_2pl(p2["diff"], p2["disc"], t)),
        ("3PL", p3, lambda t: iif_3pl(p3["diff"], p3["disc"], p3["lambdas"], t)),
    ]:
        sets = {t: top_set(iif_fn(t), params["item_ids"], TOP_FRAC) for t in THETAS}
        # Jaccard between each theta pair
        rows[f"{label}_m1_vs_0"] = jaccard(sets[-1.0], sets[0.0])
        rows[f"{label}_0_vs_p1"] = jaccard(sets[0.0], sets[1.0])
        rows[f"{label}_m1_vs_p1"] = jaccard(sets[-1.0], sets[1.0])

    # Also compute cross-model Jaccard at each theta (to check if ordering is stable)
    for t in THETAS:
        s1 = top_set(iif_1pl(p1["diff"], t), p1["item_ids"], TOP_FRAC)
        s2 = top_set(iif_2pl(p2["diff"], p2["disc"], t), p2["item_ids"], TOP_FRAC)
        s3 = top_set(iif_3pl(p3["diff"], p3["disc"], p3["lambdas"], t), p3["item_ids"], TOP_FRAC)
        tstr = str(int(t)) if t == int(t) else str(t)
        rows[f"cross_1v2_theta{tstr}"] = jaccard(s1, s2)
        rows[f"cross_1v3_theta{tstr}"] = jaccard(s1, s3)
        rows[f"cross_2v3_theta{tstr}"] = jaccard(s2, s3)

    return rows


def main() -> None:
    results_root = (PROJECT_ROOT / "results").resolve()
    all_rows = []

    for lang in MGSM_LANGUAGES:
        stem = f"mgsm_{lang}"
        r = analyse_slice(results_root, stem)
        if r:
            all_rows.append({"benchmark": "MGSM", "slice": stem, **r})

    for lang in MMLU_LANGUAGES:
        for domain in MMLU_DOMAINS:
            stem = f"mmlu_{lang}_{domain}"
            r = analyse_slice(results_root, stem)
            if r:
                all_rows.append({"benchmark": "MMLU", "slice": stem, **r})

    df = pd.DataFrame(all_rows)

    # --- Within-model theta sensitivity ---
    print("=== Within-model theta sensitivity (Jaccard of top-50% subsets across theta values) ===")
    print(f"{'Model':<6} {'theta_pair':<14} {'MGSM mean':>12} {'MGSM min':>10} {'MMLU mean':>12} {'MMLU min':>10}")
    print("-" * 60)
    for model in ["1PL", "2PL", "3PL"]:
        for pair_key, pair_label in [
            (f"{model}_m1_vs_0",   "θ=−1 vs θ=0"),
            (f"{model}_0_vs_p1",   "θ=0  vs θ=+1"),
            (f"{model}_m1_vs_p1",  "θ=−1 vs θ=+1"),
        ]:
            gsm = df[df["benchmark"] == "MGSM"][pair_key].dropna()
            mml = df[df["benchmark"] == "MMLU"][pair_key].dropna()
            print(f"{model:<6} {pair_label:<14} {gsm.mean():>12.3f} {gsm.min():>10.3f} {mml.mean():>12.3f} {mml.min():>10.3f}")
        print()

    # --- Cross-model Jaccard at each theta ---
    print("=== Cross-model IIF concordance by theta ===")
    print(f"{'Pair':<10} {'theta':>7} {'MGSM median':>13} {'MMLU median':>13}")
    print("-" * 46)
    for pair_key, pair_label in [
        ("cross_1v2", "1PL vs 2PL"),
        ("cross_1v3", "1PL vs 3PL"),
        ("cross_2v3", "2PL vs 3PL"),
    ]:
        for t in THETAS:
            tstr = str(int(t)) if t == int(t) else str(t)
            col = f"{pair_key}_theta{tstr}"
            gsm = df[df["benchmark"] == "MGSM"][col].dropna()
            mml = df[df["benchmark"] == "MMLU"][col].dropna()
            print(f"{pair_label:<10} {t:>7.1f} {gsm.median():>13.3f} {mml.median():>13.3f}")
        print()


if __name__ == "__main__":
    main()
