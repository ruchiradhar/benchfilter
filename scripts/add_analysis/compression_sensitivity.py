#!/usr/bin/env python3
"""Sensitivity of ranking fidelity to compression ratio (top_frac).

Uses existing fitted IRT parameters (no retraining). Computes in-sample
Spearman rho between subset accuracy and full-benchmark accuracy at
top_frac in {0.3, 0.5, 0.7} for each IRT model and benchmark.

Usage:
    python scripts/analysis/compression_sensitivity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MGSM_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
MMLU_LANGUAGES = ["de", "en", "es", "fr", "ja", "sw", "zh"]
MMLU_DOMAINS = ["business", "humanities", "medical", "other", "social_sciences", "stem"]
TOP_FRACS = [0.3, 0.5, 0.7]
EPS = 1e-8


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def iif_1pl(diff):
    p = sigmoid(-diff)
    return p * (1.0 - p)


def iif_2pl(diff, disc):
    p = sigmoid(disc * (-diff))
    return disc ** 2 * p * (1.0 - p)


def iif_3pl(diff, disc, lam):
    p_star = sigmoid(disc * (-diff))
    p = lam + (1.0 - lam) * p_star
    p = np.clip(p, EPS, 1.0 - EPS)
    lam_c = np.clip(lam, 0.0, 1.0 - EPS)
    return disc ** 2 * (p - lam_c) ** 2 * (1.0 - p) / ((1.0 - lam_c) ** 2 * p)


def load_params(results_root, pl, stem):
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


def compute_iif(params, pl):
    if pl == "1PL":
        return iif_1pl(params["diff"])
    elif pl == "2PL":
        return iif_2pl(params["diff"], params["disc"])
    else:
        return iif_3pl(params["diff"], params["disc"], params["lambdas"])


def load_responses(data_root, stem):
    path = data_root / f"{stem}.jsonlines"
    if not path.exists():
        return None
    responses = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                responses[obj["subject_id"]] = obj["responses"]
    return responses


def model_accs(responses, item_ids):
    item_set = set(item_ids)
    return {
        sid: float(np.mean([r for iid, r in resp.items() if iid in item_set]))
        for sid, resp in responses.items()
        if any(iid in item_set for iid in resp)
    }


def spearman_rho(acc_full, acc_sub):
    common = sorted(acc_full.keys() & acc_sub.keys())
    if len(common) < 3:
        return None
    a = np.array([acc_full[m] for m in common])
    b = np.array([acc_sub[m] for m in common])
    if np.std(a) < EPS or np.std(b) < EPS:
        return None
    rho, _ = spearmanr(a, b)
    return float(rho)


def main():
    results_root = (PROJECT_ROOT / "results").resolve()
    data_root = (PROJECT_ROOT / "data" / "irt").resolve()

    slices = (
        [("MGSM", f"mgsm_{lang}") for lang in MGSM_LANGUAGES]
        + [("MMLU", f"mmlu_{lang}_{domain}") for lang in MMLU_LANGUAGES for domain in MMLU_DOMAINS]
    )

    rows = []
    for benchmark, stem in slices:
        responses = load_responses(data_root, stem)
        if responses is None:
            continue

        for pl in ["1PL", "2PL", "3PL"]:
            params = load_params(results_root, pl, stem)
            if params is None:
                continue

            info = compute_iif(params, pl)
            all_ids = params["item_ids"]
            acc_full = model_accs(responses, all_ids)

            for top_frac in TOP_FRACS:
                k = max(1, int(round(len(all_ids) * top_frac)))
                top_ids = [all_ids[i] for i in np.argsort(info)[::-1][:k]]
                acc_sub = model_accs(responses, top_ids)
                rho = spearman_rho(acc_full, acc_sub)
                rows.append({
                    "benchmark": benchmark,
                    "slice": stem,
                    "pl": pl,
                    "top_frac": top_frac,
                    "k": k,
                    "n_items": len(all_ids),
                    "spearman_rho": rho,
                })

    df = pd.DataFrame(rows)

    print("=== In-sample ranking fidelity by compression ratio (median Spearman rho) ===")
    print(f"\n{'Model':<6} {'Frac':<6} {'MGSM':>8} {'MMLU':>8}")
    print("-" * 32)
    for pl in ["1PL", "2PL", "3PL"]:
        for frac in TOP_FRACS:
            sub = df[(df["pl"] == pl) & (df["top_frac"] == frac)]
            gsm = sub[sub["benchmark"] == "MGSM"]["spearman_rho"].dropna()
            mml = sub[sub["benchmark"] == "MMLU"]["spearman_rho"].dropna()
            print(f"{pl:<6} {frac:<6.0%} {gsm.median():>8.3f} {mml.median():>8.3f}")
        print()

    print("=== Mean values ===")
    print(f"\n{'Model':<6} {'Frac':<6} {'MGSM mean':>10} {'MMLU mean':>10} {'MGSM min':>10} {'MMLU min':>10}")
    print("-" * 54)
    for pl in ["1PL", "2PL", "3PL"]:
        for frac in TOP_FRACS:
            sub = df[(df["pl"] == pl) & (df["top_frac"] == frac)]
            gsm = sub[sub["benchmark"] == "MGSM"]["spearman_rho"].dropna()
            mml = sub[sub["benchmark"] == "MMLU"]["spearman_rho"].dropna()
            print(f"{pl:<6} {frac:<6.0%} {gsm.mean():>10.3f} {mml.mean():>10.3f} {gsm.min():>10.3f} {mml.min():>10.3f}")
        print()


if __name__ == "__main__":
    main()
