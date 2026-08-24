#!/usr/bin/env python3
"""Cross-lingual parameter decomposition analysis.

For each (src_lang, tgt_lang) pair, selects items using progressively richer
IIF formulas derived from the *source* language IRT parameters, then evaluates
ranking fidelity on *target* language model responses.

Strategies (same as same-language param_decomposition.py):
  1PL (diff)         : P(1-P)     where P = sigmoid(-b_1pl)
  2PL diff-only      : P(1-P)     where P = sigmoid(-b_2pl)   [ignores disc]
  2PL full (+ disc)  : a²·P(1-P) where P = sigmoid(a·-b_2pl) [adds disc]
  3PL diff-only      : P(1-P)     where P = sigmoid(-b_3pl)   [ignores disc+guess]
  3PL diff+disc      : a²·P(1-P) where P = sigmoid(a·-b_3pl) [ignores guess]
  3PL full (+ guess) : 3PL formula                            [adds guess]

Gap between consecutive rows isolates each parameter's cross-lingual transferability.
If adding discrimination *hurts* cross-lingual ρ, the disc parameter is language-specific.

Both in-sample and held-out (MAP refit) splits are analysed.

Outputs:
  results/core_analysis/crosslingual_param_decomposition/summary.csv
  results/core_analysis/crosslingual_param_decomposition/summary_table.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
DATA_ROOT    = PROJECT_ROOT / "data" / "irt"
OUTPUT_DIR   = RESULTS_ROOT / "core_analysis" / "crosslingual_param_decomposition"

PL_DIR       = {"1PL": "1PL", "2PL": "2PL_map", "3PL": "3PL_map"}
HELDOUT_IRT  = RESULTS_ROOT / "core_analysis" / "subset_ranking_fidelity_map" / "heldout_irt"

MGSM_LANGS   = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
MMLU_LANGS   = ["de", "en", "es", "fr", "ja", "sw", "zh"]
MMLU_DOMAINS = ["business", "humanities", "medical", "other", "social_sciences", "stem"]

TOP_FRAC     = 0.5
RANDOM_SEED  = 42
EPS          = 1e-8

STRATEGY_ORDER = [
    "1PL (diff)",
    "2PL diff-only",
    "2PL full (+ disc)",
    "3PL diff-only",
    "3PL diff+disc",
    "3PL full (+ guess)",
]


# ---------------------------------------------------------------------------
# IIF helpers
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def iif_diff_only(diff: np.ndarray) -> np.ndarray:
    p = sigmoid(-diff)
    return p * (1.0 - p)


def iif_diff_disc(diff: np.ndarray, disc: np.ndarray) -> np.ndarray:
    p = sigmoid(disc * (-diff))
    return disc ** 2 * p * (1.0 - p)


def iif_3pl_full(diff: np.ndarray, disc: np.ndarray, lam: np.ndarray) -> np.ndarray:
    p_star = sigmoid(disc * (-diff))
    p = lam + (1.0 - lam) * p_star
    p = np.clip(p, EPS, 1.0 - EPS)
    lam_c = np.clip(lam, 0.0, 1.0 - EPS)
    return disc ** 2 * (p - lam_c) ** 2 * (1.0 - p) / ((1.0 - lam_c) ** 2 * p)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_params_insample(pl: str, stem: str) -> dict | None:
    path = RESULTS_ROOT / PL_DIR[pl] / f"{stem}_{pl.lower()}.json"
    return _load_raw(path)


def load_params_heldout(pl: str, stem: str) -> dict | None:
    path = HELDOUT_IRT / pl / f"{stem}_{pl.lower()}.json"
    return _load_raw(path)


def _load_raw(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    n = len(data["diff"])
    out = {
        "item_ids": [data["item_ids"][str(i)] for i in range(n)],
        "diff": np.array(data["diff"], dtype=float),
    }
    if "disc"    in data: out["disc"]    = np.array(data["disc"],    dtype=float)
    if "lambdas" in data: out["lambdas"] = np.array(data["lambdas"], dtype=float)
    return out


def load_responses(stem: str) -> dict | None:
    path = DATA_ROOT / f"{stem}.jsonlines"
    if not path.exists():
        return None
    responses: dict = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            responses[obj["subject_id"]] = obj["responses"]
    return responses


def filter_test(responses: dict) -> dict:
    all_models = sorted(responses.keys())
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(all_models).tolist()
    n_test = max(1, int(round(len(all_models) * 0.2)))
    test_models = set(shuffled[:n_test])
    return {m: r for m, r in responses.items() if m in test_models}


# ---------------------------------------------------------------------------
# Fidelity helpers
# ---------------------------------------------------------------------------

def top_items(item_ids: list[str], scores: np.ndarray, top_frac: float) -> set[str]:
    k = max(1, int(round(len(item_ids) * top_frac)))
    idx = np.argsort(scores)[::-1][:k]
    return {item_ids[i] for i in idx}


def acc_on_items(responses: dict, item_set: set[str]) -> dict[str, float]:
    out = {}
    for model, resp in responses.items():
        scores = [resp[iid] for iid in item_set if iid in resp]
        if scores:
            out[model] = float(np.mean(scores))
    return out


def spearman_rho(full: dict, sub: dict) -> float | None:
    common = sorted(full.keys() & sub.keys())
    if len(common) < 3:
        return None
    a = np.array([full[m] for m in common])
    b = np.array([sub[m] for m in common])
    if np.std(a) < EPS or np.std(b) < EPS:
        return None
    rho, _ = spearmanr(a, b)
    return float(rho)


def selection_strategies(
    params_1pl: dict,
    params_2pl: dict | None,
    params_3pl: dict | None,
) -> dict[str, set[str]]:
    strategies = {}
    strategies["1PL (diff)"] = top_items(
        params_1pl["item_ids"], iif_diff_only(params_1pl["diff"]), TOP_FRAC)
    if params_2pl:
        strategies["2PL diff-only"] = top_items(
            params_2pl["item_ids"], iif_diff_only(params_2pl["diff"]), TOP_FRAC)
        strategies["2PL full (+ disc)"] = top_items(
            params_2pl["item_ids"],
            iif_diff_disc(params_2pl["diff"], params_2pl["disc"]), TOP_FRAC)
    if params_3pl:
        strategies["3PL diff-only"] = top_items(
            params_3pl["item_ids"], iif_diff_only(params_3pl["diff"]), TOP_FRAC)
        strategies["3PL diff+disc"] = top_items(
            params_3pl["item_ids"],
            iif_diff_disc(params_3pl["diff"], params_3pl["disc"]), TOP_FRAC)
        strategies["3PL full (+ guess)"] = top_items(
            params_3pl["item_ids"],
            iif_3pl_full(params_3pl["diff"], params_3pl["disc"],
                         params_3pl["lambdas"]), TOP_FRAC)
    return strategies


# ---------------------------------------------------------------------------
# Cross-lingual analysis
# ---------------------------------------------------------------------------

def run_crosslingual_pair(
    benchmark: str,
    src_stem: str,
    tgt_stem: str,
    src_lang: str,
    tgt_lang: str,
    split: str,
    domain: str | None = None,
) -> list[dict]:
    load_fn = load_params_heldout if split == "heldout" else load_params_insample

    p1_src = load_fn("1PL", src_stem)
    p2_src = load_fn("2PL", src_stem)
    p3_src = load_fn("3PL", src_stem)
    if p1_src is None:
        return []

    tgt_responses = load_responses(tgt_stem)
    if tgt_responses is None:
        return []
    if split == "heldout":
        tgt_responses = filter_test(tgt_responses)

    # Full tgt accuracy (all items from tgt 1PL param set)
    p1_tgt = load_fn("1PL", tgt_stem)
    if p1_tgt is None:
        return []
    all_tgt_items = set(p1_tgt["item_ids"])
    acc_full = acc_on_items(tgt_responses, all_tgt_items)
    if len(acc_full) < 3:
        return []

    strategies = selection_strategies(p1_src, p2_src, p3_src)
    rows = []
    for name, src_item_set in strategies.items():
        # Intersect with tgt items (items are parallel / same IDs)
        valid_ids = src_item_set & all_tgt_items
        if len(valid_ids) < 3:
            continue
        acc_sub = acc_on_items(tgt_responses, valid_ids)
        rho = spearman_rho(acc_full, acc_sub)
        if rho is not None:
            rows.append({
                "benchmark": benchmark,
                "domain": domain,
                "source_lang": src_lang,
                "target_lang": tgt_lang,
                "split": split,
                "strategy": name,
                "spearman_rho": rho,
            })
    return rows


def run_all(split: str) -> list[dict]:
    all_rows = []

    # MGSM: all (src, tgt) pairs
    for src_lang in MGSM_LANGS:
        for tgt_lang in MGSM_LANGS:
            rows = run_crosslingual_pair(
                "MGSM",
                f"mgsm_{src_lang}", f"mgsm_{tgt_lang}",
                src_lang, tgt_lang, split,
            )
            all_rows.extend(rows)

    # MMLU: all (src, tgt) pairs within each domain
    for domain in MMLU_DOMAINS:
        for src_lang in MMLU_LANGS:
            for tgt_lang in MMLU_LANGS:
                rows = run_crosslingual_pair(
                    "MMLU",
                    f"mmlu_{src_lang}_{domain}", f"mmlu_{tgt_lang}_{domain}",
                    src_lang, tgt_lang, split, domain=domain,
                )
                all_rows.extend(rows)

    return all_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for split in ["insample", "heldout"]:
        print(f"\n=== {split} ===")
        rows = run_all(split)
        all_rows.extend(rows)
        n_pairs = len({(r["benchmark"], r.get("domain"), r["source_lang"], r["target_lang"])
                       for r in rows})
        print(f"  {len(rows)} rows across {n_pairs} (src, tgt) pairs")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    # Off-diagonal only (cross-lingual transfer; exclude src == tgt)
    df_off = df[df["source_lang"] != df["target_lang"]]

    # Median ρ per strategy × benchmark × split
    pivot = (
        df_off.groupby(["benchmark", "split", "strategy"])["spearman_rho"]
        .median()
        .reset_index()
        .pivot(index="strategy", columns=["benchmark", "split"], values="spearman_rho")
        .reindex(STRATEGY_ORDER)
    )
    col_order = [
        ("MGSM", "insample"), ("MGSM", "heldout"),
        ("MMLU", "insample"), ("MMLU", "heldout"),
    ]
    pivot = pivot[[c for c in col_order if c in pivot.columns]]
    pivot.columns = ["MGSM in-sample", "MGSM held-out", "MMLU in-sample", "MMLU held-out"]

    print("\n\n=== Median cross-lingual Spearman ρ by selection strategy (off-diagonal) ===")
    print(pivot.round(3).to_string())

    with open(OUTPUT_DIR / "summary_table.txt", "w") as f:
        f.write(pivot.round(3).to_string())
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
