#!/usr/bin/env python3
"""Parameter decomposition analysis for ranking fidelity.

For 2PL and 3PL fits, computes ranking fidelity under progressively richer
IIF selection strategies to isolate each parameter's contribution:

  1PL IIF           : P(1-P)              where P = sigmoid(-b_1pl)
  2PL diff-only     : P(1-P)              where P = sigmoid(-b_2pl)   [ignores disc]
  2PL full IIF      : a²·P(1-P)          where P = sigmoid(a·-b_2pl) [adds disc]
  3PL diff-only     : P(1-P)              where P = sigmoid(-b_3pl)   [ignores disc+guess]
  3PL diff+disc     : a²·P(1-P)          where P = sigmoid(a·-b_3pl) [ignores guess]
  3PL full IIF      : 3PL formula                                      [adds guess]

Gap between consecutive rows isolates the marginal effect of each parameter.

Outputs:
  results/core_analysis/param_decomposition/summary.csv
  results/core_analysis/param_decomposition/summary_table.txt
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
OUTPUT_DIR   = RESULTS_ROOT / "core_analysis" / "param_decomposition"

PL_DIR = {"1PL": "1PL", "2PL": "2PL_map", "3PL": "3PL_map"}

MGSM_LANGS   = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
MMLU_LANGS   = ["de", "en", "es", "fr", "ja", "sw", "zh"]
MMLU_DOMAINS = ["business", "humanities", "medical", "other", "social_sciences", "stem"]

TOP_FRAC    = 0.5
RANDOM_SEED = 42
EPS         = 1e-8


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

def load_params(pl: str, stem: str, irt_root: Path | None = None) -> dict | None:
    root = irt_root if irt_root else RESULTS_ROOT / PL_DIR[pl]
    path = root / f"{stem}_{pl.lower()}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    n = len(data["diff"])
    out = {
        "item_ids": [data["item_ids"][str(i)] for i in range(n)],
        "diff": np.array(data["diff"], dtype=float),
    }
    if "disc" in data:
        out["disc"] = np.array(data["disc"], dtype=float)
    if "lambdas" in data:
        out["lambdas"] = np.array(data["lambdas"], dtype=float)
    return out


def load_responses(stem: str) -> dict[str, dict[str, int]] | None:
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


# ---------------------------------------------------------------------------
# Fidelity computation
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


def selection_strategies(params_1pl: dict, params_2pl: dict | None,
                         params_3pl: dict | None) -> dict[str, set[str]]:
    """Return {strategy_name: set_of_item_ids} for all available strategies."""
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
# Main analysis
# ---------------------------------------------------------------------------

STRATEGY_ORDER = [
    "1PL (diff)",
    "2PL diff-only",
    "2PL full (+ disc)",
    "3PL diff-only",
    "3PL diff+disc",
    "3PL full (+ guess)",
]

HELDOUT_IRT = RESULTS_ROOT / "core_analysis" / "subset_ranking_fidelity_map" / "heldout_irt"


def run_slice(benchmark: str, stem: str, split: str) -> list[dict]:
    responses = load_responses(stem)
    if responses is None:
        return []

    all_models = sorted(responses.keys())
    all_item_ids = None

    if split == "heldout":
        rng = np.random.default_rng(RANDOM_SEED)
        shuffled = rng.permutation(all_models).tolist()
        n_test = max(1, int(round(len(all_models) * 0.2)))
        test_models = set(shuffled[:n_test])
        eval_responses = {m: responses[m] for m in test_models}
        irt_root_fn = lambda pl: HELDOUT_IRT / pl
    else:
        eval_responses = responses
        irt_root_fn = lambda pl: None  # use default results root

    # Load params
    p1 = load_params("1PL", stem, irt_root_fn("1PL"))
    p2 = load_params("2PL", stem, irt_root_fn("2PL"))
    p3 = load_params("3PL", stem, irt_root_fn("3PL"))
    if p1 is None:
        return []

    all_item_ids = p1["item_ids"]
    acc_full = acc_on_items(eval_responses, set(all_item_ids))
    if len(acc_full) < 3:
        return []

    strategies = selection_strategies(p1, p2, p3)
    rows = []
    for name, item_set in strategies.items():
        rho = spearman_rho(acc_full, acc_on_items(eval_responses, item_set))
        if rho is not None:
            rows.append({
                "benchmark": benchmark,
                "stem": stem,
                "split": split,
                "strategy": name,
                "spearman_rho": rho,
            })
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    slices = (
        [("MGSM", f"mgsm_{lang}") for lang in MGSM_LANGS]
        + [("MMLU", f"mmlu_{lang}_{dom}") for lang in MMLU_LANGS for dom in MMLU_DOMAINS]
    )

    all_rows = []
    for split in ["insample", "heldout"]:
        print(f"\n=== {split} ===")
        for benchmark, stem in slices:
            rows = run_slice(benchmark, stem, split)
            all_rows.extend(rows)
            if rows:
                print(f"  {stem}: {len(rows)} strategies")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    # Summary table: median ρ per strategy × benchmark × split
    pivot = (
        df.groupby(["benchmark", "split", "strategy"])["spearman_rho"]
        .median()
        .reset_index()
        .pivot(index="strategy", columns=["benchmark", "split"], values="spearman_rho")
        .reindex(STRATEGY_ORDER)
    )
    # Reorder columns: MGSM insample, MGSM heldout, MMLU insample, MMLU heldout
    col_order = [
        ("MGSM", "insample"), ("MGSM", "heldout"),
        ("MMLU", "insample"), ("MMLU", "heldout"),
    ]
    pivot = pivot[[c for c in col_order if c in pivot.columns]]
    pivot.columns = ["MGSM in-sample", "MGSM held-out", "MMLU in-sample", "MMLU held-out"]

    print("\n\n=== Median Spearman ρ by selection strategy ===")
    print(pivot.round(3).to_string())

    with open(OUTPUT_DIR / "summary_table.txt", "w") as f:
        f.write(pivot.round(3).to_string())
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
