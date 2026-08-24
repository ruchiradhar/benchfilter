#!/usr/bin/env python3
"""Mean-accuracy baseline for benchmark filtering.

Selects top-50% items by Bernoulli variance (p*(1-p), descending), which is
maximized at mean accuracy p=0.5 — the direct frequentist analogue of IIF at
theta=0 under a 1PL model. Compares ranking fidelity against 1PL, 2PL, 3PL.

Produces:
  results/core_analysis/mean_accuracy_baseline/
    same_lang_fidelity.csv       -- in-sample and held-out same-language
    crosslingual_fidelity.csv    -- in-sample and held-out cross-lingual
    summary_table.csv            -- consolidated median ρ table
    baseline_vs_irt_fidelity.png -- figure for paper
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

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT    = PROJECT_ROOT / "data" / "irt"
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_DIR   = RESULTS_ROOT / "core_analysis" / "mean_accuracy_baseline"

MGSM_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
MMLU_LANGUAGES = ["de", "en", "es", "fr", "ja", "sw", "zh"]
MMLU_DOMAINS   = ["business", "humanities", "medical", "other", "social_sciences", "stem"]
PL_MODELS      = ["1PL", "2PL", "3PL"]
PL_DIR         = {"1PL": "1PL", "2PL": "2PL_map", "3PL": "3PL_map"}

LANG_LABELS = {
    "bn": "Bengali", "de": "German",  "en": "English",  "es": "Spanish",
    "fr": "French",  "ja": "Japanese","ru": "Russian",  "sw": "Swahili",
    "te": "Telugu",  "th": "Thai",    "zh": "Chinese",
}

TOP_FRAC      = 0.5
TEST_FRAC     = 0.2
RANDOM_SEED   = 42
N_RAND_TRIALS = 200
EPS           = 1e-8

# colours: baseline in crimson so it stands out vs IRT blues/oranges
COLORS = {
    "MeanAcc": "#C0392B",
    "1PL":     "steelblue",
    "2PL":     "darkorange",
    "3PL":     "seagreen",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_responses(stem: str) -> dict[str, dict[str, int]] | None:
    path = DATA_ROOT / f"{stem}.jsonlines"
    if not path.exists():
        return None
    resp: dict[str, dict[str, int]] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            resp[obj["subject_id"]] = obj["responses"]
    return resp


def build_matrix(responses: dict[str, dict[str, int]]) -> tuple[np.ndarray, list[str], list[str]]:
    """Return (R, item_ids, subject_ids) where R[s, i] = correctness."""
    subject_ids = sorted(responses.keys())
    # Use item ordering from first subject, sorted numerically
    sample_resp = next(iter(responses.values()))
    item_ids = sorted(sample_resp.keys(), key=lambda x: int(x.split("_")[-1]))
    item_idx = {iid: i for i, iid in enumerate(item_ids)}
    R = np.zeros((len(subject_ids), len(item_ids)), dtype=float)
    for s, sid in enumerate(subject_ids):
        for iid, r in responses[sid].items():
            if iid in item_idx:
                R[s, item_idx[iid]] = float(r)
    return R, item_ids, subject_ids


def item_variance_scores(R: np.ndarray) -> np.ndarray:
    """Bernoulli variance p*(1-p) per item — maximised at p=0.5."""
    p = R.mean(axis=0)
    return p * (1 - p)


def select_top_mean_acc(R: np.ndarray, item_ids: list[str], top_frac: float) -> set[str]:
    scores = item_variance_scores(R)
    k = max(1, int(round(len(item_ids) * top_frac)))
    top_idx = np.argsort(scores)[::-1][:k]
    return {item_ids[i] for i in top_idx}


def load_irt_params(pl: str, stem: str) -> dict | None:
    path = RESULTS_ROOT / PL_DIR[pl] / f"{stem}_{pl.lower()}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    n = len(data["diff"])
    out: dict = {
        "item_ids": [data["item_ids"][str(i)] for i in range(n)],
        "diff": np.array(data["diff"], dtype=float),
    }
    if "disc"    in data: out["disc"]    = np.array(data["disc"],    dtype=float)
    if "lambdas" in data: out["lambdas"] = np.array(data["lambdas"], dtype=float)
    return out


def load_heldout_irt_params(pl: str, stem: str) -> dict | None:
    """Load IRT params from the MAP held-out refit directory."""
    path = (RESULTS_ROOT / "core_analysis" / "subset_ranking_fidelity_map" /
            "heldout_irt" / pl / f"{stem}_{pl.lower()}.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    n = len(data["diff"])
    out: dict = {
        "item_ids": [data["item_ids"][str(i)] for i in range(n)],
        "diff": np.array(data["diff"], dtype=float),
    }
    if "disc"    in data: out["disc"]    = np.array(data["disc"],    dtype=float)
    if "lambdas" in data: out["lambdas"] = np.array(data["lambdas"], dtype=float)
    return out


# ---------------------------------------------------------------------------
# IRT IIF selection (same formulae as existing scripts)
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def compute_iif(params: dict, pl: str) -> np.ndarray:
    b = params["diff"]
    if pl == "1PL":
        p = sigmoid(-b)
        return p * (1 - p)
    a = params["disc"]
    if pl == "2PL":
        p = sigmoid(a * (-b))
        return a**2 * p * (1 - p)
    # 3PL
    c = params["lambdas"]
    p_star = sigmoid(a * (-b))
    p = c + (1 - c) * p_star
    p = np.clip(p, EPS, 1 - EPS)
    c_c = np.clip(c, 0, 1 - EPS)
    return a**2 * (p - c_c)**2 * (1 - p) / ((1 - c_c)**2 * p)


def select_irt_top(params: dict, pl: str, top_frac: float) -> set[str]:
    info = compute_iif(params, pl)
    k = max(1, int(round(len(params["item_ids"]) * top_frac)))
    return {params["item_ids"][i] for i in np.argsort(info)[::-1][:k]}


# ---------------------------------------------------------------------------
# Fidelity metrics
# ---------------------------------------------------------------------------

def model_accuracies(responses: dict[str, dict], item_ids: set) -> dict[str, float]:
    out: dict[str, float] = {}
    for sid, resp in responses.items():
        scores = [resp[iid] for iid in item_ids if iid in resp]
        if scores:
            out[sid] = float(np.mean(scores))
    return out


def spearman_rho(a: dict, b: dict) -> float | None:
    common = sorted(a.keys() & b.keys())
    if len(common) < 3:
        return None
    x = np.array([a[m] for m in common])
    y = np.array([b[m] for m in common])
    if np.std(x) < EPS or np.std(y) < EPS:
        return None
    rho, _ = spearmanr(x, y)
    return float(rho)


def random_baseline(acc_full: dict, responses: dict, all_item_ids: list[str],
                    k: int, n_trials: int = N_RAND_TRIALS, seed: int = RANDOM_SEED) -> float | None:
    rng  = np.random.default_rng(seed)
    arr  = np.array(all_item_ids)
    rhos = []
    for _ in range(n_trials):
        chosen = set(rng.choice(arr, size=k, replace=False).tolist())
        r = spearman_rho(acc_full, model_accuracies(responses, chosen))
        if r is not None:
            rhos.append(r)
    return float(np.mean(rhos)) if rhos else None


# ---------------------------------------------------------------------------
# Train/test split (identical seed to IRT held-out scripts)
# ---------------------------------------------------------------------------

def split_models(responses: dict, test_frac: float = TEST_FRAC,
                 seed: int = RANDOM_SEED) -> tuple[dict, dict]:
    all_models = sorted(responses.keys())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(all_models).tolist()
    n_test = max(1, int(round(len(all_models) * test_frac)))
    test_models  = set(shuffled[:n_test])
    train_models = set(shuffled[n_test:])
    return ({m: responses[m] for m in train_models},
            {m: responses[m] for m in test_models})


# ---------------------------------------------------------------------------
# Same-language fidelity
# ---------------------------------------------------------------------------

def run_same_language() -> pd.DataFrame:
    rows = []
    slices = (
        [("MGSM", f"mgsm_{lang}") for lang in MGSM_LANGUAGES]
        + [("MMLU", f"mmlu_{lang}_{dom}")
           for lang in MMLU_LANGUAGES for dom in MMLU_DOMAINS]
    )

    for benchmark, stem in slices:
        responses = load_responses(stem)
        if responses is None:
            continue

        R_full, item_ids, _ = build_matrix(responses)
        all_item_set = set(item_ids)
        acc_full = model_accuracies(responses, all_item_set)

        train_resp, test_resp = split_models(responses)
        R_train, _, _ = build_matrix(train_resp)
        acc_full_test = model_accuracies(test_resp, all_item_set)

        k = max(1, int(round(len(item_ids) * TOP_FRAC)))
        rand_rho = random_baseline(acc_full, responses, item_ids, k)
        rand_rho_ho = random_baseline(acc_full_test, test_resp, item_ids, k)

        # ---- Mean-accuracy baseline ----
        for split_name, R_sel, resp_sel, acc_ref, rand_r in [
            ("insample", R_full,  responses, acc_full,      rand_rho),
            ("heldout",  R_train, test_resp, acc_full_test, rand_rho_ho),
        ]:
            top_ids = select_top_mean_acc(R_sel, item_ids, TOP_FRAC)
            acc_sub = model_accuracies(resp_sel, top_ids)
            rho = spearman_rho(acc_ref, acc_sub)
            rows.append({
                "benchmark": benchmark, "slice": stem, "split": split_name,
                "strategy": "MeanAcc", "spearman_rho": rho,
                "random_baseline_rho": rand_r,
            })

        # ---- IRT baselines ----
        for pl in PL_MODELS:
            # in-sample: use full IRT params
            params_full = load_irt_params(pl, stem)
            if params_full is not None:
                top_ids = select_irt_top(params_full, pl, TOP_FRAC)
                acc_sub = model_accuracies(responses, top_ids)
                rho = spearman_rho(acc_full, acc_sub)
                rows.append({
                    "benchmark": benchmark, "slice": stem, "split": "insample",
                    "strategy": pl, "spearman_rho": rho,
                    "random_baseline_rho": rand_rho,
                })
            # held-out: use IRT params refitted on training split (pre-computed)
            params_ho = load_heldout_irt_params(pl, stem)
            if params_ho is not None:
                top_ids = select_irt_top(params_ho, pl, TOP_FRAC)
                acc_sub = model_accuracies(test_resp, top_ids)
                rho = spearman_rho(acc_full_test, acc_sub)
                rows.append({
                    "benchmark": benchmark, "slice": stem, "split": "heldout",
                    "strategy": pl, "spearman_rho": rho,
                    "random_baseline_rho": rand_rho_ho,
                })

        print(f"  {stem} done")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-lingual fidelity
# ---------------------------------------------------------------------------

def run_crosslingual() -> pd.DataFrame:
    rows = []

    def _cross_for_benchmark(
        languages: list[str],
        get_stem,
        get_all_langs,
        is_heldout: bool,
    ):
        # Load responses and item sets
        resp_map: dict[str, dict] = {}
        for lang in languages:
            stem = get_stem(lang)
            r = load_responses(stem)
            if r is not None:
                resp_map[lang] = r

        for lang in languages:
            if lang not in resp_map:
                continue
            if is_heldout:
                _, resp_map[lang] = split_models(resp_map[lang])

        for src_lang in languages:
            if src_lang not in resp_map:
                continue
            stem_src = get_stem(src_lang)

            # Mean-accuracy top items from source (train split for heldout)
            if is_heldout:
                train_src, _ = split_models(load_responses(stem_src))
                R_src, item_ids_src, _ = build_matrix(train_src)
            else:
                R_src, item_ids_src, _ = build_matrix(resp_map[src_lang])
            top_ids_ma = select_top_mean_acc(R_src, item_ids_src, TOP_FRAC)

            # IRT top items
            top_ids_irt: dict[str, set | None] = {}
            for pl in PL_MODELS:
                if is_heldout:
                    p = load_heldout_irt_params(pl, stem_src)
                else:
                    p = load_irt_params(pl, stem_src)
                top_ids_irt[pl] = select_irt_top(p, pl, TOP_FRAC) if p is not None else None

            for tgt_lang in languages:
                if tgt_lang not in resp_map:
                    continue
                tgt_resp = resp_map[tgt_lang]
                stem_tgt = get_stem(tgt_lang)
                ref_params = load_irt_params("1PL", stem_tgt)
                if ref_params is None:
                    continue
                all_item_ids = set(ref_params["item_ids"])
                acc_full = model_accuracies(tgt_resp, all_item_ids)

                # Mean-accuracy
                valid_ma = top_ids_ma & all_item_ids
                if len(valid_ma) >= 3:
                    acc_sub = model_accuracies(tgt_resp, valid_ma)
                    rho = spearman_rho(acc_full, acc_sub)
                    rows.append({
                        "benchmark": benchmark, "domain": domain if benchmark == "MMLU" else None,
                        "split": "heldout" if is_heldout else "insample",
                        "strategy": "MeanAcc",
                        "source_lang": src_lang, "target_lang": tgt_lang,
                        "spearman_rho": rho,
                    })

                # IRT
                for pl in PL_MODELS:
                    if top_ids_irt[pl] is None:
                        continue
                    valid = top_ids_irt[pl] & all_item_ids
                    if len(valid) < 3:
                        continue
                    acc_sub = model_accuracies(tgt_resp, valid)
                    rho = spearman_rho(acc_full, acc_sub)
                    rows.append({
                        "benchmark": benchmark, "domain": domain if benchmark == "MMLU" else None,
                        "split": "heldout" if is_heldout else "insample",
                        "strategy": pl,
                        "source_lang": src_lang, "target_lang": tgt_lang,
                        "spearman_rho": rho,
                    })

    # MGSM
    benchmark, domain = "MGSM", None
    for is_heldout in [False, True]:
        _cross_for_benchmark(
            MGSM_LANGUAGES,
            lambda lang: f"mgsm_{lang}",
            None, is_heldout,
        )

    # MMLU (within-domain cross-language)
    benchmark = "MMLU"
    for domain in MMLU_DOMAINS:
        for is_heldout in [False, True]:
            _cross_for_benchmark(
                MMLU_LANGUAGES,
                lambda lang, d=domain: f"mmlu_{lang}_{d}",
                None, is_heldout,
            )
        print(f"  MMLU {domain} done")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

STRATEGY_ORDER = ["MeanAcc", "1PL", "2PL", "3PL"]

def make_summary(df_same: pd.DataFrame, df_cross: pd.DataFrame) -> pd.DataFrame:
    records = []
    for strategy in STRATEGY_ORDER:
        row = {"Strategy": strategy}
        for bm in ["MGSM", "MMLU"]:
            for split in ["insample", "heldout"]:
                sub = df_same[(df_same["strategy"] == strategy) &
                              (df_same["benchmark"] == bm) &
                              (df_same["split"] == split)]
                row[f"{bm}_same_{split}"] = round(sub["spearman_rho"].median(), 3)
        records.append(row)

    # Cross-lingual off-diagonal
    for strategy in STRATEGY_ORDER:
        rec = next(r for r in records if r["Strategy"] == strategy)
        for bm in ["MGSM", "MMLU"]:
            for split in ["insample", "heldout"]:
                off = df_cross[(df_cross["strategy"] == strategy) &
                               (df_cross["benchmark"] == bm) &
                               (df_cross["split"] == split) &
                               (df_cross["source_lang"] != df_cross["target_lang"])]
                rec[f"{bm}_cross_{split}"] = round(off["spearman_rho"].median(), 3)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_fidelity_comparison(df_same: pd.DataFrame, output_path: Path) -> None:
    strategies = STRATEGY_ORDER
    splits     = ["insample", "heldout"]
    split_label = {"insample": "In-sample", "heldout": "Held-out"}
    benchmarks  = ["MGSM", "MMLU"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for row_idx, bm in enumerate(benchmarks):
        for col_idx, split in enumerate(splits):
            ax = axes[row_idx][col_idx]
            sub = df_same[(df_same["benchmark"] == bm) & (df_same["split"] == split)]
            x_positions = np.arange(len(strategies))
            rand_vals = []
            for pos, strategy in enumerate(strategies):
                vals = sub[sub["strategy"] == strategy]["spearman_rho"].dropna().values
                rand_v = sub[sub["strategy"] == strategy]["random_baseline_rho"].dropna().values
                color  = COLORS[strategy]
                if len(vals):
                    ax.boxplot(
                        [vals], positions=[pos], widths=0.4, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2),
                        boxprops=dict(facecolor=color, alpha=0.4),
                        whiskerprops=dict(color=color), capprops=dict(color=color),
                        flierprops=dict(marker="o", color=color, markersize=3, alpha=0.5),
                    )
                    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
                    ax.scatter(pos + jitter, vals, color=color, alpha=0.5, s=10, zorder=3)
                    med = float(np.median(vals))
                    ax.text(pos, med + 0.008, f"{med:.3f}",
                            ha="center", va="bottom", fontsize=7.5, fontweight="bold")
                if len(rand_v):
                    rand_vals.append(float(np.nanmean(rand_v)))
                else:
                    rand_vals.append(None)

            for pos, rv in enumerate(rand_vals):
                if rv is not None:
                    ax.hlines(rv, pos - 0.22, pos + 0.22,
                              colors=COLORS[strategies[pos]], linestyles="dashed", linewidth=1.3)

            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                ["Mean Acc\n(baseline)", "1PL", "2PL", "3PL"], fontsize=9
            )
            ax.set_ylim(-0.2, 1.08)
            ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
            ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8)
            if col_idx == 0:
                ax.set_ylabel("Spearman ρ (subset vs. full)", fontsize=10)
            title = f"{bm} — {split_label[split]}"
            ax.set_title(title, fontsize=11, fontweight="bold")

    fig.suptitle(
        "Same-language ranking fidelity: mean-accuracy baseline vs IRT variants",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Same-language fidelity ===")
    df_same = run_same_language()
    df_same.to_csv(OUTPUT_DIR / "same_lang_fidelity.csv", index=False)
    print(f"Saved same_lang_fidelity.csv ({len(df_same)} rows)")

    print("\n=== Cross-lingual fidelity ===")
    df_cross = run_crosslingual()
    df_cross.to_csv(OUTPUT_DIR / "crosslingual_fidelity.csv", index=False)
    print(f"Saved crosslingual_fidelity.csv ({len(df_cross)} rows)")

    print("\n=== Summary table ===")
    df_summary = make_summary(df_same, df_cross)
    df_summary.to_csv(OUTPUT_DIR / "summary_table.csv", index=False)

    # Pretty-print the summary
    cols_same  = [c for c in df_summary.columns if "same" in c]
    cols_cross = [c for c in df_summary.columns if "cross" in c]
    print("\n--- Same-language median Spearman ρ ---")
    print(df_summary[["Strategy"] + cols_same].to_string(index=False))
    print("\n--- Cross-lingual median Spearman ρ (off-diagonal) ---")
    print(df_summary[["Strategy"] + cols_cross].to_string(index=False))

    print("\n=== Figure ===")
    plot_fidelity_comparison(df_same, OUTPUT_DIR / "baseline_vs_irt_fidelity.png")

    print("\nDone. Results in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
