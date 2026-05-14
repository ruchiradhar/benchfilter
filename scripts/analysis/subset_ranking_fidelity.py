#!/usr/bin/env python3
"""Subset ranking fidelity via IRT-filtered items.

Asks: when you select the top-k items by Item Information Function (IIF) from
an x-PL IRT model, how well does model accuracy on that subset reproduce the
model ranking on the full benchmark?

The correct IRT-based filtering pipeline is:
  1. Calibration: evaluate existing ("calibration") models on the full
     benchmark, fit IRT, select the most informative items.
  2. Deployment: a *new* model runs only the subset; its score should
     predict where it would rank on the full benchmark.

Four analyses are performed:

1. **In-sample same-language fidelity** (MGSM + MMLU) — reference only.
   Item selection and ranking evaluation use the *same* set of models.
   This is circular and serves only as an upper bound.

2. **Held-out same-language fidelity** (MGSM + MMLU) — the correct evaluation.
   A single train/test model split (default 80/20):
     - IRT is re-fitted on the training models only.
     - Item selection and ranking fidelity are measured on the held-out test
       models, simulating deployment to unseen models.

3. **In-sample cross-lingual transfer** (MGSM + MMLU) — reference only.
   For each (source language, target language, IRT model) triple:
     - Use the *source* language's IIF (fitted on all models) to select items.
     - Compute per-model accuracy on those items in the *target* language.
     - Report Spearman ρ against the full-benchmark target-language ranking.
   This is circular on models and serves only as an upper bound.

4. **Held-out cross-lingual transfer** (MGSM + MMLU) — the correct evaluation.
   Same as Analysis 3, but uses the held-out model split and IRT refitted on
   train models (reuses the saved IRT from Analysis 2). Ranking fidelity is
   measured only on the test models.

IIF formulas at θ=0:
  1PL: I = P(1-P)                          P = sigmoid(-b)
  2PL: I = a² · P(1-P)                     P = sigmoid(a·(-b))
  3PL: I = a²·(P-c)²·(1-P) / P            P = c + (1-c)·sigmoid(a·(-b))

Usage:
    python scripts/analysis/subset_ranking_fidelity.py
    python scripts/analysis/subset_ranking_fidelity.py \\
        --results_root results --data_root data/irt \\
        --output_dir results/analysis/subset_ranking_fidelity \\
        --top_frac 0.5 --n_folds 5
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import pyro
import torch
from py_irt.config import IrtConfig
from py_irt.dataset import Dataset
from py_irt.training import IrtModelTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MGSM_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
MMLU_LANGUAGES = ["de", "en", "es", "fr", "ja", "sw", "zh"]
MMLU_DOMAINS = ["business", "humanities", "medical", "other", "social_sciences", "stem"]

PL_MODELS = ["1PL", "2PL", "3PL"]
PL_COLORS = {"1PL": "steelblue", "2PL": "darkorange", "3PL": "seagreen"}

LANG_LABELS = {
    "bn": "Bengali", "de": "German", "en": "English", "es": "Spanish",
    "fr": "French", "ja": "Japanese", "ru": "Russian", "sw": "Swahili",
    "te": "Telugu", "th": "Thai", "zh": "Chinese",
}

CMAP_RYG = LinearSegmentedColormap.from_list(
    "ryg", [(0.0, "#CC0000"), (0.5, "#FFFF00"), (1.0, "#007700")]
)

EPS = 1e-8
RANDOM_SEED = 42
N_RANDOM_TRIALS = 200  # number of random draws per slice for baseline
DEFAULT_TEST_FRAC = 0.2


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------------------------------------------------------------------------
# IIF computation
# ---------------------------------------------------------------------------

def iif_1pl(diff: np.ndarray) -> np.ndarray:
    p = sigmoid(-diff)
    return p * (1.0 - p)


def iif_2pl(diff: np.ndarray, disc: np.ndarray) -> np.ndarray:
    p = sigmoid(disc * (-diff))
    return disc ** 2 * p * (1.0 - p)


def iif_3pl(diff: np.ndarray, disc: np.ndarray, lam: np.ndarray) -> np.ndarray:
    p_star = sigmoid(disc * (-diff))
    p = lam + (1.0 - lam) * p_star
    p = np.clip(p, EPS, 1.0 - EPS)
    return disc ** 2 * (p - lam) ** 2 * (1.0 - p) / p


# ---------------------------------------------------------------------------
# IRT parameter loading
# ---------------------------------------------------------------------------

def load_irt_params(results_root: Path, pl: str, stem: str) -> dict | None:
    """Load IRT parameters for a slice; return None if file missing."""
    path = results_root / pl / f"{stem}_{pl.lower()}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    n = len(data["diff"])
    item_ids = [data["item_ids"][str(i)] for i in range(n)]
    out: dict = {
        "item_ids": item_ids,
        "diff": np.array(data["diff"], dtype=float),
    }
    if "disc" in data:
        out["disc"] = np.array(data["disc"], dtype=float)
    if "lambdas" in data:
        out["lambdas"] = np.array(data["lambdas"], dtype=float)
    return out


def compute_iif(params: dict, pl: str) -> np.ndarray:
    if pl == "1PL":
        return iif_1pl(params["diff"])
    elif pl == "2PL":
        return iif_2pl(params["diff"], params["disc"])
    else:  # 3PL
        return iif_3pl(params["diff"], params["disc"], params["lambdas"])


def select_top_items(params: dict, pl: str, top_frac: float) -> set[str]:
    """Return the set of item IDs in the top-frac by IIF."""
    info = compute_iif(params, pl)
    k = max(1, int(round(len(params["item_ids"]) * top_frac)))
    top_idx = np.argsort(info)[::-1][:k]
    return {params["item_ids"][i] for i in top_idx}


# ---------------------------------------------------------------------------
# Response data loading
# ---------------------------------------------------------------------------

def load_responses(data_root: Path, stem: str) -> dict[str, dict[str, int]] | None:
    """Load the IRT jsonlines file and return {subject_id: {item_id: 0/1}}."""
    path = data_root / f"{stem}.jsonlines"
    if not path.exists():
        return None
    responses: dict[str, dict[str, int]] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            responses[obj["subject_id"]] = obj["responses"]
    return responses


def model_accuracies(
    responses: dict[str, dict[str, int]],
    item_ids: list[str] | set[str],
) -> dict[str, float]:
    """Compute accuracy for each model on the given item_ids."""
    item_set = set(item_ids)
    accs: dict[str, float] = {}
    for subject, resp in responses.items():
        scores = [resp[iid] for iid in item_set if iid in resp]
        if scores:
            accs[subject] = float(np.mean(scores))
    return accs


def spearman_between(
    acc_full: dict[str, float],
    acc_sub: dict[str, float],
) -> float | None:
    """Spearman ρ between two accuracy dicts on their common models."""
    common = sorted(acc_full.keys() & acc_sub.keys())
    if len(common) < 3:
        return None
    a = np.array([acc_full[m] for m in common])
    b = np.array([acc_sub[m] for m in common])
    # If either vector is constant, spearman is undefined — return None
    if np.std(a) < EPS or np.std(b) < EPS:
        return None
    rho, _ = spearmanr(a, b)
    return float(rho)


def accuracy_prediction_stats(
    acc_full: dict[str, float],
    acc_sub: dict[str, float],
) -> dict[str, float] | None:
    """MAE, RMSE, R², and bias (subset minus full) on common models."""
    common = sorted(acc_full.keys() & acc_sub.keys())
    if len(common) < 2:
        return None
    y_true = np.array([acc_full[m] for m in common])
    y_pred = np.array([acc_sub[m] for m in common])
    residuals = y_pred - y_true
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    bias = float(np.mean(residuals))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > EPS else float("nan")
    return {"mae": mae, "rmse": rmse, "bias": bias, "r2": r2}


def random_baseline_rho(
    acc_full: dict[str, float],
    responses: dict[str, dict[str, int]],
    all_item_ids: list[str],
    k: int,
    n_trials: int = N_RANDOM_TRIALS,
    seed: int = RANDOM_SEED,
) -> float | None:
    """
    Average Spearman ρ over `n_trials` random draws of `k` items from
    `all_item_ids`, comparing model accuracy on the random subset vs. the
    full benchmark.  Provides a chance-level baseline.
    """
    rng = np.random.default_rng(seed)
    rhos: list[float] = []
    item_arr = np.array(all_item_ids)
    for _ in range(n_trials):
        chosen = set(rng.choice(item_arr, size=k, replace=False).tolist())
        acc_rand = model_accuracies(responses, chosen)
        rho = spearman_between(acc_full, acc_rand)
        if rho is not None:
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else None


# ---------------------------------------------------------------------------
# Analysis 1: Same-language fidelity
# ---------------------------------------------------------------------------

def run_same_language(
    results_root: Path,
    data_root: Path,
    top_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns a tuple of DataFrames:
      - summary: benchmark, slice, pl, n_items, k, spearman_rho, mae, rmse, bias, r2
      - pairs:   benchmark, slice, pl, model, acc_full, acc_sub
    """
    rows: list[dict] = []
    pair_rows: list[dict] = []

    slices = (
        [("MGSM", f"mgsm_{lang}") for lang in MGSM_LANGUAGES]
        + [
            ("MMLU", f"mmlu_{lang}_{domain}")
            for lang in MMLU_LANGUAGES
            for domain in MMLU_DOMAINS
        ]
    )

    for benchmark, stem in slices:
        responses = load_responses(data_root, stem)
        if responses is None:
            print(f"  [skip] no response data: {stem}")
            continue

        # full item set (from any PL file; items are the same)
        all_item_ids: set[str] | None = None
        all_item_list: list[str] = []

        for pl in PL_MODELS:
            params = load_irt_params(results_root, pl, stem)
            if params is None:
                continue

            if all_item_ids is None:
                all_item_ids = set(params["item_ids"])
                all_item_list = params["item_ids"]
                acc_full = model_accuracies(responses, all_item_ids)

            top_ids = select_top_items(params, pl, top_frac)
            acc_sub = model_accuracies(responses, top_ids)
            rho = spearman_between(acc_full, acc_sub)
            acc_stats = accuracy_prediction_stats(acc_full, acc_sub)

            # Random baseline: same k, random items
            rho_rand = random_baseline_rho(
                acc_full, responses, all_item_list, len(top_ids)
            )

            rows.append({
                "benchmark": benchmark,
                "slice": stem,
                "pl": pl,
                "n_items": len(params["item_ids"]),
                "k": len(top_ids),
                "spearman_rho": rho,
                "random_baseline_rho": rho_rand,
                **(acc_stats or {}),
            })

            for m in sorted(acc_full.keys() & acc_sub.keys()):
                pair_rows.append({
                    "benchmark": benchmark,
                    "slice": stem,
                    "pl": pl,
                    "model": m,
                    "acc_full": acc_full[m],
                    "acc_sub": acc_sub[m],
                })

    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


# ---------------------------------------------------------------------------
# Analysis 2: Held-out model fidelity (train/test model split + IRT refit)
# ---------------------------------------------------------------------------

PL_TO_MODEL_TYPE = {"1PL": "1pl", "2PL": "2pl", "3PL": "3pl"}
PL_EPOCHS       = {"1PL": 2000,  "2PL": 3000,  "3PL": 3000}

def refit_irt(
    train_responses: dict[str, dict[str, int]],
    pl: str,
    seed: int = RANDOM_SEED,
    save_path: Path | None = None,
) -> dict | None:
    """
    Write a temporary jsonlines file from train_responses, fit an IRT model,
    and return the parameter dict in the same format as load_irt_params.
    If save_path is provided, also writes the parameters as JSON to that path
    (same format as the pre-computed results in results/1PL/, 2PL/, 3PL/).
    Returns None if fitting fails.
    """
    # Skip re-training if results already saved to disk
    if save_path is not None and save_path.exists():
        data = json.loads(save_path.read_text())
        iids = data["item_ids"]
        n = len(data["diff"])
        item_ids = [iids[str(i)] for i in range(n)]
        out: dict = {
            "item_ids": item_ids,
            "diff": np.array(data["diff"], dtype=float),
        }
        if "disc" in data:
            out["disc"] = np.array(data["disc"], dtype=float)
        if "lambdas" in data:
            out["lambdas"] = np.array(data["lambdas"], dtype=float)
        return out

    model_type = PL_TO_MODEL_TYPE[pl]
    epochs = PL_EPOCHS[pl]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonlines", delete=False
    ) as fh:
        tmp_path = Path(fh.name)
        for subject_id, responses in train_responses.items():
            fh.write(json.dumps({"subject_id": subject_id, "responses": responses}) + "\n")

    try:
        pyro.clear_param_store()
        if seed is not None:
            pyro.set_rng_seed(seed)
            torch.manual_seed(seed)

        dataset = Dataset.from_jsonlines(tmp_path)
        config = IrtConfig(
            model_type=model_type,
            epochs=epochs,
            dropout=0.5,
            lr=0.1,
            lr_decay=0.9999,
        )
        trainer = IrtModelTrainer(
            config=config,
            data_path=tmp_path,
            dataset=dataset,
            verbose=False,
        )
        trainer.train(epochs=epochs, device="cpu")
        raw = trainer.best_params or trainer.last_params

        n = len(raw["diff"])
        iids = raw["item_ids"]
        if isinstance(iids, list):
            item_ids = list(iids)
        else:
            # in-memory dict uses int keys; on-disk JSON uses str keys
            item_ids = [iids[i] if i in iids else iids[str(i)] for i in range(n)]
        out: dict = {
            "item_ids": item_ids,
            "diff": np.array(raw["diff"], dtype=float),
        }
        if "disc" in raw:
            out["disc"] = np.array(raw["disc"], dtype=float)
        if "lambdas" in raw:
            out["lambdas"] = np.array(raw["lambdas"], dtype=float)

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            payload: dict = {
                "irt_model": PL_TO_MODEL_TYPE[pl],
                "item_ids": {str(i): iid for i, iid in enumerate(item_ids)},
                "diff": out["diff"].tolist(),
            }
            if "disc" in out:
                payload["disc"] = out["disc"].tolist()
            if "lambdas" in out:
                payload["lambdas"] = out["lambdas"].tolist()
            save_path.write_text(json.dumps(payload, indent=2))

        return out
    except Exception as e:
        print(f"    [refit warning] {pl} fit failed: {e}")
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


def run_held_out_fidelity(
    results_root: Path,
    data_root: Path,
    top_frac: float,
    test_frac: float = 0.2,
    irt_save_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Proper evaluation of the IRT filtering pipeline via a single train/test
    model split.

    For each benchmark slice and IRT model:
      1. Randomly split models into train (80%) and test (20%).
      2. Refit IRT on train models only → select top-k items by IIF.
      3. Evaluate test models on the FULL benchmark (ground truth) and on
         the IRT-selected SUBSET (cheap proxy).
      4. Measure Spearman ρ between the two test-model rankings.
      5. Repeat for a random-item baseline of the same k.

    Returns a tuple of DataFrames:
      - summary: benchmark, slice, pl, n_train, n_test, n_items, k,
                 spearman_rho, random_baseline_rho, mae, rmse, bias, r2
      - pairs:   benchmark, slice, pl, model, acc_full, acc_sub
    """
    rows: list[dict] = []
    pair_rows: list[dict] = []

    slices = (
        [("MGSM", f"mgsm_{lang}") for lang in MGSM_LANGUAGES]
        + [
            ("MMLU", f"mmlu_{lang}_{domain}")
            for lang in MMLU_LANGUAGES
            for domain in MMLU_DOMAINS
        ]
    )

    for benchmark, stem in slices:
        responses = load_responses(data_root, stem)
        if responses is None:
            continue

        all_models = sorted(responses.keys())
        rng = np.random.default_rng(RANDOM_SEED)
        shuffled = rng.permutation(all_models).tolist()
        n_test = max(1, int(round(len(all_models) * test_frac)))
        test_models  = shuffled[:n_test]
        train_models = shuffled[n_test:]

        train_responses = {m: responses[m] for m in train_models}
        test_responses  = {m: responses[m] for m in test_models}

        # Full-benchmark accuracy for test models (ground truth)
        # Use item IDs from any existing PL file (items are benchmark-invariant)
        ref_params = load_irt_params(results_root, "1PL", stem)
        if ref_params is None:
            continue
        all_item_ids: list[str] = ref_params["item_ids"]
        acc_full_test = model_accuracies(test_responses, set(all_item_ids))

        print(f"  {stem}: train={len(train_models)}, test={len(test_models)}", end="")

        for pl in PL_MODELS:
            print(f"  [{pl}]", end="", flush=True)
            # Refit IRT on training models
            save_path = (
                irt_save_dir / pl / f"{stem}_{pl.lower()}.json"
                if irt_save_dir is not None else None
            )
            train_params = refit_irt(train_responses, pl, seed=RANDOM_SEED, save_path=save_path)
            if train_params is None:
                continue

            top_ids = select_top_items(train_params, pl, top_frac)
            k = len(top_ids)

            acc_sub_test = model_accuracies(test_responses, top_ids)
            rho = spearman_between(acc_full_test, acc_sub_test)
            acc_stats = accuracy_prediction_stats(acc_full_test, acc_sub_test)

            rho_rand = random_baseline_rho(
                acc_full_test, test_responses, all_item_ids, k,
                n_trials=N_RANDOM_TRIALS, seed=RANDOM_SEED,
            )

            rows.append({
                "benchmark": benchmark,
                "slice": stem,
                "pl": pl,
                "n_train": len(train_models),
                "n_test": len(test_models),
                "n_items": len(all_item_ids),
                "k": k,
                "spearman_rho": rho,
                "random_baseline_rho": rho_rand,
                **(acc_stats or {}),
            })

            for m in sorted(acc_full_test.keys() & acc_sub_test.keys()):
                pair_rows.append({
                    "benchmark": benchmark,
                    "slice": stem,
                    "pl": pl,
                    "model": m,
                    "acc_full": acc_full_test[m],
                    "acc_sub": acc_sub_test[m],
                })
        print()  # newline after per-slice progress

    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def plot_held_out_fidelity(
    df_held: pd.DataFrame,
    df_insample: pd.DataFrame,
    top_frac: float,
    test_frac: float,
    output_path: Path,
) -> None:
    """
    Side-by-side box plots comparing in-sample (upper bound) vs.
    held-out (correct) ranking fidelity, for each IRT model.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    fig.suptitle(
        f"Ranking fidelity: in-sample (top row) vs. held-out model split (bottom row)\n"
        f"Spearman ρ between subset accuracy and full-benchmark accuracy\n"
        f"top-{int(top_frac*100)}% items by IIF at θ=0  |  "
        f"train={int((1-test_frac)*100)}% / test={int(test_frac*100)}% model split  |  "
        f"dashed = random-item baseline",
        fontsize=9,
    )

    for col, benchmark in enumerate(["MGSM", "MMLU"]):
        for row, (df, label) in enumerate([
            (df_insample, "In-sample (all models, reference)"),
            (df_held,     f"Held-out ({int((1-test_frac)*100)}/{int(test_frac*100)} model split, IRT refit)"),
        ]):
            ax = axes[row][col]
            sub = df[df["benchmark"] == benchmark]
            positions = np.arange(len(PL_MODELS))

            for pos, pl in zip(positions, PL_MODELS):
                pl_sub = sub[sub["pl"] == pl]
                vals = pl_sub["spearman_rho"].dropna().values
                rand_vals = pl_sub["random_baseline_rho"].dropna().values
                color = PL_COLORS[pl]

                ax.boxplot(
                    [vals],
                    positions=[pos],
                    widths=0.4,
                    patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    boxprops=dict(facecolor=color, alpha=0.4),
                    whiskerprops=dict(color=color),
                    capprops=dict(color=color),
                    flierprops=dict(marker="o", color=color, markersize=3, alpha=0.4),
                )
                rng_jit = np.random.default_rng(42)
                jitter = rng_jit.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(pos + jitter, vals, color=color, alpha=0.5, s=12, zorder=3)

                med = float(np.median(vals)) if len(vals) > 0 else float("nan")
                ax.text(pos, med + 0.01, f"{med:.3f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

                if len(rand_vals) > 0:
                    rand_mean = float(np.nanmean(rand_vals))
                    ax.hlines(rand_mean, pos - 0.22, pos + 0.22,
                              colors=color, linestyles="dashed", linewidth=1.5)

            n_slices = len(sub["slice"].unique()) if "slice" in sub.columns else len(sub)
            ax.set_title(f"{benchmark} — {label}  (n={n_slices} slices)", fontsize=8)
            ax.set_xticks(positions)
            ax.set_xticklabels(PL_MODELS, fontsize=10)
            ax.set_ylim(-0.5, 1.08)
            if col == 0:
                ax.set_ylabel("Spearman ρ", fontsize=10)
            ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
            ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, label="ρ=0.9")
            ax.axhline(0.7, color="gray", linestyle="-.", linewidth=0.8, label="ρ=0.7")
            ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Plotting: accuracy scatter (subset accuracy vs. full accuracy)
# ---------------------------------------------------------------------------

def plot_accuracy_scatter(
    df_pairs: pd.DataFrame,
    top_frac: float,
    output_path: Path,
    title_prefix: str = "Held-out",
) -> None:
    """
    2×3 grid (benchmark × IRT model).  Each point is one (model, slice) pair.
    x-axis = accuracy on the IRT-selected subset, y-axis = accuracy on the full
    benchmark.  The dashed diagonal is the ideal y=x line.  Stats (MAE, bias,
    R²) are annotated in each panel.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f"{title_prefix}: subset accuracy vs. full-benchmark accuracy\n"
        f"top-{int(top_frac * 100)}% items by IIF at θ=0  |  "
        f"each point = one (model, slice) pair  |  dashed = ideal y=x",
        fontsize=9,
    )

    for row_idx, benchmark in enumerate(["MGSM", "MMLU"]):
        for col_idx, pl in enumerate(PL_MODELS):
            ax = axes[row_idx][col_idx]
            sub = df_pairs[
                (df_pairs["benchmark"] == benchmark) & (df_pairs["pl"] == pl)
            ]
            if sub.empty:
                ax.set_visible(False)
                continue

            color = PL_COLORS[pl]
            ax.scatter(sub["acc_sub"], sub["acc_full"], color=color, alpha=0.25, s=10)

            # y = x reference line
            lo = min(sub["acc_sub"].min(), sub["acc_full"].min()) - 0.02
            hi = max(sub["acc_sub"].max(), sub["acc_full"].max()) + 0.02
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, alpha=0.6)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

            # Stats annotation
            resid = sub["acc_sub"].values - sub["acc_full"].values
            mae = float(np.mean(np.abs(resid)))
            bias = float(np.mean(resid))
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((sub["acc_full"].values - sub["acc_full"].mean()) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > EPS else float("nan")

            ax.text(
                0.05, 0.95,
                f"MAE  = {mae:.3f}\nBias = {bias:+.3f}\nR²   = {r2:.3f}",
                transform=ax.transAxes, va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
            )
            ax.set_title(f"{benchmark} — {pl}  (n={len(sub)})", fontsize=9)
            ax.set_xlabel("Subset accuracy", fontsize=8)
            ax.set_ylabel("Full accuracy", fontsize=8)
            ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Analysis 3: Cross-lingual ranking transfer (MGSM)
# ---------------------------------------------------------------------------

def run_crosslingual(
    results_root: Path,
    data_root: Path,
    top_frac: float,
    heldout_irt_dir: Path | None = None,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> pd.DataFrame:
    """
    Cross-lingual item transfer for MGSM and MMLU.

    MGSM: items are parallel across all 11 languages (same problem, translated).
      For each (src_lang, tgt_lang, PL): select top-k items by src IRT, evaluate
      in tgt language, compare vs. full tgt benchmark.

    MMLU: items are parallel across 7 languages *within the same domain*.
      For each (domain, src_lang, tgt_lang, PL): same logic, domain fixed.

    Returns a DataFrame with columns:
      benchmark, domain, pl, source_lang, target_lang, n_subset,
      spearman_rho, mae, rmse, bias, r2

    If heldout_irt_dir is provided, uses held-out mode: IRT loaded from
    heldout_irt_dir (refitted on train models), responses filtered to test
    models only (same deterministic split as Analysis 2).
    """
    rows: list[dict] = []
    irt_root = heldout_irt_dir if heldout_irt_dir is not None else results_root

    def _filter_test(resp: dict) -> dict:
        all_models = sorted(resp.keys())
        rng = np.random.default_rng(RANDOM_SEED)
        shuffled = rng.permutation(all_models).tolist()
        n_test = max(1, int(round(len(all_models) * test_frac)))
        test_models = set(shuffled[:n_test])
        return {m: r for m, r in resp.items() if m in test_models}

    # ------------------------------------------------------------------
    # MGSM
    # ------------------------------------------------------------------
    mgsm_responses: dict[str, dict[str, dict[str, int]]] = {}
    mgsm_params: dict[str, dict[str, dict]] = {}
    for lang in MGSM_LANGUAGES:
        stem = f"mgsm_{lang}"
        resp = load_responses(data_root, stem)
        if resp is not None:
            mgsm_responses[lang] = _filter_test(resp) if heldout_irt_dir is not None else resp
        for pl in PL_MODELS:
            params = load_irt_params(irt_root, pl, stem)
            if params is not None:
                mgsm_params.setdefault(pl, {})[lang] = params

    for pl in PL_MODELS:
        if pl not in mgsm_params:
            continue
        for src_lang in MGSM_LANGUAGES:
            if src_lang not in mgsm_params[pl] or src_lang not in mgsm_responses:
                continue
            top_ids = select_top_items(mgsm_params[pl][src_lang], pl, top_frac)
            for tgt_lang in MGSM_LANGUAGES:
                if tgt_lang not in mgsm_responses:
                    continue
                tgt_resp = mgsm_responses[tgt_lang]
                tgt_params = mgsm_params[pl].get(tgt_lang)
                if tgt_params is None:
                    continue
                all_item_ids = set(tgt_params["item_ids"])
                acc_full = model_accuracies(tgt_resp, all_item_ids)
                valid_ids = top_ids & all_item_ids
                if len(valid_ids) < 3:
                    continue
                acc_sub = model_accuracies(tgt_resp, valid_ids)
                rho = spearman_between(acc_full, acc_sub)
                acc_stats = accuracy_prediction_stats(acc_full, acc_sub)
                rows.append({
                    "benchmark": "MGSM",
                    "domain": None,
                    "pl": pl,
                    "source_lang": src_lang,
                    "target_lang": tgt_lang,
                    "n_subset": len(valid_ids),
                    "spearman_rho": rho,
                    **(acc_stats or {}),
                })

    # ------------------------------------------------------------------
    # MMLU: cross-language within the same domain
    # ------------------------------------------------------------------
    for domain in MMLU_DOMAINS:
        mmlu_responses: dict[str, dict[str, dict[str, int]]] = {}
        mmlu_params: dict[str, dict[str, dict]] = {}
        for lang in MMLU_LANGUAGES:
            stem = f"mmlu_{lang}_{domain}"
            resp = load_responses(data_root, stem)
            if resp is not None:
                mmlu_responses[lang] = _filter_test(resp) if heldout_irt_dir is not None else resp
            for pl in PL_MODELS:
                params = load_irt_params(irt_root, pl, stem)
                if params is not None:
                    mmlu_params.setdefault(pl, {})[lang] = params

        for pl in PL_MODELS:
            if pl not in mmlu_params:
                continue
            for src_lang in MMLU_LANGUAGES:
                if src_lang not in mmlu_params[pl] or src_lang not in mmlu_responses:
                    continue
                top_ids = select_top_items(mmlu_params[pl][src_lang], pl, top_frac)
                for tgt_lang in MMLU_LANGUAGES:
                    if tgt_lang not in mmlu_responses:
                        continue
                    tgt_resp = mmlu_responses[tgt_lang]
                    tgt_params = mmlu_params[pl].get(tgt_lang)
                    if tgt_params is None:
                        continue
                    all_item_ids = set(tgt_params["item_ids"])
                    acc_full = model_accuracies(tgt_resp, all_item_ids)
                    valid_ids = top_ids & all_item_ids
                    if len(valid_ids) < 3:
                        continue
                    acc_sub = model_accuracies(tgt_resp, valid_ids)
                    rho = spearman_between(acc_full, acc_sub)
                    acc_stats = accuracy_prediction_stats(acc_full, acc_sub)
                    rows.append({
                        "benchmark": "MMLU",
                        "domain": domain,
                        "pl": pl,
                        "source_lang": src_lang,
                        "target_lang": tgt_lang,
                        "n_subset": len(valid_ids),
                        "spearman_rho": rho,
                        **(acc_stats or {}),
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting: same-language box plots
# ---------------------------------------------------------------------------

def plot_same_language(df: pd.DataFrame, top_frac: float, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    fig.suptitle(
        f"Subset ranking fidelity — Spearman ρ (full vs. top-{int(top_frac*100)}% IIF subset)\n"
        f"Each dot is one benchmark slice; annotated values are medians. "
        f"Dashed lines = random-subset baseline (avg over {N_RANDOM_TRIALS} draws).",
        fontsize=9,
    )

    for ax, benchmark in zip(axes, ["MGSM", "MMLU"]):
        sub = df[df["benchmark"] == benchmark]
        positions = np.arange(len(PL_MODELS))
        for pos, pl in zip(positions, PL_MODELS):
            pl_sub = sub[sub["pl"] == pl]
            vals = pl_sub["spearman_rho"].dropna().values
            rand_vals = pl_sub["random_baseline_rho"].dropna().values
            color = PL_COLORS[pl]
            ax.boxplot(
                [vals],
                positions=[pos],
                widths=0.4,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=2),
                boxprops=dict(facecolor=color, alpha=0.4),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                flierprops=dict(marker="o", color=color, markersize=4, alpha=0.5),
            )
            rng = np.random.default_rng(42)
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(pos + jitter, vals, color=color, alpha=0.7, s=20, zorder=3)
            med = float(np.median(vals)) if len(vals) > 0 else float("nan")
            ax.text(pos, med + 0.01, f"{med:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

            # Random baseline tick mark
            if len(rand_vals) > 0:
                rand_mean = float(np.mean(rand_vals))
                ax.hlines(rand_mean, pos - 0.22, pos + 0.22,
                          colors=color, linestyles="dashed", linewidth=1.5)

        n_slices = len(sub["slice"].unique())
        ax.set_title(f"{benchmark}  (n={n_slices} slices)", fontsize=10)
        ax.set_xticks(positions)
        ax.set_xticklabels(PL_MODELS, fontsize=10)
        ax.set_ylim(-0.5, 1.05)
        ax.set_ylabel("Spearman ρ" if benchmark == "MGSM" else "", fontsize=10)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, label="ρ = 0.9")
        ax.axhline(0.7, color="gray", linestyle="-.", linewidth=0.8, label="ρ = 0.7")
        ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Plotting: cross-lingual heatmaps
# ---------------------------------------------------------------------------

def plot_crosslingual_heatmaps(
    df: pd.DataFrame,
    benchmark: str,
    top_frac: float,
    output_path: Path,
    title_prefix: str = "",
) -> None:
    """
    1×3 heatmap grid (one panel per PL model).
    For MGSM: 11×11 language matrix, one entry per (src, tgt) pair.
    For MMLU: 7×7 language matrix, entries averaged across domains.
    """
    sub_all = df[df["benchmark"] == benchmark]
    if benchmark == "MGSM":
        languages = [l for l in MGSM_LANGUAGES if l in sub_all["source_lang"].values]
    else:
        languages = [l for l in MMLU_LANGUAGES if l in sub_all["source_lang"].values]
    labels = [LANG_LABELS[l] for l in languages]
    n = len(languages)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    domain_note = " (averaged across domains)" if benchmark == "MMLU" else ""
    prefix = f"{title_prefix} " if title_prefix else ""
    fig.suptitle(
        f"{prefix}{benchmark} cross-lingual subset ranking fidelity — Spearman ρ{domain_note}\n"
        f"(top-{int(top_frac*100)}% items from *source* IRT evaluated in *target* language)\n"
        f"Row = source language (IRT fitted here), Col = target language (evaluated here). "
        f"Diagonal = same-language.",
        fontsize=9,
    )

    for ax, pl in zip(axes, PL_MODELS):
        sub = sub_all[sub_all["pl"] == pl]
        mat = np.full((n, n), np.nan)
        for i, src in enumerate(languages):
            for j, tgt in enumerate(languages):
                cell = sub[(sub["source_lang"] == src) & (sub["target_lang"] == tgt)]
                if not cell.empty:
                    # Average over domains for MMLU; single value for MGSM
                    mat[i, j] = cell["spearman_rho"].mean()

        off_diag = mat[~np.eye(n, dtype=bool)]
        avg_rho = np.nanmean(off_diag)

        ax.imshow(mat, vmin=-1, vmax=1, cmap=CMAP_RYG, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f"{pl}  (off-diag avg ρ = {avg_rho:.3f})", fontsize=10)

        for i in range(n):
            for j in range(n):
                if not np.isnan(mat[i, j]):
                    ax.text(
                        j, i, f"{mat[i, j]:.2f}",
                        ha="center", va="center", fontsize=5.5,
                        color="black",
                    )

    fig.subplots_adjust(right=0.88, wspace=0.35)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=CMAP_RYG, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Spearman ρ")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=Path("results"),
                        help="Root directory containing 1PL/, 2PL/, 3PL/ sub-dirs.")
    parser.add_argument("--data_root", type=Path, default=Path("data/irt"),
                        help="Directory containing *.jsonlines response files.")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("results/analysis/subset_ranking_fidelity"))
    parser.add_argument("--top_frac", type=float, default=0.5,
                        help="Fraction of items to select by IIF (default: 0.5).")
    parser.add_argument("--test_frac", type=float, default=0.2,
                        help="Fraction of models held out as test set (default: 0.2).")
    args = parser.parse_args()
    args.results_root = resolve_path(args.results_root)
    args.data_root = resolve_path(args.data_root)
    args.output_dir = resolve_path(args.output_dir)
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"top_frac  = {args.top_frac} (top {int(args.top_frac*100)}% of items)")
    print(f"test_frac = {args.test_frac} ({int(args.test_frac*100)}% models held out)")
    print(f"Results root : {args.results_root}")
    print(f"Data root    : {args.data_root}")
    print(f"Output dir   : {args.output_dir}")

    # ------------------------------------------------------------------
    # Analysis 1: in-sample fidelity (reference / upper bound)
    # ------------------------------------------------------------------
    print("\n=== Analysis 1: in-sample ranking fidelity (reference) ===")
    df_same, df_same_pairs = run_same_language(args.results_root, args.data_root, args.top_frac)

    df_same.to_csv(args.output_dir / "insample_fidelity.csv", index=False)
    print(f"  Saved insample_fidelity.csv  ({len(df_same)} rows)")
    df_same_pairs.to_csv(args.output_dir / "insample_accuracy_pairs.csv", index=False)
    print(f"  Saved insample_accuracy_pairs.csv  ({len(df_same_pairs)} rows)")

    summary_rows = []
    for benchmark in ["MGSM", "MMLU"]:
        for pl in PL_MODELS:
            sub = df_same[(df_same["benchmark"] == benchmark) & (df_same["pl"] == pl)]
            vals = sub["spearman_rho"].dropna()
            rand_vals = sub["random_baseline_rho"].dropna()
            summary_rows.append({
                "benchmark": benchmark, "pl": pl,
                "n_slices": len(vals),
                "mean_rho": vals.mean(), "median_rho": vals.median(),
                "min_rho": vals.min(), "max_rho": vals.max(),
                "random_mean_rho": rand_vals.mean(),
                "mean_mae": sub["mae"].mean(), "mean_rmse": sub["rmse"].mean(),
                "mean_bias": sub["bias"].mean(), "mean_r2": sub["r2"].mean(),
            })
    df_insample_summary = pd.DataFrame(summary_rows)
    df_insample_summary.to_csv(args.output_dir / "insample_summary.csv", index=False)
    print(df_insample_summary.to_string(index=False))

    plot_same_language(df_same, args.top_frac, args.output_dir / "insample_boxplot.png")
    plot_accuracy_scatter(
        df_same_pairs, args.top_frac,
        args.output_dir / "insample_accuracy_scatter.png",
        title_prefix="In-sample",
    )

    # ------------------------------------------------------------------
    # Analysis 2: held-out model fidelity (correct pipeline evaluation)
    # ------------------------------------------------------------------
    print(f"\n=== Analysis 2: held-out model fidelity ({int((1-args.test_frac)*100)}/{int(args.test_frac*100)} split + IRT refit) ===")
    irt_save_dir = args.output_dir / "heldout_irt"
    df_held, df_held_pairs = run_held_out_fidelity(
        args.results_root, args.data_root, args.top_frac, args.test_frac,
        irt_save_dir=irt_save_dir,
    )

    df_held.to_csv(args.output_dir / "heldout_fidelity.csv", index=False)
    print(f"  Saved heldout_fidelity.csv  ({len(df_held)} rows)")
    df_held_pairs.to_csv(args.output_dir / "heldout_accuracy_pairs.csv", index=False)
    print(f"  Saved heldout_accuracy_pairs.csv  ({len(df_held_pairs)} rows)")

    held_summary_rows = []
    for benchmark in ["MGSM", "MMLU"]:
        for pl in PL_MODELS:
            sub = df_held[(df_held["benchmark"] == benchmark) & (df_held["pl"] == pl)]
            vals = sub["spearman_rho"].dropna()
            rand_vals = sub["random_baseline_rho"].dropna()
            held_summary_rows.append({
                "benchmark": benchmark, "pl": pl,
                "n_slice_folds": len(vals),
                "mean_rho": vals.mean(), "median_rho": vals.median(),
                "min_rho": vals.min(), "max_rho": vals.max(),
                "random_mean_rho": rand_vals.mean(),
                "mean_mae": sub["mae"].mean(), "mean_rmse": sub["rmse"].mean(),
                "mean_bias": sub["bias"].mean(), "mean_r2": sub["r2"].mean(),
            })
    df_held_summary = pd.DataFrame(held_summary_rows)
    df_held_summary.to_csv(args.output_dir / "heldout_summary.csv", index=False)
    print(df_held_summary.to_string(index=False))

    plot_held_out_fidelity(
        df_held, df_same, args.top_frac, args.test_frac,
        args.output_dir / "heldout_vs_insample.png",
    )
    plot_accuracy_scatter(
        df_held_pairs, args.top_frac,
        args.output_dir / "heldout_accuracy_scatter.png",
        title_prefix="Held-out",
    )

    # ------------------------------------------------------------------
    # Analysis 3: cross-lingual ranking transfer
    # ------------------------------------------------------------------
    print("\n=== Analysis 3: in-sample cross-lingual ranking transfer (MGSM + MMLU) ===")
    df_cross = run_crosslingual(args.results_root, args.data_root, args.top_frac)

    df_cross.to_csv(args.output_dir / "crosslang_fidelity.csv", index=False)
    print(f"  Saved crosslang_fidelity.csv  ({len(df_cross)} rows)")

    crosslang_rows = []
    for benchmark in ["MGSM", "MMLU"]:
        for pl in PL_MODELS:
            sub = df_cross[(df_cross["benchmark"] == benchmark) & (df_cross["pl"] == pl)]
            off  = sub[sub["source_lang"] != sub["target_lang"]]
            diag = sub[sub["source_lang"] == sub["target_lang"]]
            crosslang_rows.append({
                "benchmark": benchmark,
                "pl": pl,
                "same_lang_mean_rho":    diag["spearman_rho"].mean(),
                "cross_lang_mean_rho":   off["spearman_rho"].mean(),
                "cross_lang_median_rho": off["spearman_rho"].median(),
                "cross_lang_min_rho":    off["spearman_rho"].min(),
                "cross_lang_max_rho":    off["spearman_rho"].max(),
                "cross_lang_mean_mae":   off["mae"].mean(),
                "cross_lang_mean_bias":  off["bias"].mean(),
                "cross_lang_mean_r2":    off["r2"].mean(),
                "same_lang_mean_mae":    diag["mae"].mean(),
                "same_lang_mean_bias":   diag["bias"].mean(),
                "same_lang_mean_r2":     diag["r2"].mean(),
            })
    df_cross_summary = pd.DataFrame(crosslang_rows)
    df_cross_summary.to_csv(args.output_dir / "crosslang_summary.csv", index=False)
    print(df_cross_summary.to_string(index=False))

    plot_crosslingual_heatmaps(
        df_cross, "MGSM", args.top_frac,
        args.output_dir / "crosslang_heatmap_mgsm.png",
        title_prefix="In-sample",
    )
    plot_crosslingual_heatmaps(
        df_cross, "MMLU", args.top_frac,
        args.output_dir / "crosslang_heatmap_mmlu.png",
        title_prefix="In-sample",
    )

    # ------------------------------------------------------------------
    # Analysis 3 (held-out): cross-lingual transfer with held-out models
    # ------------------------------------------------------------------
    print("\n=== Analysis 4: held-out cross-lingual ranking transfer (MGSM + MMLU) ===")
    df_cross_held = run_crosslingual(
        args.results_root, args.data_root, args.top_frac,
        heldout_irt_dir=irt_save_dir,
        test_frac=args.test_frac,
    )

    df_cross_held.to_csv(args.output_dir / "crosslang_heldout_fidelity.csv", index=False)
    print(f"  Saved crosslang_heldout_fidelity.csv  ({len(df_cross_held)} rows)")

    crosslang_held_rows = []
    for benchmark in ["MGSM", "MMLU"]:
        for pl in PL_MODELS:
            sub = df_cross_held[
                (df_cross_held["benchmark"] == benchmark) & (df_cross_held["pl"] == pl)
            ]
            off  = sub[sub["source_lang"] != sub["target_lang"]]
            diag = sub[sub["source_lang"] == sub["target_lang"]]
            crosslang_held_rows.append({
                "benchmark": benchmark,
                "pl": pl,
                "same_lang_mean_rho":    diag["spearman_rho"].mean(),
                "cross_lang_mean_rho":   off["spearman_rho"].mean(),
                "cross_lang_median_rho": off["spearman_rho"].median(),
                "cross_lang_min_rho":    off["spearman_rho"].min(),
                "cross_lang_max_rho":    off["spearman_rho"].max(),
                "cross_lang_mean_mae":   off["mae"].mean(),
                "cross_lang_mean_bias":  off["bias"].mean(),
                "cross_lang_mean_r2":    off["r2"].mean(),
                "same_lang_mean_mae":    diag["mae"].mean(),
                "same_lang_mean_bias":   diag["bias"].mean(),
                "same_lang_mean_r2":     diag["r2"].mean(),
            })
    df_cross_held_summary = pd.DataFrame(crosslang_held_rows)
    df_cross_held_summary.to_csv(args.output_dir / "crosslang_heldout_summary.csv", index=False)
    print(df_cross_held_summary.to_string(index=False))

    plot_crosslingual_heatmaps(
        df_cross_held, "MGSM", args.top_frac,
        args.output_dir / "crosslang_heldout_heatmap_mgsm.png",
        title_prefix="Held-out",
    )
    plot_crosslingual_heatmaps(
        df_cross_held, "MMLU", args.top_frac,
        args.output_dir / "crosslang_heldout_heatmap_mmlu.png",
        title_prefix="Held-out",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
