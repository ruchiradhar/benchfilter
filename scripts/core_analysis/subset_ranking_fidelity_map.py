#!/usr/bin/env python3
"""Held-out subset ranking fidelity with MAP refitting for 2PL and 3PL.

Identical to subset_ranking_fidelity.py except the held-out IRT refitting
uses MAP (Adam + cosine annealing) for 2PL and 3PL instead of SVI.
This fixes the 3PL MGSM convergence failures seen with py-irt's SVI.

1PL still uses py-irt SVI (unchanged — it works reliably).

Saved held-out IRT params go to:
  results/core_analysis/subset_ranking_fidelity_map/heldout_irt/

All figures and CSVs saved to:
  results/core_analysis/subset_ranking_fidelity_map/
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

# -- allow importing from scripts/irt --
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "irt"))

import pyro
from py_irt.config import IrtConfig
from py_irt.dataset import Dataset
from py_irt.training import IrtModelTrainer

# ---------------------------------------------------------------------------
# Constants (same as original)
# ---------------------------------------------------------------------------

MGSM_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
MMLU_LANGUAGES = ["de", "en", "es", "fr", "ja", "sw", "zh"]
MMLU_DOMAINS   = ["business", "humanities", "medical", "other", "social_sciences", "stem"]
PL_MODELS      = ["1PL", "2PL", "3PL"]
PL_COLORS      = {"1PL": "steelblue", "2PL": "darkorange", "3PL": "seagreen"}
LANG_LABELS    = {
    "bn": "Bengali", "de": "German", "en": "English", "es": "Spanish",
    "fr": "French",  "ja": "Japanese","ru": "Russian", "sw": "Swahili",
    "te": "Telugu",  "th": "Thai",    "zh": "Chinese",
}
CMAP_RYG = LinearSegmentedColormap.from_list(
    "ryg", [(0.0, "#CC0000"), (0.5, "#FFFF00"), (1.0, "#007700")]
)

EPS              = 1e-8
RANDOM_SEED      = 42
N_RANDOM_TRIALS  = 200
DEFAULT_TEST_FRAC= 0.2
PL_DIR           = {"1PL": "1PL", "2PL": "2PL_map", "3PL": "3PL_map"}

# MAP hyperparameters for held-out refit
MAP_EPOCHS   = {"2PL": 4000, "3PL": 6000}
MAP_LR       = 0.05
MAP_RESTARTS = 3

OUTPUT_DIR   = PROJECT_ROOT / "results" / "core_analysis" / "subset_ranking_fidelity_map"
RESULTS_ROOT = PROJECT_ROOT / "results"
DATA_ROOT    = PROJECT_ROOT / "data" / "irt"


def resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# MAP model classes (from irt_map.py)
# ---------------------------------------------------------------------------

def _softplus_inv(x: float) -> float:
    return float(np.log(np.exp(x) - 1))


class TwoPL(torch.nn.Module):
    def __init__(self, n_subjects: int, n_items: int, device: torch.device) -> None:
        super().__init__()
        self.theta  = torch.nn.Parameter(torch.zeros(n_subjects, device=device))
        self.b      = torch.nn.Parameter(torch.zeros(n_items, device=device))
        self.raw_a  = torch.nn.Parameter(
            torch.full((n_items,), _softplus_inv(1.0), device=device))

    @property
    def a(self) -> torch.Tensor:
        return F.softplus(self.raw_a)

    def log_likelihood(self, s, i, r):
        logits = self.a[i] * (self.theta[s] - self.b[i])
        return -F.binary_cross_entropy_with_logits(logits, r, reduction="sum")

    def log_prior(self) -> torch.Tensor:
        log_a = torch.log(self.a)
        lp  = -0.5 * (self.theta ** 2).sum()
        lp += -0.5 * (self.b ** 2).sum()
        lp += -0.5 * ((log_a / 0.5) ** 2).sum() - log_a.sum()
        return lp


class ThreePL(torch.nn.Module):
    def __init__(self, n_subjects: int, n_items: int, device: torch.device) -> None:
        super().__init__()
        self.theta   = torch.nn.Parameter(torch.zeros(n_subjects, device=device))
        self.b       = torch.nn.Parameter(torch.zeros(n_items, device=device))
        self.raw_a   = torch.nn.Parameter(
            torch.full((n_items,), _softplus_inv(1.0), device=device))
        self.logit_c = torch.nn.Parameter(
            torch.full((n_items,), float(np.log(0.25 / 0.75)), device=device))

    @property
    def a(self) -> torch.Tensor:
        return F.softplus(self.raw_a)

    @property
    def c(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_c)

    def log_likelihood(self, s, i, r):
        p_star = torch.sigmoid(self.a[i] * (self.theta[s] - self.b[i]))
        p = self.c[i] + (1 - self.c[i]) * p_star
        p = p.clamp(1e-7, 1 - 1e-7)
        return (r * torch.log(p) + (1 - r) * torch.log(1 - p)).sum()

    def log_prior(self) -> torch.Tensor:
        log_a = torch.log(self.a)
        lp  = -0.5 * (self.theta ** 2).sum()
        lp += -0.5 * (self.b ** 2).sum()
        lp += -0.5 * ((log_a / 0.5) ** 2).sum() - log_a.sum()
        lp += -0.5 * ((self.logit_c + 2.0) ** 2).sum()
        return lp


def _train_one_map(model_cls, s_idx, i_idx, resp, n_s, n_i,
                   epochs, lr, device, seed):
    torch.manual_seed(seed)
    model = model_cls(n_s, n_i, device)
    with torch.no_grad():
        model.theta.add_(torch.randn_like(model.theta) * 0.5)
        model.b.add_(torch.randn_like(model.b) * 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)
    best_loss  = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = -(model.log_likelihood(s_idx, i_idx, resp) + model.log_prior())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_loss


# ---------------------------------------------------------------------------
# IRT utilities
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def iif_1pl(diff):
    p = sigmoid(-diff); return p * (1 - p)

def iif_2pl(diff, disc):
    p = sigmoid(disc * (-diff)); return disc**2 * p * (1 - p)

def iif_3pl(diff, disc, lam):
    p_star = sigmoid(disc * (-diff))
    p = lam + (1 - lam) * p_star
    p = np.clip(p, EPS, 1 - EPS)
    lam_c = np.clip(lam, 0, 1 - EPS)
    return disc**2 * (p - lam_c)**2 * (1 - p) / ((1 - lam_c)**2 * p)

def compute_iif(params: dict, pl: str) -> np.ndarray:
    if pl == "1PL": return iif_1pl(params["diff"])
    if pl == "2PL": return iif_2pl(params["diff"], params["disc"])
    return iif_3pl(params["diff"], params["disc"], params["lambdas"])

def select_top_items(params: dict, pl: str, top_frac: float) -> set[str]:
    info = compute_iif(params, pl)
    k = max(1, int(round(len(params["item_ids"]) * top_frac)))
    return {params["item_ids"][i] for i in np.argsort(info)[::-1][:k]}

def load_irt_params(results_root: Path, pl: str, stem: str) -> dict | None:
    path = results_root / PL_DIR[pl] / f"{stem}_{pl.lower()}.json"
    if not path.exists(): return None
    data = json.loads(path.read_text())
    n    = len(data["diff"])
    out  = {"item_ids": [data["item_ids"][str(i)] for i in range(n)],
            "diff": np.array(data["diff"], dtype=float)}
    if "disc"    in data: out["disc"]    = np.array(data["disc"],    dtype=float)
    if "lambdas" in data: out["lambdas"] = np.array(data["lambdas"], dtype=float)
    return out

def load_saved_params(save_dir: Path, pl: str, stem: str) -> dict | None:
    path = save_dir / pl / f"{stem}_{pl.lower()}.json"
    return load_irt_params(save_dir, pl, stem) if False else (
        _load_raw(path) if path.exists() else None)

def _load_raw(path: Path) -> dict | None:
    if not path.exists(): return None
    data = json.loads(path.read_text())
    n    = len(data["diff"])
    out  = {"item_ids": [data["item_ids"][str(i)] for i in range(n)],
            "diff": np.array(data["diff"], dtype=float)}
    if "disc"    in data: out["disc"]    = np.array(data["disc"],    dtype=float)
    if "lambdas" in data: out["lambdas"] = np.array(data["lambdas"], dtype=float)
    return out

def load_responses(data_root: Path, stem: str) -> dict | None:
    path = data_root / f"{stem}.jsonlines"
    if not path.exists(): return None
    resp = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            resp[obj["subject_id"]] = obj["responses"]
    return resp

def model_accuracies(responses: dict, item_ids: set) -> dict:
    out = {}
    for subject, resp in responses.items():
        scores = [resp[iid] for iid in item_ids if iid in resp]
        if scores: out[subject] = float(np.mean(scores))
    return out

def spearman_between(a: dict, b: dict) -> float | None:
    common = sorted(a.keys() & b.keys())
    if len(common) < 3: return None
    x = np.array([a[m] for m in common])
    y = np.array([b[m] for m in common])
    if np.std(x) < EPS or np.std(y) < EPS: return None
    rho, _ = spearmanr(x, y)
    return float(rho)

def accuracy_prediction_stats(a: dict, b: dict) -> dict | None:
    common = sorted(a.keys() & b.keys())
    if len(common) < 2: return None
    y_true = np.array([a[m] for m in common])
    y_pred = np.array([b[m] for m in common])
    res = y_pred - y_true
    ss_res = float(np.sum(res**2))
    ss_tot = float(np.sum((y_true - y_true.mean())**2))
    return {"mae": float(np.mean(np.abs(res))), "rmse": float(np.sqrt(np.mean(res**2))),
            "bias": float(np.mean(res)), "r2": float(1 - ss_res/ss_tot) if ss_tot > EPS else float("nan")}

def random_baseline_rho(acc_full, responses, all_item_ids, k,
                        n_trials=N_RANDOM_TRIALS, seed=RANDOM_SEED) -> float | None:
    rng   = np.random.default_rng(seed)
    arr   = np.array(all_item_ids)
    rhos  = []
    for _ in range(n_trials):
        chosen = set(rng.choice(arr, size=k, replace=False).tolist())
        rho    = spearman_between(acc_full, model_accuracies(responses, chosen))
        if rho is not None: rhos.append(rho)
    return float(np.mean(rhos)) if rhos else None


# ---------------------------------------------------------------------------
# Refit: SVI for 1PL, MAP for 2PL / 3PL
# ---------------------------------------------------------------------------

def refit_1pl_svi(train_responses: dict, seed: int, save_path: Path | None) -> dict | None:
    """Refit 1PL using py-irt SVI (unchanged from original)."""
    if save_path is not None and save_path.exists():
        return _load_raw(save_path)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonlines", delete=False) as fh:
        tmp = Path(fh.name)
        for sid, resp in train_responses.items():
            fh.write(json.dumps({"subject_id": sid, "responses": resp}) + "\n")
    try:
        pyro.clear_param_store()
        if seed is not None:
            pyro.set_rng_seed(seed); torch.manual_seed(seed)
        dataset = Dataset.from_jsonlines(tmp)
        config  = IrtConfig(model_type="1pl", epochs=2000, dropout=0.5, lr=0.1, lr_decay=0.9999)
        trainer = IrtModelTrainer(config=config, data_path=tmp, dataset=dataset, verbose=False)
        trainer.train(epochs=2000, device="cpu")
        raw = trainer.best_params or trainer.last_params
        n   = len(raw["diff"])
        iids = raw["item_ids"]
        item_ids = list(iids) if isinstance(iids, list) else [iids[i] if i in iids else iids[str(i)] for i in range(n)]
        out = {"item_ids": item_ids, "diff": np.array(raw["diff"], dtype=float)}
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json.dumps({
                "irt_model": "1pl",
                "item_ids": {str(i): v for i, v in enumerate(item_ids)},
                "diff": out["diff"].tolist(),
            }, indent=2))
        return out
    except Exception as e:
        print(f"    [refit warning] 1PL SVI failed: {e}")
        return None
    finally:
        tmp.unlink(missing_ok=True)


def refit_map(train_responses: dict, pl: str, seed: int, save_path: Path | None) -> dict | None:
    """Refit 2PL or 3PL using MAP (Adam + cosine annealing, multiple restarts)."""
    if save_path is not None and save_path.exists():
        return _load_raw(save_path)

    # Build response tensor
    all_items: list[str] = []
    for resp in train_responses.values():
        for k in resp:
            if k not in all_items: all_items.append(k)
    all_items = sorted(all_items, key=lambda x: int(x.split("_")[-1]))
    item_idx  = {item: i for i, item in enumerate(all_items)}
    subject_ids = list(train_responses.keys())
    n_s, n_i  = len(subject_ids), len(all_items)

    device = torch.device("cpu")
    R = torch.zeros(n_s, n_i, device=device)
    for s_idx, sid in enumerate(subject_ids):
        for item, r in train_responses[sid].items():
            R[s_idx, item_idx[item]] = float(r)

    s_flat = torch.arange(n_s, device=device).repeat_interleave(n_i)
    i_flat = torch.arange(n_i, device=device).repeat(n_s)
    r_flat = R.flatten()

    model_cls = TwoPL if pl == "2PL" else ThreePL
    epochs    = MAP_EPOCHS[pl]

    best_model = None
    best_loss  = float("inf")
    for restart in range(MAP_RESTARTS):
        m, loss = _train_one_map(
            model_cls, s_flat, i_flat, r_flat, n_s, n_i,
            epochs=epochs, lr=MAP_LR, device=device, seed=seed + restart * 42)
        if loss < best_loss:
            best_loss, best_model = loss, m

    with torch.no_grad():
        diff = best_model.b.cpu().numpy()
        disc = best_model.a.cpu().numpy()

    out: dict = {"item_ids": all_items, "diff": diff, "disc": disc}
    if pl == "3PL":
        with torch.no_grad():
            out["lambdas"] = best_model.c.cpu().numpy()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            "irt_model": pl.lower(),
            "item_ids": {str(i): v for i, v in enumerate(all_items)},
            "diff": diff.tolist(), "disc": disc.tolist(),
        }
        if "lambdas" in out: payload["lambdas"] = out["lambdas"].tolist()
        save_path.write_text(json.dumps(payload, indent=2))

    return out


def refit_irt(train_responses: dict, pl: str, seed: int, save_path: Path | None) -> dict | None:
    if pl == "1PL":
        return refit_1pl_svi(train_responses, seed, save_path)
    return refit_map(train_responses, pl, seed, save_path)


# ---------------------------------------------------------------------------
# Held-out analysis
# ---------------------------------------------------------------------------

def run_held_out_fidelity(
    results_root: Path,
    data_root: Path,
    top_frac: float,
    irt_save_dir: Path,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, pair_rows = [], []
    slices = (
        [("MGSM", f"mgsm_{lang}") for lang in MGSM_LANGUAGES]
        + [("MMLU", f"mmlu_{lang}_{dom}") for lang in MMLU_LANGUAGES for dom in MMLU_DOMAINS]
    )

    for benchmark, stem in slices:
        responses = load_responses(data_root, stem)
        if responses is None: continue

        all_models = sorted(responses.keys())
        rng        = np.random.default_rng(RANDOM_SEED)
        shuffled   = rng.permutation(all_models).tolist()
        n_test     = max(1, int(round(len(all_models) * test_frac)))
        test_models  = shuffled[:n_test]
        train_models = shuffled[n_test:]
        train_resp   = {m: responses[m] for m in train_models}
        test_resp    = {m: responses[m] for m in test_models}

        ref_params = load_irt_params(results_root, "1PL", stem)
        if ref_params is None: continue
        all_item_ids  = ref_params["item_ids"]
        acc_full_test = model_accuracies(test_resp, set(all_item_ids))

        print(f"  {stem}: train={len(train_models)}, test={len(test_models)}", end="")
        for pl in PL_MODELS:
            print(f"  [{pl}]", end="", flush=True)
            save_path = irt_save_dir / pl / f"{stem}_{pl.lower()}.json"
            train_params = refit_irt(train_resp, pl, RANDOM_SEED, save_path)
            if train_params is None: continue

            top_ids = select_top_items(train_params, pl, top_frac)
            k       = len(top_ids)
            acc_sub = model_accuracies(test_resp, top_ids)
            rho     = spearman_between(acc_full_test, acc_sub)
            stats   = accuracy_prediction_stats(acc_full_test, acc_sub)
            rho_rand= random_baseline_rho(acc_full_test, test_resp, all_item_ids, k)

            rows.append({
                "benchmark": benchmark, "slice": stem, "pl": pl,
                "n_train": len(train_models), "n_test": len(test_models),
                "n_items": len(all_item_ids), "k": k,
                "spearman_rho": rho, "random_baseline_rho": rho_rand,
                **(stats or {}),
            })
            for m in sorted(acc_full_test.keys() & acc_sub.keys()):
                pair_rows.append({"benchmark": benchmark, "slice": stem, "pl": pl,
                                  "model": m, "acc_full": acc_full_test[m],
                                  "acc_sub": acc_sub[m]})
        print()

    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


# ---------------------------------------------------------------------------
# Figure: heldout vs insample box plots (same style as original)
# ---------------------------------------------------------------------------

def plot_accuracy_scatter(
    df_pairs: pd.DataFrame,
    output_path: Path,
    title_prefix: str = "Held-out",
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for row_idx, benchmark in enumerate(["MGSM", "MMLU"]):
        for col_idx, pl in enumerate(PL_MODELS):
            ax = axes[row_idx][col_idx]
            sub = df_pairs[(df_pairs["benchmark"] == benchmark) & (df_pairs["pl"] == pl)]
            if sub.empty:
                ax.set_visible(False)
                continue
            color = PL_COLORS[pl]
            ax.scatter(sub["acc_sub"], sub["acc_full"], color=color, alpha=0.25, s=10)
            lo = min(sub["acc_sub"].min(), sub["acc_full"].min()) - 0.02
            hi = max(sub["acc_sub"].max(), sub["acc_full"].max()) + 0.02
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, alpha=0.6)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            resid = sub["acc_sub"].values - sub["acc_full"].values
            mae  = float(np.mean(np.abs(resid)))
            bias = float(np.mean(resid))
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((sub["acc_full"].values - sub["acc_full"].mean()) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > EPS else float("nan")
            ax.text(0.05, 0.95,
                    f"MAE  = {mae:.3f}\nBias = {bias:+.3f}\nR²   = {r2:.3f}",
                    transform=ax.transAxes, va="top", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))
            ax.set_xlabel("Subset accuracy", fontsize=8)
            ax.set_ylabel("Full accuracy", fontsize=8)
            ax.set_aspect("equal", adjustable="box")
            if row_idx == 0:
                ax.set_title(f"{title_prefix} — {pl}", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


def plot_heldout_vs_insample(
    df_held: pd.DataFrame,
    df_insample: pd.DataFrame,
    top_frac: float,
    test_frac: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    for col, benchmark in enumerate(["MGSM", "MMLU"]):
        for row, (df, label) in enumerate([
            (df_insample, "In-sample (all models, reference)"),
            (df_held, f"Held-out ({int((1-test_frac)*100)}/{int(test_frac*100)} split, MAP refit)"),
        ]):
            ax  = axes[row][col]
            sub = df[df["benchmark"] == benchmark]
            positions = np.arange(len(PL_MODELS))
            for pos, pl in zip(positions, PL_MODELS):
                pl_sub = sub[sub["pl"] == pl]
                vals   = pl_sub["spearman_rho"].dropna().values
                rand_v = pl_sub["random_baseline_rho"].dropna().values
                color  = PL_COLORS[pl]
                ax.boxplot([vals], positions=[pos], widths=0.4, patch_artist=True,
                           medianprops=dict(color="black", linewidth=2),
                           boxprops=dict(facecolor=color, alpha=0.4),
                           whiskerprops=dict(color=color), capprops=dict(color=color),
                           flierprops=dict(marker="o", color=color, markersize=3, alpha=0.4))
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(pos + jitter, vals, color=color, alpha=0.5, s=12, zorder=3)
                if len(vals):
                    med = float(np.median(vals))
                    ax.text(pos, med + 0.01, f"{med:.3f}",
                            ha="center", va="bottom", fontsize=8, fontweight="bold")
                if len(rand_v):
                    ax.hlines(float(np.nanmean(rand_v)), pos-0.22, pos+0.22,
                              colors=color, linestyles="dashed", linewidth=1.5)
            ax.set_xticks(positions)
            ax.set_xticklabels(PL_MODELS, fontsize=10)
            ax.set_ylim(-0.5, 1.08)
            if col == 0:
                ax.set_ylabel("Spearman ρ", fontsize=10)
            if row == 0:
                ax.set_title(f"{benchmark}", fontsize=11, fontweight="bold")
            ax.set_xlabel(label, fontsize=8)
            ax.axhline(1.0, color="gray", linestyle=":",  linewidth=0.8)
            ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, label="ρ=0.9")
            ax.axhline(0.7, color="gray", linestyle="-.", linewidth=0.8, label="ρ=0.7")
            ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Cross-lingual analysis
# ---------------------------------------------------------------------------

def load_heldout_params(irt_save_dir: Path, pl: str, stem: str) -> dict | None:
    """Load IRT params from held-out refit dir (no PL_DIR suffix mapping)."""
    path = irt_save_dir / pl / f"{stem}_{pl.lower()}.json"
    return _load_raw(path)


def run_crosslingual(
    results_root: Path,
    data_root: Path,
    top_frac: float,
    heldout_irt_dir: Path | None = None,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> pd.DataFrame:
    rows: list[dict] = []

    def _load_params(pl: str, stem: str) -> dict | None:
        if heldout_irt_dir is not None:
            return load_heldout_params(heldout_irt_dir, pl, stem)
        return load_irt_params(results_root, pl, stem)

    def _filter_test(resp: dict) -> dict:
        all_models = sorted(resp.keys())
        rng = np.random.default_rng(RANDOM_SEED)
        shuffled = rng.permutation(all_models).tolist()
        n_test = max(1, int(round(len(all_models) * test_frac)))
        test_models = set(shuffled[:n_test])
        return {m: r for m, r in resp.items() if m in test_models}

    # MGSM
    mgsm_responses: dict[str, dict] = {}
    mgsm_params: dict[str, dict[str, dict]] = {}
    for lang in MGSM_LANGUAGES:
        stem = f"mgsm_{lang}"
        resp = load_responses(data_root, stem)
        if resp is not None:
            mgsm_responses[lang] = _filter_test(resp) if heldout_irt_dir else resp
        for pl in PL_MODELS:
            p = _load_params(pl, stem)
            if p is not None:
                mgsm_params.setdefault(pl, {})[lang] = p

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
                    "benchmark": "MGSM", "domain": None, "pl": pl,
                    "source_lang": src_lang, "target_lang": tgt_lang,
                    "n_subset": len(valid_ids), "spearman_rho": rho,
                    **(acc_stats or {}),
                })

    # MMLU: cross-language within the same domain
    for domain in MMLU_DOMAINS:
        mmlu_responses: dict[str, dict] = {}
        mmlu_params: dict[str, dict[str, dict]] = {}
        for lang in MMLU_LANGUAGES:
            stem = f"mmlu_{lang}_{domain}"
            resp = load_responses(data_root, stem)
            if resp is not None:
                mmlu_responses[lang] = _filter_test(resp) if heldout_irt_dir else resp
            for pl in PL_MODELS:
                p = _load_params(pl, stem)
                if p is not None:
                    mmlu_params.setdefault(pl, {})[lang] = p

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
                        "benchmark": "MMLU", "domain": domain, "pl": pl,
                        "source_lang": src_lang, "target_lang": tgt_lang,
                        "n_subset": len(valid_ids), "spearman_rho": rho,
                        **(acc_stats or {}),
                    })

    return pd.DataFrame(rows)


def plot_crosslingual_heatmaps(
    df: pd.DataFrame,
    benchmark: str,
    top_frac: float,
    output_path: Path,
    title_prefix: str = "",
) -> None:
    sub_all = df[df["benchmark"] == benchmark]
    if benchmark == "MGSM":
        languages = [l for l in MGSM_LANGUAGES if l in sub_all["source_lang"].values]
    else:
        languages = [l for l in MMLU_LANGUAGES if l in sub_all["source_lang"].values]
    labels = [LANG_LABELS[l] for l in languages]
    n = len(languages)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, pl in zip(axes, PL_MODELS):
        sub = sub_all[sub_all["pl"] == pl]
        mat = np.full((n, n), np.nan)
        for i, src in enumerate(languages):
            for j, tgt in enumerate(languages):
                cell = sub[(sub["source_lang"] == src) & (sub["target_lang"] == tgt)]
                if not cell.empty:
                    mat[i, j] = cell["spearman_rho"].mean()

        ax.imshow(mat, vmin=-1, vmax=1, cmap=CMAP_RYG, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        title = f"{title_prefix}{pl}" if title_prefix else pl
        ax.set_title(title, fontsize=11, fontweight="bold")

        for i in range(n):
            for j in range(n):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}",
                            ha="center", va="center", fontsize=5.5, color="black")

    fig.subplots_adjust(right=0.88, wspace=0.35)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=CMAP_RYG, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Spearman ρ")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    irt_save_dir = OUTPUT_DIR / "heldout_irt"

    # Load pre-computed in-sample results (no need to rerun)
    insample_path = RESULTS_ROOT / "core_analysis" / "subset_ranking_fidelity" / "insample_fidelity.csv"
    if insample_path.exists():
        df_insample = pd.read_csv(insample_path)
        print(f"Loaded in-sample results from {insample_path}")
    else:
        print("WARNING: insample_fidelity.csv not found — run original script first.")
        df_insample = pd.DataFrame()

    print("\n=== Held-out ranking fidelity (MAP refit for 2PL/3PL) ===")
    df_held, df_held_pairs = run_held_out_fidelity(
        RESULTS_ROOT, DATA_ROOT, top_frac=0.5,
        irt_save_dir=irt_save_dir, test_frac=DEFAULT_TEST_FRAC,
    )

    df_held.to_csv(OUTPUT_DIR / "heldout_fidelity.csv", index=False)
    df_held_pairs.to_csv(OUTPUT_DIR / "heldout_accuracy_pairs.csv", index=False)
    print(f"\nSaved heldout_fidelity.csv ({len(df_held)} rows)")

    plot_accuracy_scatter(
        df_held_pairs,
        output_path=OUTPUT_DIR / "heldout_accuracy_scatter.png",
        title_prefix="Held-out (MAP)",
    )

    # Summary table
    print("\n=== Median Spearman ρ (held-out, MAP refit) ===")
    summary = (df_held.groupby(["benchmark", "pl"])["spearman_rho"]
               .agg(["median", "mean", "count"]).round(3))
    print(summary)
    summary.to_csv(OUTPUT_DIR / "heldout_summary.csv")

    # Figure
    if not df_insample.empty:
        plot_heldout_vs_insample(
            df_held, df_insample, top_frac=0.5,
            test_frac=DEFAULT_TEST_FRAC,
            output_path=OUTPUT_DIR / "heldout_vs_insample_map.png",
        )

    # Cross-lingual analysis
    print("\n=== Cross-lingual ranking fidelity ===")
    print("  In-sample (full IRT params)...")
    df_cl_insample = run_crosslingual(
        RESULTS_ROOT, DATA_ROOT, top_frac=0.5,
        heldout_irt_dir=None,
    )
    df_cl_insample.to_csv(OUTPUT_DIR / "crosslingual_insample.csv", index=False)
    print(f"  Saved crosslingual_insample.csv ({len(df_cl_insample)} rows)")

    print("  Held-out (MAP refit IRT params)...")
    df_cl_heldout = run_crosslingual(
        RESULTS_ROOT, DATA_ROOT, top_frac=0.5,
        heldout_irt_dir=irt_save_dir,
    )
    df_cl_heldout.to_csv(OUTPUT_DIR / "crosslingual_heldout.csv", index=False)
    print(f"  Saved crosslingual_heldout.csv ({len(df_cl_heldout)} rows)")

    for df_cl, label in [(df_cl_insample, "insample"), (df_cl_heldout, "heldout")]:
        if df_cl.empty:
            continue
        for bm in ["MGSM", "MMLU"]:
            fname = f"crosslingual_{label}_{bm.lower()}.png"
            plot_crosslingual_heatmaps(
                df_cl, bm, top_frac=0.5,
                output_path=OUTPUT_DIR / fname,
                title_prefix=f"{label.capitalize()} — ",
            )

    # Summary cross-lingual ρ
    if not df_cl_heldout.empty:
        print("\n=== Median cross-lingual Spearman ρ (off-diagonal, held-out) ===")
        off = df_cl_heldout[df_cl_heldout["source_lang"] != df_cl_heldout["target_lang"]]
        summary_cl = off.groupby(["benchmark", "pl"])["spearman_rho"].agg(
            ["median", "mean", "count"]).round(3)
        print(summary_cl)


if __name__ == "__main__":
    main()
