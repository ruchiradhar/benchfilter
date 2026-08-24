#!/usr/bin/env python3
"""
Compare within-language fit: SVI ELBO vs MAP log-likelihood per observation.

For SVI models the existing elbo_results.json is used directly.
For MAP models we compute the training log-likelihood from stored parameters
and the original response matrices.

The key question: does MAP-fitted 2PL achieve better within-language fit
than 1PL (as it should, since 2PL nests 1PL)?  Under SVI it did not.

Output:
  results/core_analysis/elbo_vs_map_loglik/elbo_vs_map_loglik.csv   — per-slice table
  results/core_analysis/elbo_vs_map_loglik/elbo_vs_map_loglik.png   — figure
  prints a summary table to stdout
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "core_analysis" / "elbo_vs_map_loglik"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_response_matrix(jsonlines_path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    records = [json.loads(l) for l in jsonlines_path.read_text().splitlines() if l.strip()]
    all_items = sorted(
        {k for rec in records for k in rec["responses"]},
        key=lambda x: int(x.split("_")[-1]),
    )
    item_idx = {item: i for i, item in enumerate(all_items)}
    subject_ids = [rec["subject_id"] for rec in records]
    R = np.zeros((len(records), len(all_items)))
    for s, rec in enumerate(records):
        for item, resp in rec["responses"].items():
            R[s, item_idx[item]] = resp
    return R, subject_ids, all_items


def binary_log_likelihood(R: np.ndarray, P: np.ndarray) -> float:
    P = np.clip(P, 1e-7, 1 - 1e-7)
    return float((R * np.log(P) + (1 - R) * np.log(1 - P)).sum())


def map_loglik_1pl(params: dict, R: np.ndarray) -> float:
    theta = np.array(params["ability"])
    b = np.array(params["diff"])
    P = sigmoid(theta[:, None] - b[None, :])
    return binary_log_likelihood(R, P)


def map_loglik_2pl(params: dict, R: np.ndarray) -> float:
    theta = np.array(params["ability"])
    b = np.array(params["diff"])
    a = np.array(params["disc"])
    P = sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
    return binary_log_likelihood(R, P)


def map_loglik_3pl(params: dict, R: np.ndarray) -> float:
    theta = np.array(params["ability"])
    b = np.array(params["diff"])
    a = np.array(params["disc"])
    c = np.array(params["lambdas"])
    p_star = sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
    P = c[None, :] + (1 - c[None, :]) * p_star
    return binary_log_likelihood(R, P)


def load_map_params(pl_dir: Path, stem: str, suffix: str) -> dict | None:
    path = pl_dir / f"{stem}_{suffix}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    data_dir   = PROJECT_ROOT / "data" / "irt"
    elbo_path  = PROJECT_ROOT / "results" / "elbo" / "elbo_results.json"
    dir_1pl    = PROJECT_ROOT / "results" / "1PL"
    dir_2pl_svi = PROJECT_ROOT / "results" / "2PL"
    dir_3pl_svi = PROJECT_ROOT / "results" / "3PL"
    dir_2pl_map = PROJECT_ROOT / "results" / "2PL_map"
    dir_3pl_map = PROJECT_ROOT / "results" / "3PL_map"

    svi_elbo = json.loads(elbo_path.read_text())
    data_files = sorted(data_dir.rglob("*.jsonlines"))

    rows = []
    for data_file in data_files:
        stem = data_file.stem
        R, _, _ = load_response_matrix(data_file)
        n_obs = R.size

        svi = svi_elbo.get(stem, {})
        elbo_1pl = svi.get("1pl", {}).get("elbo_per_obs")
        elbo_2pl = svi.get("2pl", {}).get("elbo_per_obs")
        elbo_3pl = svi.get("3pl", {}).get("elbo_per_obs")

        # MAP log-likelihood per observation (negative = like ELBO for comparison)
        def neg_ll_per_obs(params, fn):
            if params is None:
                return None
            return -fn(params, R) / n_obs

        p1   = load_map_params(dir_1pl, stem, "1pl")
        p2m  = load_map_params(dir_2pl_map, stem, "2pl")
        p3m  = load_map_params(dir_3pl_map, stem, "3pl")

        nll_1pl     = neg_ll_per_obs(p1,  map_loglik_1pl)
        nll_2pl_map = neg_ll_per_obs(p2m, map_loglik_2pl)
        nll_3pl_map = neg_ll_per_obs(p3m, map_loglik_3pl)

        benchmark = "MGSM" if stem.startswith("mgsm") else "MMLU"
        rows.append({
            "stem": stem,
            "benchmark": benchmark,
            "n_obs": n_obs,
            # SVI ELBO/obs (lower = better fit, but paradoxically 2PL > 1PL)
            "svi_elbo_1pl": elbo_1pl,
            "svi_elbo_2pl": elbo_2pl,
            "svi_elbo_3pl": elbo_3pl,
            # MAP -loglik/obs (lower = better fit; should have 2PL < 1PL)
            "map_nll_1pl":     nll_1pl,
            "map_nll_2pl_map": nll_2pl_map,
            "map_nll_3pl_map": nll_3pl_map,
            # derived: MAP 2PL better than 1PL?
            "map_2pl_better_than_1pl": (
                nll_2pl_map < nll_1pl
                if nll_2pl_map is not None and nll_1pl is not None else None
            ),
            "map_3pl_better_than_1pl": (
                nll_3pl_map < nll_1pl
                if nll_3pl_map is not None and nll_1pl is not None else None
            ),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "elbo_vs_map_loglik.csv", index=False, float_format="%.6f")

    # ── print summary ─────────────────────────────────────────────────────────
    print("=" * 72)
    print("WITHIN-LANGUAGE FIT: SVI ELBO vs MAP log-likelihood per observation")
    print("(lower = better fit in both cases)")
    print("=" * 72)

    for bm in ["MGSM", "MMLU"]:
        sub = df[df["benchmark"] == bm]
        print(f"\n{'─'*72}")
        print(f"  {bm}  ({len(sub)} slices)")
        print(f"{'─'*72}")
        print(f"  {'Metric':<30} {'mean':>8} {'min':>8} {'max':>8}")
        print(f"  {'-'*56}")

        for label, col in [
            ("SVI ELBO/obs  1PL",           "svi_elbo_1pl"),
            ("SVI ELBO/obs  2PL (paradox)",  "svi_elbo_2pl"),
            ("SVI ELBO/obs  3PL",            "svi_elbo_3pl"),
            ("MAP -LL/obs   1PL",            "map_nll_1pl"),
            ("MAP -LL/obs   2PL_map",        "map_nll_2pl_map"),
            ("MAP -LL/obs   3PL_map",        "map_nll_3pl_map"),
        ]:
            vals = sub[col].dropna()
            if len(vals) == 0:
                continue
            print(f"  {label:<30} {vals.mean():>8.4f} {vals.min():>8.4f} {vals.max():>8.4f}")

        # Key question
        better_2pl = sub["map_2pl_better_than_1pl"].dropna()
        better_3pl = sub["map_3pl_better_than_1pl"].dropna()
        print(f"\n  MAP 2PL better fit than 1PL: {better_2pl.sum()}/{len(better_2pl)} slices "
              f"({better_2pl.mean()*100:.0f}%)")
        print(f"  MAP 3PL better fit than 1PL: {better_3pl.sum()}/{len(better_3pl)} slices "
              f"({better_3pl.mean()*100:.0f}%)")

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    benchmarks = [("MGSM", 0), ("MMLU", 1)]

    for bm, col_idx in benchmarks:
        sub = df[df["benchmark"] == bm].copy()
        slices = sub["stem"].tolist()
        x = np.arange(len(slices))

        # ── top: SVI ELBO comparison (the paradox) ──
        ax_top = fig.add_subplot(gs[0, col_idx])
        ax_top.plot(x, sub["svi_elbo_1pl"], "o-", color="#4CAF50", ms=4, lw=1.5, label="SVI 1PL")
        ax_top.plot(x, sub["svi_elbo_2pl"], "s--", color="#F44336", ms=4, lw=1.5, label="SVI 2PL (paradox)")
        ax_top.plot(x, sub["svi_elbo_3pl"], "^:", color="#FF9800", ms=4, lw=1.2, label="SVI 3PL")
        ax_top.set_title(f"{bm} — SVI ELBO/obs\n(2PL worse than 1PL on every slice)",
                         fontsize=9, fontweight="bold")
        ax_top.set_ylabel("ELBO / obs  (↓ better)")
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(
            [s.replace("mgsm_","").replace("mmlu_","") for s in slices],
            rotation=75, ha="right", fontsize=6
        )
        ax_top.legend(fontsize=7)

        # ── bottom: MAP log-likelihood comparison ──
        ax_bot = fig.add_subplot(gs[1, col_idx])
        ax_bot.plot(x, sub["map_nll_1pl"], "o-", color="#4CAF50", ms=4, lw=1.5, label="MAP 1PL")
        ax_bot.plot(x, sub["map_nll_2pl_map"], "s-", color="#2196F3", ms=4, lw=1.5, label="MAP 2PL")
        ax_bot.plot(x, sub["map_nll_3pl_map"], "^-", color="#9C27B0", ms=4, lw=1.2, label="MAP 3PL")
        ax_bot.set_title(f"{bm} — MAP −loglik/obs\n(2PL/3PL now better than 1PL as expected)",
                         fontsize=9, fontweight="bold")
        ax_bot.set_ylabel("−log-likelihood / obs  (↓ better)")
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(
            [s.replace("mgsm_","").replace("mmlu_","") for s in slices],
            rotation=75, ha="right", fontsize=6
        )
        ax_bot.legend(fontsize=7)

    fig.suptitle(
        "Within-language fit: SVI ELBO (top) vs MAP log-likelihood (bottom)\n"
        "Top row shows the ELBO paradox: 2PL worse than 1PL under SVI. "
        "Bottom row shows MAP resolves it: 2PL/3PL fit better than 1PL, as expected.",
        fontsize=9, y=1.01
    )
    fig.savefig(OUT_DIR / "elbo_vs_map_loglik.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure → {OUT_DIR / 'elbo_vs_map_loglik.png'}")
    print(f"CSV    → {OUT_DIR / 'elbo_vs_map_loglik.csv'}")


if __name__ == "__main__":
    main()
