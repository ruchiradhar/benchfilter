#!/usr/bin/env python3
"""
Compare SVI-1PL and MAP-1PL difficulty estimates across all benchmark slices.

Produces:
  images_2/svi_vs_map_1pl_scatter.pdf  — per-slice Spearman ρ distribution
                                         + representative scatter panels
  stdout summary: mean/min/max Spearman ρ across all 53 slices

Usage:
    python scripts/core_analysis/svi_vs_map_1pl.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SVI_DIR = PROJECT_ROOT / "results" / "1PL"
MAP_DIR = PROJECT_ROOT / "results" / "1PL_map"
OUT_DIR = PROJECT_ROOT / "images_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_difficulty(path: Path) -> tuple[list[str], list[float]]:
    d = json.loads(path.read_text())
    item_ids = [d["item_ids"][str(i)] for i in range(len(d["item_ids"]))]
    return item_ids, d["diff"]


def main() -> None:
    svi_files = sorted(SVI_DIR.glob("*_1pl.json"))
    if not svi_files:
        raise FileNotFoundError(f"No SVI-1PL files in {SVI_DIR}")

    spearman_rhos: list[float] = []
    slice_names: list[str] = []
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []

    for svi_path in svi_files:
        slice_name = svi_path.stem.replace("_1pl", "")
        map_path = MAP_DIR / f"{slice_name}_1pl.json"
        if not map_path.exists():
            print(f"WARNING: missing MAP file for {slice_name}, skipping")
            continue

        svi_ids, svi_diff = load_difficulty(svi_path)
        map_ids, map_diff = load_difficulty(map_path)

        # align by item id
        svi_dict = dict(zip(svi_ids, svi_diff))
        map_dict = dict(zip(map_ids, map_diff))
        shared = sorted(set(svi_ids) & set(map_ids))
        if len(shared) < 10:
            print(f"WARNING: only {len(shared)} shared items for {slice_name}, skipping")
            continue

        x = np.array([svi_dict[k] for k in shared])
        y = np.array([map_dict[k] for k in shared])
        rho, _ = spearmanr(x, y)

        spearman_rhos.append(rho)
        slice_names.append(slice_name)
        pairs.append((slice_name, x, y))

    rhos = np.array(spearman_rhos)
    print(f"Slices compared: {len(rhos)}")
    print(f"Spearman ρ  mean={rhos.mean():.3f}  min={rhos.min():.3f}  max={rhos.max():.3f}")

    # ── Figure: histogram of ρ values + 4 representative scatter panels ──────
    # pick 4 representative slices (min, Q1, Q2, max)
    quantile_targets = [0.0, 0.33, 0.67, 1.0]
    rep_indices = [int(np.argmin(np.abs(rhos - np.quantile(rhos, q)))) for q in quantile_targets]

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.4)

    # top row: histogram spanning all 4 columns
    ax_hist = fig.add_subplot(gs[0, :])
    ax_hist.hist(rhos, bins=20, color="#4c72b0", edgecolor="white", linewidth=0.5)
    ax_hist.axvline(rhos.mean(), color="#dd8452", lw=1.5, ls="--",
                    label=f"mean $\\rho$ = {rhos.mean():.3f}")
    ax_hist.set_xlabel("Spearman $\\rho$ (SVI-1PL vs MAP-1PL difficulty)")
    ax_hist.set_ylabel("Number of slices")
    ax_hist.set_xlim(0, 1)
    ax_hist.legend(frameon=False, fontsize=9)
    ax_hist.set_title(
        f"Cross-inference consistency of 1PL difficulty estimates "
        f"($n={len(rhos)}$ slices)",
        fontsize=10,
    )

    # bottom row: 4 scatter panels
    for col, idx in enumerate(rep_indices):
        name, x, y = pairs[idx]
        rho_val = rhos[idx]
        ax = fig.add_subplot(gs[1, col])
        ax.scatter(x, y, s=6, alpha=0.5, color="#4c72b0", linewidths=0)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        short_name = name.replace("mgsm_", "").replace("mmlu_", "").replace("_", " ")
        ax.set_title(f"{short_name}\n$\\rho={rho_val:.3f}$", fontsize=8)
        ax.set_xlabel("SVI $b_j$", fontsize=7)
        ax.set_ylabel("MAP $b_j$", fontsize=7)
        ax.tick_params(labelsize=6)

    out_path = OUT_DIR / "svi_vs_map_1pl_scatter.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    main()
