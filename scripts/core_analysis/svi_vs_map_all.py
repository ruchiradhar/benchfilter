#!/usr/bin/env python3
"""
Combined SVI vs MAP comparison for 1PL, 2PL, and 3PL.

Layout (2 rows x 3 cols of ρ histograms + 3 illustrative scatter panels):
  Row 1: 1PL diff | 2PL diff | 2PL disc
  Row 2: 3PL diff | 3PL disc | 3PL guess
  Row 3: scatter (1PL diff, high ρ) | scatter (2PL disc, collapsed) | scatter (3PL disc, collapsed)

Produces:
  images_2/svi_vs_map_all.pdf

Usage:
    python scripts/core_analysis/svi_vs_map_all.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "images_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS = PROJECT_ROOT / "results"


def load_params(path: Path) -> dict[str, np.ndarray]:
    d = json.loads(path.read_text())
    item_ids = [d["item_ids"][str(i)] for i in range(len(d["item_ids"]))]
    out: dict = {"item_ids": item_ids, "diff": np.array(d["diff"])}
    if "disc" in d:
        out["disc"] = np.array(d["disc"])
    if "lambdas" in d:
        out["guess"] = np.array(d["lambdas"])
    return out


def collect(svi_dir: Path, map_dir: Path, suffix: str, param: str):
    """Return (rhos, pairs) for one (model, parameter) combination."""
    rhos, pairs = [], []
    for svi_path in sorted(svi_dir.glob(f"*_{suffix}.json")):
        slice_name = svi_path.stem.replace(f"_{suffix}", "")
        map_path = map_dir / f"{slice_name}_{suffix}.json"
        if not map_path.exists():
            continue
        svi = load_params(svi_path)
        mp = load_params(map_path)
        if param not in svi or param not in mp:
            continue
        rho, _ = spearmanr(svi[param], mp[param])
        rhos.append(rho)
        pairs.append((slice_name, svi[param], mp[param]))
    return rhos, pairs


def draw_hist(ax, rhos, title, color="#4c72b0"):
    arr = np.array(rhos)
    ax.hist(arr, bins=15, color=color, edgecolor="white", linewidth=0.5)
    ax.axvline(arr.mean(), color="#dd8452", lw=1.5, ls="--")
    ax.text(0.97, 0.93, f"$\\bar{{\\rho}}={arr.mean():.3f}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color="#dd8452")
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman $\\rho$ (SVI vs MAP)", fontsize=8)
    ax.set_ylabel("Slices", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.axvline(0, color="grey", lw=0.7, ls=":", alpha=0.6)


def draw_scatter(ax, x, y, title, xlabel, ylabel):
    ax.scatter(x, y, s=6, alpha=0.45, color="#4c72b0", linewidths=0)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6)


def main() -> None:
    # ── collect all (rhos, pairs) ─────────────────────────────────────────────
    data = {}
    configs = [
        ("1pl_diff",  "1PL",     "1PL_map",  "1pl", "diff"),
        ("2pl_diff",  "2PL",     "2PL_map",  "2pl", "diff"),
        ("2pl_disc",  "2PL",     "2PL_map",  "2pl", "disc"),
        ("3pl_diff",  "3PL",     "3PL_map",  "3pl", "diff"),
        ("3pl_disc",  "3PL",     "3PL_map",  "3pl", "disc"),
        ("3pl_guess", "3PL",     "3PL_map",  "3pl", "guess"),
    ]
    for key, svi_label, map_label, suffix, param in configs:
        rhos, pairs = collect(
            RESULTS / svi_label, RESULTS / map_label, suffix, param
        )
        arr = np.array(rhos)
        print(f"{key:15s}  n={len(rhos):2d}  mean={arr.mean():.3f}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}")
        data[key] = (rhos, pairs)

    # ── figure layout ─────────────────────────────────────────────────────────
    # 2×3 grid of histograms
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.subplots_adjust(hspace=0.55, wspace=0.35)

    hist_specs = [
        # (row, col, key, title, color)
        (0, 0, "1pl_diff",  "1PL — difficulty",       "#4c72b0"),
        (0, 1, "2pl_diff",  "2PL — difficulty",       "#4c72b0"),
        (0, 2, "2pl_disc",  "2PL — discrimination",   "#c44e52"),
        (1, 0, "3pl_diff",  "3PL — difficulty",       "#4c72b0"),
        (1, 1, "3pl_disc",  "3PL — discrimination",   "#c44e52"),
        (1, 2, "3pl_guess", "3PL — guessing",         "#55a868"),
    ]
    for r, c, key, title, color in hist_specs:
        draw_hist(axes[r, c], data[key][0], title, color=color)

    fig.suptitle(
        "SVI vs MAP: difficulty, discrimination, and guessing (1PL, 2PL, 3PL)",
        fontsize=11, fontweight="bold",
    )

    out_path = OUT_DIR / "svi_vs_map_all.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")


if __name__ == "__main__":
    main()
