#!/usr/bin/env python3
"""
Compare SVI vs MAP difficulty and discrimination estimates for 2PL and 3PL.

Produces:
  images/svi_vs_map_2pl_3pl.pdf  — per-parameter ρ distributions + scatter panels
                                      showing discrimination collapse under SVI

Usage:
    python scripts/core_analysis/svi_vs_map_2pl_3pl.py
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
OUT_DIR = PROJECT_ROOT / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_params(path: Path) -> dict[str, np.ndarray]:
    d = json.loads(path.read_text())
    item_ids = [d["item_ids"][str(i)] for i in range(len(d["item_ids"]))]
    out = {"item_ids": item_ids, "diff": np.array(d["diff"])}
    if "disc" in d:
        out["disc"] = np.array(d["disc"])
    if "lambdas" in d:
        out["guess"] = np.array(d["lambdas"])
    return out


def collect_rhos(
    svi_dir: Path,
    map_dir: Path,
    suffix: str,
    param: str,
) -> tuple[list[float], list[tuple[str, np.ndarray, np.ndarray]]]:
    """Return (rhos, pairs) across all slices for a given parameter."""
    rhos: list[float] = []
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []

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


def pick_reps(rhos: list[float], pairs: list, n: int = 3) -> list[int]:
    """Pick n representative indices spread across the ρ range."""
    arr = np.array(rhos)
    targets = np.linspace(arr.min(), arr.max(), n)
    return [int(np.argmin(np.abs(arr - t))) for t in targets]


def add_hist(ax: plt.Axes, rhos: list[float], title: str, xlim: tuple) -> None:
    arr = np.array(rhos)
    ax.hist(arr, bins=15, color="#4c72b0", edgecolor="white", linewidth=0.5)
    ax.axvline(arr.mean(), color="#dd8452", lw=1.5, ls="--",
               label=f"mean $\\rho={arr.mean():.3f}$")
    ax.set_xlim(*xlim)
    ax.set_xlabel("Spearman $\\rho$", fontsize=8)
    ax.set_ylabel("Slices", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(labelsize=7)


def add_scatter(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    name: str,
    rho: float,
    xlabel: str,
    ylabel: str,
) -> None:
    ax.scatter(x, y, s=5, alpha=0.5, color="#4c72b0", linewidths=0)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.5)
    short = name.replace("mgsm_", "").replace("mmlu_", "").replace("_", " ")
    ax.set_title(f"{short}  $\\rho={rho:.3f}$", fontsize=7)
    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.tick_params(labelsize=6)


def main() -> None:
    configs = [
        # (label, svi_dir, map_dir, suffix, param, x_label, y_label, xlim)
        ("2PL difficulty",
         PROJECT_ROOT / "results" / "2PL",
         PROJECT_ROOT / "results" / "2PL_map",
         "2pl", "diff", "SVI $b_j$", "MAP $b_j$", (-1.0, 1.0)),
        ("2PL discrimination",
         PROJECT_ROOT / "results" / "2PL",
         PROJECT_ROOT / "results" / "2PL_map",
         "2pl", "disc", "SVI $a_j$", "MAP $a_j$", (-1.0, 1.0)),
        ("3PL difficulty",
         PROJECT_ROOT / "results" / "3PL",
         PROJECT_ROOT / "results" / "3PL_map",
         "3pl", "diff", "SVI $b_j$", "MAP $b_j$", (-1.0, 1.0)),
        ("3PL discrimination",
         PROJECT_ROOT / "results" / "3PL",
         PROJECT_ROOT / "results" / "3PL_map",
         "3pl", "disc", "SVI $a_j$", "MAP $a_j$", (-1.0, 1.0)),
        ("3PL guessing",
         PROJECT_ROOT / "results" / "3PL",
         PROJECT_ROOT / "results" / "3PL_map",
         "3pl", "guess", "SVI $c_j$", "MAP $c_j$", (-1.0, 1.0)),
    ]

    n_rows = len(configs)
    n_scatter = 3
    n_cols = 1 + n_scatter  # hist + 3 scatter panels

    fig = plt.figure(figsize=(12, 2.8 * n_rows))
    outer_gs = gridspec.GridSpec(n_rows, 1, hspace=0.65, figure=fig)

    for row, (label, svi_dir, map_dir, suffix, param, xlabel, ylabel, _) in enumerate(configs):
        rhos, pairs = collect_rhos(svi_dir, map_dir, suffix, param)
        arr = np.array(rhos)
        print(f"{label:25s}  n={len(rhos):3d}  mean={arr.mean():.3f}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}")

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, n_cols, subplot_spec=outer_gs[row], wspace=0.4
        )

        # histogram
        ax_h = fig.add_subplot(inner_gs[0])
        add_hist(ax_h, rhos, f"{label}", xlim=(-1.0, 1.0))

        # scatter panels
        rep_idx = pick_reps(rhos, pairs, n=n_scatter)
        for col, idx in enumerate(rep_idx):
            name, x, y = pairs[idx]
            rho_val = rhos[idx]
            ax_s = fig.add_subplot(inner_gs[col + 1])
            add_scatter(ax_s, x, y, name, rho_val, xlabel, ylabel)

    fig.suptitle(
        "SVI vs MAP: difficulty, discrimination, and guessing estimates (2PL and 3PL)",
        fontsize=10, y=1.01,
    )
    out_path = OUT_DIR / "svi_vs_map_2pl_3pl.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")


if __name__ == "__main__":
    main()
