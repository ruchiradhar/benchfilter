#!/usr/bin/env python3
"""
Compare discrimination distributions: py-irt SVI vs MAP fitting.

Produces a side-by-side figure and prints a summary table.

Usage:
    python scripts/analysis/discrimination_check_map.py
    python scripts/analysis/discrimination_check_map.py --model 3pl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "add_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_disc(results_dir: Path, suffix: str) -> dict[str, np.ndarray]:
    data = {}
    for path in sorted(results_dir.glob(f"*_{suffix}.json")):
        params = json.loads(path.read_text())
        disc = np.array(params["disc"], dtype=float)
        slice_name = path.stem.replace(f"_{suffix}", "")
        data[slice_name] = disc
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["2pl", "3pl"], default="2pl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    m = args.model

    svi_dir = PROJECT_ROOT / "results" / m.upper()
    map_dir = PROJECT_ROOT / "results" / f"{m.upper()}_map"

    disc_svi = load_disc(svi_dir, m)
    disc_map = load_disc(map_dir, m)

    if not disc_svi:
        print(f"No SVI results found in {svi_dir}")
        return
    if not disc_map:
        print(f"No MAP results found in {map_dir} — run irt_map.py first")
        return

    common = sorted(set(disc_svi) & set(disc_map))
    all_svi = np.concatenate([disc_svi[s] for s in common])
    all_map = np.concatenate([disc_map[s] for s in common])

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    for ax, vals, label, color in [
        (axes[0], all_svi, f"py-irt SVI ({m.upper()})", "#F44336"),
        (axes[1], all_map, f"MAP ({m.upper()})", "#2196F3"),
    ]:
        ax.hist(vals, bins=50, color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(1.0, color="#4CAF50", lw=2, ls="--", label="a = 1 (1PL equiv.)")
        ax.axvline(0.0, color="black", lw=1, ls=":", alpha=0.6, label="a = 0")
        pct_neg = (vals < 0).mean() * 100
        pct_lt_half = (vals < 0.5).mean() * 100
        stats = (
            f"mean = {vals.mean():.3f}\n"
            f"std  = {vals.std():.3f}\n"
            f"range = [{vals.min():.2f}, {vals.max():.2f}]\n"
            f"% < 0   = {pct_neg:.1f}%\n"
            f"% < 0.5 = {pct_lt_half:.1f}%"
        )
        ax.text(0.97, 0.97, stats, transform=ax.transAxes,
                va="top", ha="right", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Discrimination $a_j$")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"{m.upper()} discrimination: SVI (collapsed) vs MAP (properly identified)\n"
        f"n = {len(common)} slices, {len(all_svi):,} items total",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = OUT_DIR / f"discrimination_svi_vs_map_{m}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")

    # ── Summary table ───────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {m.upper()} discrimination: SVI vs MAP ({len(common)} slices)")
    print(f"{'='*55}")
    print(f"{'Metric':<25} {'SVI':>12} {'MAP':>12}")
    print(f"{'-'*55}")
    for label, svi_val, map_val in [
        ("Mean", all_svi.mean(), all_map.mean()),
        ("Std", all_svi.std(), all_map.std()),
        ("Min", all_svi.min(), all_map.min()),
        ("Max", all_svi.max(), all_map.max()),
        ("% negative", (all_svi < 0).mean() * 100, (all_map < 0).mean() * 100),
        ("% < 0.5", (all_svi < 0.5).mean() * 100, (all_map < 0.5).mean() * 100),
    ]:
        print(f"  {label:<23} {svi_val:>12.3f} {map_val:>12.3f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
