#!/usr/bin/env python3
"""
Discrimination collapse analysis.

For each benchmark slice, load 2PL and 3PL fitted parameters and inspect
the distribution of discrimination (disc) values. Under a well-identified
model, disc should be positive and span a meaningful range (roughly 0.5–2.0).
Values near zero or negative indicate prior-induced collapse or optimization
failure.

Outputs:
  results/add_analysis/discrimination_collapse.png  — main figure
  results/add_analysis/discrimination_collapse.csv  — per-slice summary stats
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RES_2PL = PROJECT_ROOT / "results" / "2PL"
RES_3PL = PROJECT_ROOT / "results" / "3PL"
OUT_DIR = PROJECT_ROOT / "results" / "add_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load parameters
# ---------------------------------------------------------------------------

def load_disc(results_dir: Path, suffix: str) -> dict[str, np.ndarray]:
    """Return {slice_name: disc_array} for every fitted file in results_dir."""
    data = {}
    for path in sorted(results_dir.glob(f"*_{suffix}.json")):
        params = json.loads(path.read_text())
        disc = np.array(params["disc"], dtype=float)
        slice_name = path.stem.replace(f"_{suffix}", "")
        data[slice_name] = disc
    return data


disc_2pl = load_disc(RES_2PL, "2pl")
disc_3pl = load_disc(RES_3PL, "3pl")

# Collect all values pooled
all_2pl = np.concatenate(list(disc_2pl.values()))
all_3pl_values = [v for v in disc_3pl.values() if len(v) > 0]
all_3pl = np.concatenate(all_3pl_values) if all_3pl_values else np.array([])

# Separate MGSM and MMLU slices
mgsm_2pl = {k: v for k, v in disc_2pl.items() if k.startswith("mgsm")}
mmlu_2pl  = {k: v for k, v in disc_2pl.items() if k.startswith("mmlu")}
mgsm_3pl  = {k: v for k, v in disc_3pl.items() if k.startswith("mgsm")}
mmlu_3pl  = {k: v for k, v in disc_3pl.items() if k.startswith("mmlu")}

# ---------------------------------------------------------------------------
# Per-slice summary table
# ---------------------------------------------------------------------------

rows = []
for model_label, disc_dict in [("2PL", disc_2pl), ("3PL", disc_3pl)]:
    for slice_name, disc in disc_dict.items():
        benchmark = "MGSM" if slice_name.startswith("mgsm") else "MMLU"
        pct_negative = (disc < 0).mean() * 100
        pct_below_half = (disc < 0.5).mean() * 100
        rows.append({
            "model": model_label,
            "benchmark": benchmark,
            "slice": slice_name,
            "mean": disc.mean(),
            "std": disc.std(),
            "min": disc.min(),
            "max": disc.max(),
            "pct_negative": pct_negative,
            "pct_below_0.5": pct_below_half,
        })

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "discrimination_collapse.csv", index=False, float_format="%.4f")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Colour palette
C2 = "#2196F3"   # blue for 2PL
C3 = "#F44336"   # red for 3PL
CREF = "#4CAF50" # green for reference line (disc=1, the 1PL equivalent)

# ── Panel A: Pooled 2PL disc distribution ──────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.hist(all_2pl, bins=40, color=C2, alpha=0.8, edgecolor="white", linewidth=0.4)
ax_a.axvline(1.0, color=CREF, lw=1.8, ls="--", label="1PL equiv. (a=1)")
ax_a.axvline(0.0, color="black", lw=1.0, ls=":", alpha=0.6, label="a=0 (no info)")
ax_a.set_title("2PL discrimination (pooled)", fontsize=10, fontweight="bold")
ax_a.set_xlabel("Discrimination $a_j$")
ax_a.set_ylabel("Count")
ax_a.legend(fontsize=7)
stats_str = (f"mean={all_2pl.mean():.3f}\n"
             f"std={all_2pl.std():.3f}\n"
             f"max={all_2pl.max():.3f}\n"
             f"{(all_2pl < 0).mean()*100:.1f}% negative")
ax_a.text(0.97, 0.97, stats_str, transform=ax_a.transAxes,
          va="top", ha="right", fontsize=7.5,
          bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# ── Panel B: Pooled 3PL disc distribution ──────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
ax_b.hist(all_3pl, bins=40, color=C3, alpha=0.8, edgecolor="white", linewidth=0.4)
ax_b.axvline(1.0, color=CREF, lw=1.8, ls="--", label="1PL equiv. (a=1)")
ax_b.axvline(0.0, color="black", lw=1.0, ls=":", alpha=0.6, label="a=0 boundary")
ax_b.set_title("3PL discrimination (pooled)", fontsize=10, fontweight="bold")
ax_b.set_xlabel("Discrimination $a_j$")
ax_b.set_ylabel("Count")
ax_b.legend(fontsize=7)
stats_str = (f"mean={all_3pl.mean():.3f}\n"
             f"std={all_3pl.std():.3f}\n"
             f"max={all_3pl.max():.3f}\n"
             f"{(all_3pl < 0).mean()*100:.1f}% negative")
ax_b.text(0.97, 0.97, stats_str, transform=ax_b.transAxes,
          va="top", ha="right", fontsize=7.5,
          bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# ── Panel C: Per-slice mean discrimination (MGSM) ──────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
slices_mgsm = sorted(set(mgsm_2pl) | set(mgsm_3pl))
means_2pl_mgsm = [mgsm_2pl[s].mean() if s in mgsm_2pl else np.nan for s in slices_mgsm]
means_3pl_mgsm = [mgsm_3pl[s].mean() if s in mgsm_3pl else np.nan for s in slices_mgsm]
x = np.arange(len(slices_mgsm))
ax_c.bar(x - 0.2, means_2pl_mgsm, 0.4, color=C2, alpha=0.85, label="2PL")
ax_c.bar(x + 0.2, means_3pl_mgsm, 0.4, color=C3, alpha=0.85, label="3PL")
ax_c.axhline(1.0, color=CREF, lw=1.5, ls="--", label="1PL equiv.")
ax_c.axhline(0.0, color="black", lw=0.8, ls=":", alpha=0.5)
ax_c.set_xticks(x)
labels_mgsm = [s.replace("mgsm_", "") for s in slices_mgsm]
ax_c.set_xticklabels(labels_mgsm, rotation=45, ha="right", fontsize=7)
ax_c.set_title("MGSM: per-slice mean disc.", fontsize=10, fontweight="bold")
ax_c.set_ylabel("Mean $a_j$")
ax_c.legend(fontsize=7)

# ── Panel D: Per-slice % negative discriminations (MGSM) ───────────────────
ax_d = fig.add_subplot(gs[1, 0])
pct_neg_2pl_mgsm = [(mgsm_2pl[s] < 0).mean()*100 if s in mgsm_2pl else np.nan for s in slices_mgsm]
pct_neg_3pl_mgsm = [(mgsm_3pl[s] < 0).mean()*100 if s in mgsm_3pl else np.nan for s in slices_mgsm]
ax_d.bar(x - 0.2, pct_neg_2pl_mgsm, 0.4, color=C2, alpha=0.85, label="2PL")
ax_d.bar(x + 0.2, pct_neg_3pl_mgsm, 0.4, color=C3, alpha=0.85, label="3PL")
ax_d.set_xticks(x)
ax_d.set_xticklabels(labels_mgsm, rotation=45, ha="right", fontsize=7)
ax_d.set_title("MGSM: % items with $a_j < 0$", fontsize=10, fontweight="bold")
ax_d.set_ylabel("% negative disc.")
ax_d.legend(fontsize=7)

# ── Panel E: Per-slice % negative discriminations (MMLU, averaged over domains) ─
ax_e = fig.add_subplot(gs[1, 1])
mmlu_langs = sorted(set(k.split("_")[1] for k in mmlu_2pl))
pct_neg_2pl_mmlu, pct_neg_3pl_mmlu = [], []
means_2pl_mmlu, means_3pl_mmlu = [], []

for lang in mmlu_langs:
    slices = [k for k in mmlu_2pl if f"mmlu_{lang}_" in k]
    if slices:
        pct_neg_2pl_mmlu.append(np.mean([(mmlu_2pl[s] < 0).mean()*100 for s in slices if s in mmlu_2pl]))
        means_2pl_mmlu.append(np.mean([mmlu_2pl[s].mean() for s in slices if s in mmlu_2pl]))
    else:
        pct_neg_2pl_mmlu.append(np.nan)
        means_2pl_mmlu.append(np.nan)
    slices3 = [k for k in mmlu_3pl if f"mmlu_{lang}_" in k]
    if slices3:
        pct_neg_3pl_mmlu.append(np.mean([(mmlu_3pl[s] < 0).mean()*100 for s in slices3 if s in mmlu_3pl]))
        means_3pl_mmlu.append(np.mean([mmlu_3pl[s].mean() for s in slices3 if s in mmlu_3pl]))
    else:
        pct_neg_3pl_mmlu.append(np.nan)
        means_3pl_mmlu.append(np.nan)

x2 = np.arange(len(mmlu_langs))
ax_e.bar(x2 - 0.2, pct_neg_2pl_mmlu, 0.4, color=C2, alpha=0.85, label="2PL")
ax_e.bar(x2 + 0.2, pct_neg_3pl_mmlu, 0.4, color=C3, alpha=0.85, label="3PL")
ax_e.set_xticks(x2)
ax_e.set_xticklabels(mmlu_langs, rotation=45, ha="right", fontsize=7)
ax_e.set_title("MMLU: % items with $a_j < 0$ (avg. domains)", fontsize=10, fontweight="bold")
ax_e.set_ylabel("% negative disc.")
ax_e.legend(fontsize=7)

# ── Panel F: ICC illustration — well-identified vs collapsed ───────────────
ax_f = fig.add_subplot(gs[1, 2])
theta = np.linspace(-4, 4, 300)

# Typical 1PL equivalent (a=1, b=0)
p_1pl = 1 / (1 + np.exp(-(theta - 0)))
# 2PL with typical estimated disc (a=0.24, b=0)
p_2pl_low = 1 / (1 + np.exp(-0.24 * (theta - 0)))
# 2PL well-identified (a=1.5, b=0)
p_2pl_good = 1 / (1 + np.exp(-1.5 * (theta - 0)))
# 3PL with negative disc (a=-3, b=0) — inverted ICC
p_3pl_neg = 1 / (1 + np.exp(-(-3.0) * (theta - 0)))

ax_f.plot(theta, p_1pl,      color=CREF, lw=2,   ls="--", label="1PL (a=1, reference)")
ax_f.plot(theta, p_2pl_good, color=C2,   lw=2,   ls="-",  label="2PL well-identified (a=1.5)")
ax_f.plot(theta, p_2pl_low,  color=C2,   lw=2,   ls=":",  label="2PL estimated (a=0.24, flat)")
ax_f.plot(theta, p_3pl_neg,  color=C3,   lw=2,   ls="-",  label="3PL estimated (a=−3, inverted)")
ax_f.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5)
ax_f.set_xlabel("Latent ability $\\theta$")
ax_f.set_ylabel("P(correct)")
ax_f.set_title("ICC examples: estimated vs expected", fontsize=10, fontweight="bold")
ax_f.legend(fontsize=6.5, loc="center left")
ax_f.set_ylim(-0.05, 1.05)

fig.suptitle(
    "Discrimination collapse in 2PL and 3PL\n"
    "2PL: disc near zero (flat ICCs). 3PL: disc negative (inverted ICCs).\n"
    "1PL implicitly fixes disc=1; both 2PL/3PL priors are Normal(0,1) allowing collapse.",
    fontsize=10, y=1.00
)

fig.savefig(OUT_DIR / "discrimination_collapse.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print("=" * 60)
print("DISCRIMINATION COLLAPSE SUMMARY")
print("=" * 60)
print(f"\n2PL across all {len(disc_2pl)} slices:")
print(f"  mean disc = {all_2pl.mean():.4f}  (expect ~1.0 for well-identified)")
print(f"  std        = {all_2pl.std():.4f}")
print(f"  range      = [{all_2pl.min():.4f}, {all_2pl.max():.4f}]")
print(f"  % negative = {(all_2pl < 0).mean()*100:.1f}%")
print(f"  % < 0.5    = {(all_2pl < 0.5).mean()*100:.1f}%")

print(f"\n3PL across all {len(disc_3pl)} slices:")
print(f"  mean disc = {all_3pl.mean():.4f}  (negative = inverted ICCs)")
print(f"  std        = {all_3pl.std():.4f}")
print(f"  range      = [{all_3pl.min():.4f}, {all_3pl.max():.4f}]")
print(f"  % negative = {(all_3pl < 0).mean()*100:.1f}%")

print(f"\nBy benchmark (2PL % items with a < 0.5):")
for bench in ["MGSM", "MMLU"]:
    sub = df[(df["model"] == "2PL") & (df["benchmark"] == bench)]
    print(f"  {bench}: {sub['pct_below_0.5'].mean():.1f}% of items per slice (mean)")

print(f"\nBy benchmark (3PL % items with a < 0):")
for bench in ["MGSM", "MMLU"]:
    sub = df[(df["model"] == "3PL") & (df["benchmark"] == bench)]
    print(f"  {bench}: {sub['pct_negative'].mean():.1f}% of items per slice (mean)")

print(f"\nFigure saved to: {OUT_DIR / 'discrimination_collapse.png'}")
print(f"CSV saved to:    {OUT_DIR / 'discrimination_collapse.csv'}")
