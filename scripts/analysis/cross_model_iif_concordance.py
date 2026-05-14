#!/usr/bin/env python3
"""Cross-model item concordance via Item Information Function (IIF).

For each benchmark slice (MGSM language, MMLU language×domain), computes item
information at θ=0 under each IRT model using the correct formula for that
model, selects the top-fraction of items by information, and measures set
overlap (Jaccard similarity) across the three model pairs.

IIF formulas at θ=0:
  1PL: I = P(1−P)                          P = sigmoid(−diff)
  2PL: I = disc² · P(1−P)                  P = sigmoid(disc·(−diff))
  3PL: I = disc² · (P−λ)² · (1−P) / P     P = λ + (1−λ)·sigmoid(disc·(−diff))

Items are ranked by I; the top --top_frac fraction are selected per model.
Jaccard(A,B) = |A∩B| / |A∪B| is computed for each pair (1PL/2PL, 1PL/3PL,
2PL/3PL) and averaged across slices within each benchmark.

Outputs (in --output_dir):
  iif_by_slice.csv    per-slice Jaccard for each of the 3 pairs
  iif_summary.csv     mean / std / min / max Jaccard per pair per benchmark
  iif_boxplot.png     box + strip plot of Jaccard distributions

Usage:
    python scripts/analysis/cross_model_iif_concordance.py
    python scripts/analysis/cross_model_iif_concordance.py \
        --results_root results --output_dir results/analysis/cross_model_iif \
        --top_frac 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MGSM_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
MMLU_LANGUAGES = ["de", "en", "es", "fr", "ja", "sw", "zh"]
MMLU_DOMAINS = ["business", "humanities", "medical", "other", "social_sciences", "stem"]

PAIRS = [("1PL", "2PL"), ("1PL", "3PL"), ("2PL", "3PL")]
PAIR_KEYS = [f"{a}_{b}" for a, b in PAIRS]
PAIR_LABELS = {"1PL_2PL": "1PL vs 2PL", "1PL_3PL": "1PL vs 3PL", "2PL_3PL": "2PL vs 3PL"}
BENCHMARK_COLORS = {"MGSM": "steelblue", "MMLU": "darkorange"}

EPS = 1e-8  # guard against log(0) / div-by-zero in 3PL formula


def resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


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


def load_params(results_root: Path, pl: str, stem: str) -> dict | None:
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


def top_items(information: np.ndarray, item_ids: list[str], top_frac: float) -> set[str]:
    k = max(1, int(round(len(item_ids) * top_frac)))
    indices = np.argsort(information)[::-1][:k]
    return {item_ids[i] for i in indices}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def slice_concordance(results_root: Path, stem: str, top_frac: float) -> dict[str, float] | None:
    p1 = load_params(results_root, "1PL", stem)
    p2 = load_params(results_root, "2PL", stem)
    p3 = load_params(results_root, "3PL", stem)
    if p1 is None or p2 is None or p3 is None:
        return None

    iif1 = iif_1pl(p1["diff"])
    iif2 = iif_2pl(p2["diff"], p2["disc"])
    iif3 = iif_3pl(p3["diff"], p3["disc"], p3["lambdas"])

    set1 = top_items(iif1, p1["item_ids"], top_frac)
    set2 = top_items(iif2, p2["item_ids"], top_frac)
    set3 = top_items(iif3, p3["item_ids"], top_frac)

    return {
        "1PL_2PL": jaccard(set1, set2),
        "1PL_3PL": jaccard(set1, set3),
        "2PL_3PL": jaccard(set2, set3),
        "n_items": len(p1["diff"]),
        "top_k": len(set1),
    }


def collect_rows(results_root: Path, top_frac: float) -> list[dict]:
    rows: list[dict] = []

    for lang in MGSM_LANGUAGES:
        stem = f"mgsm_{lang}"
        r = slice_concordance(results_root, stem, top_frac)
        if r is not None:
            rows.append({"benchmark": "MGSM", "slice": stem, **r})
        else:
            print(f"  skipped (missing files): {stem}")

    for lang in MMLU_LANGUAGES:
        for domain in MMLU_DOMAINS:
            stem = f"mmlu_{lang}_{domain}"
            r = slice_concordance(results_root, stem, top_frac)
            if r is not None:
                rows.append({"benchmark": "MMLU", "slice": stem, **r})
            else:
                print(f"  skipped (missing files): {stem}")

    return rows


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for benchmark in ["MGSM", "MMLU"]:
        sub = df[df["benchmark"] == benchmark]
        for key in PAIR_KEYS:
            vals = sub[key].dropna()
            records.append({
                "benchmark": benchmark,
                "pair": PAIR_LABELS[key],
                "n_slices": len(vals),
                "mean_jaccard": vals.mean(),
                "std_jaccard": vals.std(),
                "min_jaccard": vals.min(),
                "max_jaccard": vals.max(),
            })
    return pd.DataFrame(records)


def plot_concordance(df: pd.DataFrame, top_frac: float, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    fig.suptitle(
        f"Cross-model item concordance via Item Information Function\n"
        f"(Jaccard similarity of top-{int(top_frac*100)}% items by IIF at θ=0)",
        fontsize=11,
    )

    for ax, benchmark in zip(axes, ["MGSM", "MMLU"]):
        sub = df[df["benchmark"] == benchmark]
        color = BENCHMARK_COLORS[benchmark]
        positions = np.arange(len(PAIR_KEYS))
        data_per_pair = [sub[key].dropna().values for key in PAIR_KEYS]

        ax.boxplot(
            data_per_pair,
            positions=positions,
            widths=0.4,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            boxprops=dict(facecolor=color, alpha=0.4),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            flierprops=dict(marker="o", color=color, markersize=4, alpha=0.5),
        )

        rng = np.random.default_rng(42)
        for i, vals in enumerate(data_per_pair):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(positions[i] + jitter, vals, color=color, alpha=0.6, s=18, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels([PAIR_LABELS[k] for k in PAIR_KEYS], fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Jaccard similarity" if benchmark == "MGSM" else "", fontsize=10)

        n_slices = len(sub)
        avg_k = int(sub["top_k"].mean())
        ax.set_title(f"{benchmark}  (n={n_slices} slices, avg top-k={avg_k})", fontsize=10)

        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(0.5, color="orange", linestyle="--", linewidth=0.8, label="J = 0.5")
        ax.axhline(0.25, color="red", linestyle="--", linewidth=0.8, label="J = 0.25")
        ax.legend(fontsize=8, loc="lower right")

        for i, vals in enumerate(data_per_pair):
            med = np.median(vals)
            ax.text(
                positions[i], med + 0.02, f"{med:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/analysis/cross_model_iif"))
    parser.add_argument("--top_frac", type=float, default=0.5,
                        help="Fraction of items to select by IIF (default: 0.5)")
    args = parser.parse_args()
    args.results_root = resolve_path(args.results_root)
    args.output_dir = resolve_path(args.output_dir)
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"top_frac={args.top_frac} — selecting top {int(args.top_frac*100)}% of items by IIF at θ=0")
    print("Loading slices...")
    rows = collect_rows(args.results_root, args.top_frac)
    if not rows:
        raise RuntimeError("No slices loaded — check --results_root path.")

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "iif_by_slice.csv", index=False)
    print(f"Saved iif_by_slice.csv  ({len(df)} slices)")

    summary = build_summary(df)
    summary.to_csv(args.output_dir / "iif_summary.csv", index=False)
    print("Saved iif_summary.csv")

    plot_concordance(df, args.top_frac, args.output_dir / "iif_boxplot.png")

    print("\n=== Cross-model IIF concordance (Jaccard similarity) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
