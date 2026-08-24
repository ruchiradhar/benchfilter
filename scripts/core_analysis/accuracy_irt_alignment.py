#!/usr/bin/env python3
"""Accuracy-IRT alignment across 1PL, 2PL, and 3PL.

For each benchmark (MGSM, MMLU) and each IRT model variant (1PL, 2PL, 3PL),
builds two language×language Spearman correlation matrices:
  - IRT matrix: pairwise correlation of item difficulty rank vectors across langs
  - Accuracy matrix: pairwise correlation of model accuracy vectors across langs

A Mantel test (Pearson r of upper triangles + permutation p-value) quantifies
whether language pairs that agree on difficulty ordering also behave similarly
in raw accuracy.  Side-by-side dendrograms and heatmaps allow visual comparison.

MMLU accuracy is aggregated by averaging across all six domains per language.
For 2PL and 3PL the difficulty (diff) parameter is used, consistent with 1PL.

Outputs (in --output_dir):
  dendrograms.png      2 rows (benchmarks) × 4 cols (1PL, 2PL, 3PL, Accuracy)
  mantel_results.csv   Mantel r and p-value per (benchmark, pl) pair
  lang_accuracy.csv    Mean per-language accuracy across models

Usage:
    python scripts/analysis/accuracy_irt_alignment.py
    python scripts/analysis/accuracy_irt_alignment.py \
        --results_root results --output_dir results/core_analysis/accuracy_irt
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
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from scipy.stats import spearmanr, pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLS = ["1PL", "2PL", "3PL"]

MGSM_LANGS = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
MMLU_LANGS = ["de", "en", "es", "fr", "ja", "sw", "zh"]
MMLU_DOMAINS = ["business", "humanities", "medical", "other", "social_sciences", "stem"]

LANG_LABELS = {
    "bn": "Bengali", "de": "German", "en": "English", "es": "Spanish",
    "fr": "French", "ja": "Japanese", "ru": "Russian", "sw": "Swahili",
    "te": "Telugu", "th": "Thai", "zh": "Chinese",
}

BM_COLORS = {"MGSM": "steelblue", "MMLU": "darkorange"}


PL_DIR = {"1PL": "1PL", "2PL": "2PL_map", "3PL": "3PL_map"}  # MAP-fitted dirs

def resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# Accuracy loading
# ---------------------------------------------------------------------------

def load_gsm_accuracy(results_root: Path) -> pd.DataFrame:
    rows = {}
    for model_dir in sorted((results_root / "gsm").iterdir()):
        if not model_dir.is_dir():
            continue
        files = list(model_dir.glob("results_*.json"))
        if not files:
            continue
        d = json.loads(files[0].read_text())["results"]
        row = {}
        for lang in MGSM_LANGS:
            key, metric = f"mgsm_direct_{lang}", "exact_match,flexible-extract"
            if key in d and metric in d[key]:
                row[lang] = d[key][metric]
        if len(row) == len(MGSM_LANGS):
            rows[model_dir.name] = row
    return pd.DataFrame(rows).T


def load_mmlu_accuracy(results_root: Path) -> pd.DataFrame:
    rows = {}
    for model_dir in sorted((results_root / "mmlu").iterdir()):
        if not model_dir.is_dir():
            continue
        merged: dict = {}
        for f in model_dir.glob("results_*.json"):
            merged.update(json.loads(f.read_text())["results"])
        row = {}
        for lang in MMLU_LANGS:
            scores = [
                merged[f"global_mmlu_{lang}_{dom}"]["acc,none"]
                for dom in MMLU_DOMAINS
                if f"global_mmlu_{lang}_{dom}" in merged
                and "acc,none" in merged[f"global_mmlu_{lang}_{dom}"]
            ]
            if len(scores) == len(MMLU_DOMAINS):
                row[lang] = float(np.mean(scores))
        if len(row) == len(MMLU_LANGS):
            rows[model_dir.name] = row
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# IRT parameter loading
# ---------------------------------------------------------------------------

def load_irt_param(results_root: Path, langs: list[str],
                   benchmark: str, pl: str, param: str) -> pd.DataFrame:
    """Return (items × languages) DataFrame of IRT params for given PL and param key."""
    irt_dir = results_root / PL_DIR[pl]
    suffix = pl.lower()
    cols = {}
    for lang in langs:
        if benchmark == "mgsm":
            path = irt_dir / f"mgsm_{lang}_{suffix}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            if param not in data:
                continue
            n = len(data[param])
            ids = [data["item_ids"][str(i)] for i in range(n)]
            cols[lang] = pd.Series(data[param], index=ids)
        else:
            parts = []
            for dom in MMLU_DOMAINS:
                path = irt_dir / f"mmlu_{lang}_{dom}_{suffix}.json"
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
                if param not in data:
                    continue
                n = len(data[param])
                ids = [f"{dom}_{data['item_ids'][str(i)]}" for i in range(n)]
                parts.append(pd.Series(data[param], index=ids))
            if parts:
                cols[lang] = pd.concat(parts)
    return pd.DataFrame(cols).dropna()


def load_irt_difficulty(results_root: Path, langs: list[str],
                        benchmark: str, pl: str) -> pd.DataFrame:
    return load_irt_param(results_root, langs, benchmark, pl, "diff")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def spearman_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    n = len(cols)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(df.iloc[:, i], df.iloc[:, j])
            mat[i, j] = mat[j, i] = float(rho)
    return pd.DataFrame(mat, index=cols, columns=cols)


def mantel_test(corr_a: pd.DataFrame, corr_b: pd.DataFrame,
                n_perm: int = 999, seed: int = 42) -> tuple[float, float]:
    n = corr_a.shape[0]
    idx = np.triu_indices(n, k=1)
    x = corr_a.values[idx]
    y = corr_b.values[idx]
    obs_r, _ = pearsonr(x, y)
    rng = np.random.default_rng(seed)
    count = sum(
        pearsonr(x, corr_b.values[np.ix_(rng.permutation(n), rng.permutation(n))][idx])[0] >= obs_r
        for _ in range(n_perm)
    )
    return float(obs_r), float((count + 1) / (n_perm + 1))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_dendrogram(corr_df: pd.DataFrame, ax: plt.Axes,
                    color: str) -> None:
    langs = corr_df.columns.tolist()
    dist = ssd.squareform(np.clip(1.0 - corr_df.values, 0, None), checks=False)
    linkage = sch.linkage(dist, method="average")
    sch.dendrogram(
        linkage, labels=[LANG_LABELS[l] for l in langs], ax=ax,
        leaf_rotation=45, leaf_font_size=7,
        color_threshold=0.3 * max(linkage[:, 2]),
        link_color_func=lambda _: color,
    )
    ax.set_ylabel("1 − Spearman ρ", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def make_dendrogram_figure(bm_results: dict, acc_corrs: dict,
                           output_path: Path) -> None:
    # 2 rows (benchmarks) × 4 cols (1PL, 2PL, 3PL, Accuracy)
    col_titles = ["1PL Difficulty", "2PL Difficulty", "3PL Difficulty", "Accuracy"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for col_i, title in enumerate(col_titles):
        axes[0, col_i].set_title(title, fontsize=9, fontweight="bold")
    for row_i, bm in enumerate(["MGSM", "MMLU"]):
        color = BM_COLORS[bm]
        for col_i, pl in enumerate(PLS):
            plot_dendrogram(
                bm_results[bm][pl]["irt_corr"], axes[row_i, col_i],
                color,
            )
        plot_dendrogram(
            acc_corrs[bm], axes[row_i, 3],
            color,
        )
        axes[row_i, 0].set_ylabel(f"{bm}\n1 − Spearman ρ", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path.name}")


def make_disc_guess_figure(bm_disc_guess: dict, acc_corrs: dict,
                           output_path: Path) -> None:
    # 2 rows (benchmarks) × 4 cols (2PL disc, 3PL disc, 3PL guess, Accuracy)
    col_titles = ["2PL Discrimination", "3PL Discrimination", "3PL Guessing", "Accuracy"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for col_i, title in enumerate(col_titles):
        axes[0, col_i].set_title(title, fontsize=9, fontweight="bold")
    for row_i, bm in enumerate(["MGSM", "MMLU"]):
        color = BM_COLORS[bm]
        plot_dendrogram(bm_disc_guess[bm]["2PL_disc"],  axes[row_i, 0], color)
        plot_dendrogram(bm_disc_guess[bm]["3PL_disc"],  axes[row_i, 1], color)
        plot_dendrogram(bm_disc_guess[bm]["3PL_guess"], axes[row_i, 2], color)
        plot_dendrogram(acc_corrs[bm],                  axes[row_i, 3], color)
        axes[row_i, 0].set_ylabel(f"{bm}\n1 − Spearman ρ", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path.name}")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pl(results_root: Path, benchmark: str, langs: list[str],
           acc_corr: pd.DataFrame, pl: str, n_perm: int) -> dict:
    irt_df = load_irt_difficulty(results_root, langs, benchmark, pl)
    irt_corr = spearman_corr_matrix(irt_df).loc[langs, langs]
    mantel_r, mantel_p = mantel_test(irt_corr, acc_corr, n_perm=n_perm)
    print(f"    {pl}: items={len(irt_df)}, Mantel r={mantel_r:.4f}, p={mantel_p:.4f}")
    return {"irt_corr": irt_corr, "mantel_r": mantel_r, "mantel_p": mantel_p,
            "n_models": acc_corr.shape[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path,
                        default=Path("results/core_analysis/accuracy_irt"))
    parser.add_argument("--n_perm", type=int, default=999)
    args = parser.parse_args()
    args.results_root = resolve_path(args.results_root)
    args.output_dir = resolve_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load accuracy once per benchmark
    print("Loading accuracy matrices...")
    gsm_acc = load_gsm_accuracy(args.results_root)
    mmlu_acc = load_mmlu_accuracy(args.results_root)
    print(f"  MGSM: {gsm_acc.shape[0]} models × {gsm_acc.shape[1]} langs")
    print(f"  MMLU: {mmlu_acc.shape[0]} models × {mmlu_acc.shape[1]} langs")

    acc_corrs = {
        "MGSM": spearman_corr_matrix(gsm_acc).loc[MGSM_LANGS, MGSM_LANGS],
        "MMLU": spearman_corr_matrix(mmlu_acc).loc[MMLU_LANGS, MMLU_LANGS],
    }

    bm_configs = {"MGSM": (MGSM_LANGS, gsm_acc), "MMLU": (MMLU_LANGS, mmlu_acc)}

    bm_results: dict = {"MGSM": {}, "MMLU": {}}
    mantel_rows = []

    for bm, (langs, _) in bm_configs.items():
        print(f"\n--- {bm} ---")
        for pl in PLS:
            r = run_pl(args.results_root, bm.lower(), langs,
                       acc_corrs[bm], pl, args.n_perm)
            r["n_models"] = acc_corrs[bm].shape[0]
            bm_results[bm][pl] = r
            mantel_rows.append({
                "benchmark": bm, "pl": pl,
                "mantel_r": r["mantel_r"], "mantel_p": r["mantel_p"],
                "n_models": r["n_models"],
            })

    make_dendrogram_figure(bm_results, acc_corrs, args.output_dir / "dendrograms.png")

    # --- Discrimination and guessing dendrograms ---
    print("\n--- Disc / Guessing ---")
    bm_disc_guess: dict = {"MGSM": {}, "MMLU": {}}
    mantel_disc_guess_rows = []
    for bm, (langs, _) in bm_configs.items():
        for pl, param, key, label in [
            ("2PL", "disc",    "2PL_disc",  "2PL discrimination"),
            ("3PL", "disc",    "3PL_disc",  "3PL discrimination"),
            ("3PL", "lambdas", "3PL_guess", "3PL guessing"),
        ]:
            irt_df = load_irt_param(args.results_root, langs, bm.lower(), pl, param)
            irt_corr = spearman_corr_matrix(irt_df).loc[langs, langs]
            bm_disc_guess[bm][key] = irt_corr
            mantel_r, mantel_p = mantel_test(irt_corr, acc_corrs[bm], n_perm=args.n_perm)
            print(f"  {bm} {label}: items={len(irt_df)}, Mantel r={mantel_r:.4f}, p={mantel_p:.4f}")
            mantel_disc_guess_rows.append({
                "benchmark": bm, "parameter": label,
                "mantel_r": mantel_r, "mantel_p": mantel_p,
            })
    make_disc_guess_figure(bm_disc_guess, acc_corrs, args.output_dir / "dendrograms_disc_guess.png")
    pd.DataFrame(mantel_disc_guess_rows).to_csv(
        args.output_dir / "mantel_results_disc_guess.csv", index=False)
    print("Saved mantel_results_disc_guess.csv")

    mantel_df = pd.DataFrame(mantel_rows)
    mantel_df.to_csv(args.output_dir / "mantel_results.csv", index=False)
    print("\nSaved mantel_results.csv")

    # Mean per-language accuracy
    acc_rows = []
    for bm, (langs, acc_df) in bm_configs.items():
        for lang in langs:
            acc_rows.append({
                "benchmark": bm, "language": lang,
                "label": LANG_LABELS[lang],
                "mean_accuracy": float(acc_df[lang].mean()),
                "std_accuracy": float(acc_df[lang].std()),
            })
    pd.DataFrame(acc_rows).to_csv(args.output_dir / "lang_accuracy.csv", index=False)
    print("Saved lang_accuracy.csv")

    print("\n=== Mantel test: IRT difficulty vs accuracy language structure ===")
    print(mantel_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
