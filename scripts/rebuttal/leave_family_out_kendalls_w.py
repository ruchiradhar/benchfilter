#!/usr/bin/env python3
"""Leave-Indo-European-out Kendall's W .

"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MGSM_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
MGSM_INDO_EUROPEAN = {"de", "en", "es", "fr", "ru"}  # the tight cluster reported in the paper
MGSM_RESTRICTED = [l for l in MGSM_LANGUAGES if l not in MGSM_INDO_EUROPEAN]  # bn, ja, sw, te, th, zh

MMLU_LANGUAGES = ["de", "en", "es", "fr", "ja", "sw", "zh"]
MMLU_INDO_EUROPEAN = {"de", "en", "es", "fr"}  # matches Limitations paragraph
MMLU_RESTRICTED = [l for l in MMLU_LANGUAGES if l not in MMLU_INDO_EUROPEAN]  # ja, sw, zh
MMLU_DOMAINS = ["business", "humanities", "medical", "other", "social_sciences", "stem"]

PL_DIR = {"1pl": "1PL", "2pl": "2PL_map", "3pl": "3PL_map"}


def kendalls_w(rank_matrix: np.ndarray) -> float:
    n, m = rank_matrix.shape
    col_sums = rank_matrix.sum(axis=1)
    grand_mean = col_sums.mean()
    s = np.sum((col_sums - grand_mean) ** 2)
    return float(12 * s / (m ** 2 * (n ** 3 - n)))


def load_diff_matrix(results_root: Path, pl: str, stem_fn, langs: list[str]) -> pd.DataFrame:
    cols = {}
    for lang in langs:
        path = results_root / PL_DIR[pl] / f"{stem_fn(lang)}_{pl}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        item_ids = [data["item_ids"][str(i)] for i in range(len(data["diff"]))]
        cols[lang] = pd.Series(data["diff"], index=item_ids)
    return pd.DataFrame(cols)


def w_for_langs(results_root: Path, pl: str, stem_fn, langs: list[str]) -> tuple[float, int, int]:
    df = load_diff_matrix(results_root, pl, stem_fn, langs)
    df = df.dropna()
    if df.shape[1] < 2 or df.shape[0] < 3:
        return float("nan"), df.shape[0], df.shape[1]
    rk = df.rank(axis=0, method="average")
    return kendalls_w(rk.values), df.shape[0], df.shape[1]


def main() -> None:
    results_root = PROJECT_ROOT / "results"
    output_dir = PROJECT_ROOT / "results" / "rebuttal"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # --- MGSM ---
    for pl in ["1pl", "2pl", "3pl"]:
        w_full, n_items, n_langs = w_for_langs(results_root, pl, lambda l: f"mgsm_{l}", MGSM_LANGUAGES)
        w_restricted, n_items_r, n_langs_r = w_for_langs(results_root, pl, lambda l: f"mgsm_{l}", MGSM_RESTRICTED)
        rows.append({
            "benchmark": "MGSM", "domain": "-", "model": pl.upper(),
            "W_full": w_full, "n_langs_full": n_langs,
            "W_restricted": w_restricted, "n_langs_restricted": n_langs_r,
            "restricted_langs": ",".join(MGSM_RESTRICTED),
        })

    # --- MMLU (per domain + pooled across domains) ---
    for pl in ["1pl", "2pl", "3pl"]:
        full_vals, restricted_vals = [], []
        for domain in MMLU_DOMAINS:
            w_full, n_items, n_langs = w_for_langs(
                results_root, pl, lambda l, d=domain: f"mmlu_{l}_{d}", MMLU_LANGUAGES
            )
            w_restricted, n_items_r, n_langs_r = w_for_langs(
                results_root, pl, lambda l, d=domain: f"mmlu_{l}_{d}", MMLU_RESTRICTED
            )
            rows.append({
                "benchmark": "MMLU", "domain": domain, "model": pl.upper(),
                "W_full": w_full, "n_langs_full": n_langs,
                "W_restricted": w_restricted, "n_langs_restricted": n_langs_r,
                "restricted_langs": ",".join(MMLU_RESTRICTED),
            })
            if not np.isnan(w_full):
                full_vals.append(w_full)
            if not np.isnan(w_restricted):
                restricted_vals.append(w_restricted)
        rows.append({
            "benchmark": "MMLU", "domain": "MEAN_ACROSS_DOMAINS", "model": pl.upper(),
            "W_full": float(np.mean(full_vals)) if full_vals else float("nan"),
            "n_langs_full": len(MMLU_LANGUAGES),
            "W_restricted": float(np.mean(restricted_vals)) if restricted_vals else float("nan"),
            "n_langs_restricted": len(MMLU_RESTRICTED),
            "restricted_langs": ",".join(MMLU_RESTRICTED),
        })

    df = pd.DataFrame(rows)
    out_path = output_dir / "leave_family_out_kendalls_w.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
