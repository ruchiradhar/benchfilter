#!/usr/bin/env python3
"""Within-language split-half reliability of 2PL/3PL discrimination and guessing.

Tests whether low cross-lingual concordance of
discrimination/guessing reflects language-specific signal or simply estimation
noise from ~100 subjects. For each language slice, randomly splits the model
pool in half, re-fits 2PL/3PL via MAP independently on each half (same item
set), and computes Spearman rho between the two halves' item parameter
estimates. 
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "irt"))
from irt_map import TwoPL, ThreePL, load_jsonlines, train_one  # noqa: E402

MGSM_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]

# Reduced relative to the main-paper MAP schedule (4000/6000 epochs, 3 restarts)
# for tractable rebuttal turnaround; single restart per half, multiple random
# splits per slice instead to characterise variance.
EPOCHS = {"2pl": 1500, "3pl": 2000}
LR = 0.05
N_SPLITS = 3


def fit_on_subject_subset(
    R: torch.Tensor, subject_idx: list[int], model_cls, epochs: int, lr: float, seed: int
):
    n_i = R.shape[1]
    sub_R = R[subject_idx]
    n_s = sub_R.shape[0]
    s_idx = torch.arange(n_s).repeat_interleave(n_i)
    i_idx = torch.arange(n_i).repeat(n_s)
    resp = sub_R.flatten()
    model, _ = train_one(
        model_cls, s_idx, i_idx, resp, n_s, n_i,
        epochs=epochs, lr=lr, device=torch.device("cpu"), seed=seed,
    )
    return model


def split_half_for_slice(path: Path, n_splits: int, rng: np.random.Generator) -> list[dict]:
    subject_ids, item_ids, R = load_jsonlines(path)
    n_s = len(subject_ids)
    rows = []

    for split_i in range(n_splits):
        perm = rng.permutation(n_s)
        half = n_s // 2
        idx_a, idx_b = perm[:half].tolist(), perm[half:].tolist()

        for model_name, model_cls in [("2pl", TwoPL), ("3pl", ThreePL)]:
            epochs = EPOCHS[model_name]
            m_a = fit_on_subject_subset(R, idx_a, model_cls, epochs, LR, seed=split_i * 100 + 1)
            m_b = fit_on_subject_subset(R, idx_b, model_cls, epochs, LR, seed=split_i * 100 + 2)

            with torch.no_grad():
                disc_a, disc_b = m_a.a.cpu().numpy(), m_b.a.cpu().numpy()
            rho_disc, _ = spearmanr(disc_a, disc_b)
            row = {
                "slice": path.stem,
                "split": split_i,
                "model": model_name,
                "n_half_a": len(idx_a),
                "n_half_b": len(idx_b),
                "rho_disc": float(rho_disc),
            }
            if model_name == "3pl":
                with torch.no_grad():
                    c_a, c_b = m_a.c.cpu().numpy(), m_b.c.cpu().numpy()
                rho_c, _ = spearmanr(c_a, c_b)
                row["rho_guess"] = float(rho_c)
            rows.append(row)
            print(f"  {path.stem} split={split_i} {model_name}: rho_disc={rho_disc:.3f}"
                  + (f" rho_guess={row.get('rho_guess'):.3f}" if model_name == "3pl" else ""))

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/irt"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/core_analysis/split_half"))
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    parser.add_argument("--slices", nargs="*", default=None,
                         help="Slice stems to run (default: all 11 MGSM languages).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.data_dir = args.data_dir if args.data_dir.is_absolute() else (PROJECT_ROOT / args.data_dir).resolve()
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir).resolve()
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    slice_stems = args.slices or [f"mgsm_{lang}" for lang in MGSM_LANGUAGES]
    all_rows: list[dict] = []
    for stem in slice_stems:
        path = args.data_dir / f"{stem}.jsonlines"
        if not path.exists():
            print(f"  skip (missing): {path}")
            continue
        print(f"Fitting split-halves for {stem} ...")
        all_rows.extend(split_half_for_slice(path, args.n_splits, rng))

    df = pd.DataFrame(all_rows)
    out_path = args.output_dir / "split_half_reliability.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    print("\n=== Split-half reliability summary (within-language, same N as cross-lingual fits) ===")
    for model_name in ["2pl", "3pl"]:
        sub = df[df["model"] == model_name]
        print(f"{model_name.upper()} discrimination: mean rho = {sub['rho_disc'].mean():.3f} "
              f"[{sub['rho_disc'].min():.3f}, {sub['rho_disc'].max():.3f}]  (n={len(sub)} split-fits)")
        if model_name == "3pl":
            print(f"3PL guessing:        mean rho = {sub['rho_guess'].mean():.3f} "
                  f"[{sub['rho_guess'].min():.3f}, {sub['rho_guess'].max():.3f}]")


if __name__ == "__main__":
    main()
