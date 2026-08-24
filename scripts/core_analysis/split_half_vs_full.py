#!/usr/bin/env python3
"""Compare split-half MAP re-fits (N=51) against the full-sample (N=102) MAP
estimates reported in the paper.

Reuses the exact same random splits as split_half_reliability.py (same seed,
same language order, same n_splits) so per-half discrimination/guessing
estimates can be directly correlated against the stored full-sample
results/{2PL,3PL}_map/{stem}_{model}.json parameters, and their spread (std)
compared to the full sample's.
"""

from __future__ import annotations

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
EPOCHS = {"2pl": 1500, "3pl": 2000}
LR = 0.05
N_SPLITS = 3
SEED = 42

DATA_DIR = PROJECT_ROOT / "data" / "irt"
FULL_DIRS = {"2pl": PROJECT_ROOT / "results" / "2PL_map", "3pl": PROJECT_ROOT / "results" / "3PL_map"}
OUT_DIR = PROJECT_ROOT / "results" / "core_analysis" / "split_half"


def fit_on_subject_subset(R, subject_idx, model_cls, epochs, lr, seed):
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


def load_full(stem: str, model_name: str) -> dict | None:
    path = FULL_DIRS[model_name] / f"{stem}_{model_name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []

    for lang in MGSM_LANGUAGES:
        stem = f"mgsm_{lang}"
        path = DATA_DIR / f"{stem}.jsonlines"
        if not path.exists():
            print(f"  skip (missing): {path}")
            continue
        subject_ids, item_ids, R = load_jsonlines(path)
        n_s = len(subject_ids)

        full_2pl = load_full(stem, "2pl")
        full_3pl = load_full(stem, "3pl")
        full_disc_2pl = np.array(full_2pl["disc"]) if full_2pl else None
        full_disc_3pl = np.array(full_3pl["disc"]) if full_3pl else None
        full_guess_3pl = np.array(full_3pl["lambdas"]) if full_3pl else None

        print(f"Fitting split-halves for {stem} ...")
        for split_i in range(N_SPLITS):
            perm = rng.permutation(n_s)
            half = n_s // 2
            idx_a, idx_b = perm[:half].tolist(), perm[half:].tolist()

            for model_name, model_cls in [("2pl", TwoPL), ("3pl", ThreePL)]:
                epochs = EPOCHS[model_name]
                m_a = fit_on_subject_subset(R, idx_a, model_cls, epochs, LR, seed=split_i * 100 + 1)
                m_b = fit_on_subject_subset(R, idx_b, model_cls, epochs, LR, seed=split_i * 100 + 2)

                with torch.no_grad():
                    disc_a, disc_b = m_a.a.cpu().numpy(), m_b.a.cpu().numpy()

                full_disc = full_disc_2pl if model_name == "2pl" else full_disc_3pl
                rho_a_disc, _ = spearmanr(disc_a, full_disc) if full_disc is not None else (np.nan, None)
                rho_b_disc, _ = spearmanr(disc_b, full_disc) if full_disc is not None else (np.nan, None)

                row = {
                    "slice": stem, "split": split_i, "model": model_name,
                    "std_disc_a": float(disc_a.std()), "std_disc_b": float(disc_b.std()),
                    "std_disc_full": float(full_disc.std()) if full_disc is not None else np.nan,
                    "rho_disc_a_vs_full": float(rho_a_disc),
                    "rho_disc_b_vs_full": float(rho_b_disc),
                }

                if model_name == "3pl":
                    with torch.no_grad():
                        guess_a, guess_b = m_a.c.cpu().numpy(), m_b.c.cpu().numpy()
                    rho_a_guess, _ = spearmanr(guess_a, full_guess_3pl) if full_guess_3pl is not None else (np.nan, None)
                    rho_b_guess, _ = spearmanr(guess_b, full_guess_3pl) if full_guess_3pl is not None else (np.nan, None)
                    row.update({
                        "std_guess_a": float(guess_a.std()), "std_guess_b": float(guess_b.std()),
                        "std_guess_full": float(full_guess_3pl.std()) if full_guess_3pl is not None else np.nan,
                        "rho_guess_a_vs_full": float(rho_a_guess),
                        "rho_guess_b_vs_full": float(rho_b_guess),
                    })

                rows.append(row)
                print(f"  {stem} split={split_i} {model_name}: "
                      f"rho_disc(a,full)={row['rho_disc_a_vs_full']:.3f} "
                      f"rho_disc(b,full)={row['rho_disc_b_vs_full']:.3f}"
                      + (f" rho_guess(a,full)={row.get('rho_guess_a_vs_full'):.3f}"
                         f" rho_guess(b,full)={row.get('rho_guess_b_vs_full'):.3f}"
                         if model_name == "3pl" else ""))

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "split_half_vs_full.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    print("\n=== Split-half vs full-sample (N=102) summary ===")
    for model_name in ["2pl", "3pl"]:
        sub = df[df["model"] == model_name]
        rho_all = pd.concat([sub["rho_disc_a_vs_full"], sub["rho_disc_b_vs_full"]])
        std_half = pd.concat([sub["std_disc_a"], sub["std_disc_b"]])
        print(f"{model_name.upper()} discrimination vs full: rho range [{rho_all.min():.3f}, {rho_all.max():.3f}], "
              f"mean={rho_all.mean():.3f}")
        print(f"  spread: half std mean={std_half.mean():.3f} [{std_half.min():.3f}, {std_half.max():.3f}]  "
              f"vs full std mean={sub['std_disc_full'].mean():.3f} "
              f"[{sub['std_disc_full'].min():.3f}, {sub['std_disc_full'].max():.3f}]")
        if model_name == "3pl":
            rho_g = pd.concat([sub["rho_guess_a_vs_full"], sub["rho_guess_b_vs_full"]])
            std_g_half = pd.concat([sub["std_guess_a"], sub["std_guess_b"]])
            print(f"3PL guessing vs full: rho range [{rho_g.min():.3f}, {rho_g.max():.3f}], mean={rho_g.mean():.3f}")
            print(f"  spread: half std mean={std_g_half.mean():.3f} [{std_g_half.min():.3f}, {std_g_half.max():.3f}]  "
                  f"vs full std mean={sub['std_guess_full'].mean():.3f} "
                  f"[{sub['std_guess_full'].min():.3f}, {sub['std_guess_full'].max():.3f}]")


if __name__ == "__main__":
    main()
