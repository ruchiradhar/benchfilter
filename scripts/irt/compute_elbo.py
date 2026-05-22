#!/usr/bin/env python3
"""Re-train 1PL / 2PL / 3PL on every data/irt slice and record minimum ELBO.

The ELBO is normalised by the number of item-response observations so that
values are comparable across slices of different sizes. Outputs a JSON summary
and a LaTeX table fragment for inclusion in the paper.

Usage:
    python scripts/irt/compute_elbo.py
    python scripts/irt/compute_elbo.py --output_dir results/elbo --epochs_1pl 2000 --epochs_2pl 3000 --epochs_3pl 3000
"""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path

import pyro
import torch
from py_irt.config import IrtConfig
from py_irt.dataset import Dataset
from py_irt.models import IrtModel
from py_irt.training import IrtModelTrainer
from pyro.infer import SVI, Trace_ELBO
import pyro.optim


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(p: Path) -> Path:
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


class TrackingTrainer(IrtModelTrainer):
    """IrtModelTrainer that records per-epoch ELBO and the best (minimum) loss."""

    def train(self, *, epochs=None, device="cpu"):
        if epochs is None:
            epochs = self._config.epochs
        self._device = device

        args = {
            "device": device,
            "num_items": len(self._dataset.ix_to_item_id),
            "num_subjects": len(self._dataset.ix_to_subject_id),
            "priors": self._config.priors if self._config.priors is not None else "vague",
            "dropout": self._config.dropout,
            "hidden": self._config.hidden,
            "vocab_size": self._config.vocab_size,
        }
        if self._config.dims is not None:
            args["dims"] = self._config.dims

        model_type = self._config.model_type
        irt_model = (
            IrtModel.from_name(model_type)(**args)
            if isinstance(model_type, str)
            else model_type(**args)
        )
        pyro.clear_param_store()
        pyro_model = irt_model.get_model()
        pyro_guide = irt_model.get_guide()

        device_obj = torch.device(device)
        scheduler = pyro.optim.ExponentialLR({
            "optimizer": torch.optim.Adam,
            "optim_args": {"lr": self._config.lr},
            "gamma": self._config.lr_decay,
        })
        svi = SVI(pyro_model, pyro_guide, scheduler, loss=Trace_ELBO())

        subjects = torch.tensor(self._dataset.observation_subjects, dtype=torch.long, device=device_obj)
        items = torch.tensor(self._dataset.observation_items, dtype=torch.long, device=device_obj)
        responses = torch.tensor(self._dataset.observations, dtype=torch.float, device=device_obj)

        _ = pyro_model(subjects, items, responses)
        _ = pyro_guide(subjects, items, responses)
        for init in self._initializers:
            init.initialize()

        self.losses: list[float] = []
        self.best_loss = float("inf")

        for epoch in range(epochs):
            loss = float(svi.step(subjects, items, responses))
            self.losses.append(loss)
            if loss < self.best_loss:
                self.best_loss = loss
            scheduler.step()

        self.n_obs = int(subjects.shape[0])


def train_slice(data_path: Path, model_type: str, epochs: int, lr: float,
                lr_decay: float, dropout: float, device: str) -> dict:
    """Train one IRT variant on one slice; return ELBO summary."""
    dataset = Dataset.from_jsonlines(data_path)
    config = IrtConfig(
        model_type=model_type,
        epochs=epochs,
        log_every=epochs + 1,  # suppress internal logging
        dropout=dropout,
        lr=lr,
        lr_decay=lr_decay,
    )
    trainer = TrackingTrainer(config=config, data_path=data_path, dataset=dataset, verbose=False)
    trainer.train(epochs=epochs, device=device)
    return {
        "best_elbo": trainer.best_loss,
        "n_obs": trainer.n_obs,
        "elbo_per_obs": trainer.best_loss / trainer.n_obs,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute minimum ELBO per IRT model and slice.")
    p.add_argument("--data_dir", type=Path, default=Path("data/irt"))
    p.add_argument("--output_dir", type=Path, default=Path("results/elbo"))
    p.add_argument("--epochs_1pl", type=int, default=2000)
    p.add_argument("--epochs_2pl", type=int, default=3000)
    p.add_argument("--epochs_3pl", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--lr_decay", type=float, default=0.9999)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    args.data_dir = resolve_repo_path(args.data_dir)
    args.output_dir = resolve_repo_path(args.output_dir)
    return args


def slice_label(stem: str) -> str:
    """Convert filename stem to a readable slice label."""
    return stem.replace("_", " ").upper()


def make_latex_table(results: dict) -> str:
    """Generate a LaTeX table of mean ELBO/obs by model type and benchmark."""
    gsm_1pl, gsm_2pl, gsm_3pl = [], [], []
    mmlu_1pl, mmlu_2pl, mmlu_3pl = [], [], []

    for stem, variants in results.items():
        is_gsm = stem.startswith("mgsm")
        for v_label, elbo_per_obs in [
            ("1pl", variants.get("1pl", {}).get("elbo_per_obs")),
            ("2pl", variants.get("2pl", {}).get("elbo_per_obs")),
            ("3pl", variants.get("3pl", {}).get("elbo_per_obs")),
        ]:
            if elbo_per_obs is None:
                continue
            target = (gsm_1pl if is_gsm else mmlu_1pl) if v_label == "1pl" else \
                     (gsm_2pl if is_gsm else mmlu_2pl) if v_label == "2pl" else \
                     (gsm_3pl if is_gsm else mmlu_3pl)
            target.append(elbo_per_obs)

    def fmt(vals):
        if not vals:
            return "--"
        return f"{sum(vals)/len(vals):.4f}"

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{lcc}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{MGSM} & \textbf{MMLU} \\",
        r"    & ELBO/obs & ELBO/obs \\",
        r"    \midrule",
        f"    1PL & {fmt(gsm_1pl)} & {fmt(mmlu_1pl)} \\\\",
        f"    2PL & {fmt(gsm_2pl)} & {fmt(mmlu_2pl)} \\\\",
        f"    3PL & {fmt(gsm_3pl)} & {fmt(mmlu_3pl)} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Mean minimum ELBO per observation across all slices for each IRT",
        r"    model variant, averaged over MGSM languages and MMLU language$\times$domain",
        r"    slices respectively. Lower = better within-language fit. 2PL and 3PL achieve",
        r"    better fit at the cost of cross-lingual stability (Tables~\ref{tab:science_vs_nonsci},",
        r"    \ref{tab:rank_std}).}",
        r"  \label{tab:elbo}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("compute_elbo")

    data_files = sorted(args.data_dir.rglob("*.jsonlines"))
    if not data_files:
        raise FileNotFoundError(f"No .jsonlines files found in {args.data_dir}")

    log.info("Found %d slices", len(data_files))

    model_configs = [
        ("1pl", args.epochs_1pl),
        ("2pl", args.epochs_2pl),
        ("3pl", args.epochs_3pl),
    ]

    results: dict[str, dict] = {}

    # Suppress py-irt / pyro stdout noise
    import os, sys
    devnull = open(os.devnull, "w")

    for data_file in data_files:
        stem = data_file.stem
        results[stem] = {}
        for model_type, epochs in model_configs:
            log.info("  %s  %s  (%d epochs)", stem, model_type, epochs)
            try:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    summary = train_slice(
                        data_path=data_file,
                        model_type=model_type,
                        epochs=epochs,
                        lr=args.lr,
                        lr_decay=args.lr_decay,
                        dropout=args.dropout,
                        device=args.device,
                    )
                results[stem][model_type] = summary
                log.info("    ELBO/obs = %.4f  (n_obs=%d)", summary["elbo_per_obs"], summary["n_obs"])
            except Exception as exc:
                log.error("    FAILED: %s", exc)
                results[stem][model_type] = {"error": str(exc)}

    devnull.close()

    # Save raw results
    raw_path = args.output_dir / "elbo_results.json"
    raw_path.write_text(json.dumps(results, indent=2))
    log.info("Saved raw results to %s", raw_path)

    # Generate and save LaTeX table
    latex = make_latex_table(results)
    latex_path = args.output_dir / "elbo_table.tex"
    latex_path.write_text(latex)
    log.info("Saved LaTeX table to %s", latex_path)

    # Print summary to stdout
    print("\n=== ELBO per observation summary ===")
    print(f"{'Slice':<35} {'1PL':>10} {'2PL':>10} {'3PL':>10}")
    print("-" * 67)
    for stem, variants in sorted(results.items()):
        row = [stem[:34]]
        for v in ("1pl", "2pl", "3pl"):
            val = variants.get(v, {})
            if "elbo_per_obs" in val:
                row.append(f"{val['elbo_per_obs']:>10.4f}")
            else:
                row.append(f"{'ERR':>10}")
        print("  ".join(row))

    print(f"\nLaTeX table written to: {latex_path}")
    print(f"Raw JSON written to:    {raw_path}")


if __name__ == "__main__":
    main()
