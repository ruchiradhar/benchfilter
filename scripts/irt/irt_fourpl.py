#!/usr/bin/env python3
"""Train a 4PL IRT model from files in data/irt and save results.

Examples:
    python scripts/irt/irt_fourpl.py --dataset mgsm --language en
    python scripts/irt/irt_fourpl.py --dataset mmlu --language en
    python scripts/irt/irt_fourpl.py --dataset mmlu --language en --category humanities
"""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyro
import torch
from py_irt.config import IrtConfig
from py_irt.dataset import Dataset
from py_irt.models import IrtModel
from py_irt.training import IrtModelTrainer
from pyro.infer import SVI, Trace_ELBO
import pyro.optim
from rich.live import Live
from rich.table import Table


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_value: Path) -> Path:
    if path_value.is_absolute():
        return path_value
    return (PROJECT_ROOT / path_value).resolve()


class StreamToLogger:
    """Redirect writes from stdout/stderr to a logger."""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message: str) -> int:
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.logger.log(self.level, line)
        return len(message)

    def flush(self) -> None:
        remaining = self._buffer.strip()
        if remaining:
            self.logger.log(self.level, remaining)
        self._buffer = ""


class TrackingTrainer(IrtModelTrainer):
    """IrtModelTrainer that records per-epoch ELBO losses in self.losses."""

    def train(self, *, epochs=None, device="cpu"):
        if epochs is None:
            epochs = self._config.epochs
        self._device = device
        self._priors = self._config.priors
        self._epochs = epochs

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
        self.irt_model = irt_model
        pyro.clear_param_store()
        pyro_model = irt_model.get_model()
        pyro_guide = irt_model.get_guide()
        self._pyro_model = pyro_model
        self._pyro_guide = pyro_guide

        device_obj = torch.device(device)
        scheduler = pyro.optim.ExponentialLR({  # type: ignore[attr-defined]
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
        best_loss = float("inf")
        table = Table()
        table.add_column("Epoch")
        table.add_column("Loss")
        table.add_column("Best Loss")
        live = Live(table) if self._verbose else nullcontext()

        with live:
            for epoch in range(epochs):
                loss = float(svi.step(subjects, items, responses))  # type: ignore[arg-type]
                self.losses.append(loss)
                if loss < best_loss:
                    best_loss = loss
                    self.best_params = self.export(items)
                scheduler.step()
                if epoch % self._config.log_every == 0:
                    table.add_row(f"{epoch + 1}", f"{loss:.4f}", f"{best_loss:.4f}")
            table.add_row(f"{epochs}", f"{loss:.4f}", f"{best_loss:.4f}")
            self.last_params = self.export(items)


def save_loss_plot(losses: list[float], plot_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 4PL IRT model for dataset/language data in data/irt."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["mmlu", "mgsm"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--language",
        required=True,
        help="Language code to train for, e.g. en, de, es, zh.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help=(
            "MMLU category filter (e.g. business, humanities, medical, "
            "other, social_sciences, stem). Ignored for MGSM."
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/irt"),
        help="Directory containing converted IRT jsonlines files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/4PL"),
        help="Directory to write trained parameter files.",
    )
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs.")
    parser.add_argument(
        "--log_every",
        type=int,
        default=1000,
        help="Log interval in training epochs.",
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout value.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument(
        "--lr_decay", type=float, default=0.9999, help="Learning rate decay."
    )
    parser.add_argument(
        "--device", default="cpu", help="Torch device for training, e.g. cpu, cuda."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional random seed for reproducibility."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable py-irt console progress output (default is quiet mode).",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path("logs/irt"),
        help="Directory for run logs.",
    )
    args = parser.parse_args()
    args.data_dir = resolve_repo_path(args.data_dir)
    args.output_dir = resolve_repo_path(args.output_dir)
    args.log_dir = resolve_repo_path(args.log_dir)
    return args


def setup_logger(log_dir: Path, dataset: str, language: str) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    run_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    log_path = log_dir / f"irt_fourpl_{dataset}_{language}_{run_date}.log"

    logger = logging.getLogger("irt_fourpl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger, log_path


def normalize_category(category: str) -> str:
    return category.strip().lower().replace("-", "_").replace(" ", "_")


def resolve_input_files(
    dataset: str, language: str, data_dir: Path, category: str | None = None
) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    if dataset == "mgsm":
        if category is not None:
            raise ValueError("--category is only supported with --dataset mmlu")
        exact = data_dir / f"mgsm_{language}.jsonlines"
        if exact.exists():
            return [exact]
        candidates = sorted(data_dir.glob(f"mgsm_{language}*.jsonlines"))
        if not candidates:
            raise FileNotFoundError(
                f"No MGSM files found for language '{language}' in {data_dir}"
            )
        return candidates

    if category is not None:
        category_name = normalize_category(category)
        target = data_dir / f"mmlu_{language}_{category_name}.jsonlines"
        if not target.exists():
            available = sorted(
                p.stem.replace(f"mmlu_{language}_", "")
                for p in data_dir.glob(f"mmlu_{language}_*.jsonlines")
            )
            raise FileNotFoundError(
                f"No MMLU file for language '{language}' and category '{category_name}' in {data_dir}. "
                f"Available categories: {available}"
            )
        return [target]

    candidates = sorted(data_dir.glob(f"mmlu_{language}_*.jsonlines"))
    if not candidates:
        raise FileNotFoundError(
            f"No MMLU files found for language '{language}' in {data_dir}"
        )
    return candidates


def build_output_path(input_file: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_file.stem}_4pl.json"


def train_one_file(input_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    dataset = Dataset.from_jsonlines(input_path)
    config = IrtConfig(
        model_type="4pl",
        epochs=args.epochs,
        log_every=args.log_every,
        dropout=args.dropout,
        lr=args.lr,
        lr_decay=args.lr_decay,
        seed=args.seed,
    )
    trainer = TrackingTrainer(
        config=config,
        data_path=input_path,
        dataset=dataset,
        verbose=args.verbose,
    )
    trainer.train(epochs=args.epochs, device=args.device)
    output_path.write_text(json.dumps(trainer.last_params, indent=2), encoding="utf-8")
    save_loss_plot(trainer.losses, output_path.with_suffix(".png"), f"4PL Training Loss — {input_path.stem}")


def main() -> None:
    args = parse_args()
    logger, _ = setup_logger(args.log_dir, args.dataset, args.language)

    stdout_logger = StreamToLogger(logger, logging.INFO)
    stderr_logger = StreamToLogger(logger, logging.ERROR)

    with redirect_stdout(stdout_logger), redirect_stderr(stderr_logger):
        logger.info("Starting run")
        logger.info(
            "Arguments: dataset=%s language=%s category=%s data_dir=%s output_dir=%s epochs=%d device=%s",
            args.dataset,
            args.language,
            args.category,
            args.data_dir,
            args.output_dir,
            args.epochs,
            args.device,
        )

        if args.seed is not None:
            pyro.set_rng_seed(args.seed)
            torch.manual_seed(args.seed)
            logger.info("Set random seed to %d", args.seed)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        input_files = resolve_input_files(
            args.dataset, args.language, args.data_dir, args.category
        )
        logger.info("Resolved %d input file(s)", len(input_files))

        for input_file in input_files:
            output_file = build_output_path(input_file=input_file, output_dir=args.output_dir)
            logger.info("Training 4PL on %s", input_file)
            train_one_file(input_file, output_file, args)
            logger.info("Saved model params to %s", output_file)
        logger.info("Run completed successfully")

if __name__ == "__main__":
    main()
