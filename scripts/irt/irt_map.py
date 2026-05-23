#!/usr/bin/env python3
"""
MAP-based IRT fitting for 1PL, 2PL, and 3PL using direct gradient descent.

Replaces py-irt's SVI for 2PL/3PL, which collapses discrimination parameters
toward zero. Also provides a MAP-1PL variant for inference-method comparisons.
Key differences:
  - Discrimination constrained positive via softplus (a = softplus(raw_a))
  - Guessing (3PL) constrained to [0,1] via sigmoid
  - Joint MAP objective (log-likelihood + Normal priors) optimised with Adam
  - Multiple random restarts; best loss retained

Output format is identical to py-irt so all downstream analysis scripts work
unchanged.

Usage:
    python scripts/irt/irt_map.py --model 1pl
    python scripts/irt/irt_map.py --model 2pl
    python scripts/irt/irt_map.py --model 3pl
    python scripts/irt/irt_map.py --model 2pl --output_dir results/2PL_map
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── IRT models ────────────────────────────────────────────────────────────────

def _softplus_inv(x: float) -> float:
    """Inverse softplus: find raw such that softplus(raw) = x."""
    return float(np.log(np.exp(x) - 1))


class OnePL(torch.nn.Module):
    """1PL (Rasch) IRT: P = sigmoid(theta_i - b_j)."""

    def __init__(self, n_subjects: int, n_items: int, device: torch.device) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(n_subjects, device=device))
        self.b = torch.nn.Parameter(torch.zeros(n_items, device=device))

    def log_likelihood(
        self, subjects: torch.Tensor, items: torch.Tensor, responses: torch.Tensor
    ) -> torch.Tensor:
        logits = self.theta[subjects] - self.b[items]
        return -F.binary_cross_entropy_with_logits(logits, responses, reduction="sum")

    def log_prior(self) -> torch.Tensor:
        lp = -0.5 * (self.theta ** 2).sum()
        lp += -0.5 * (self.b ** 2).sum()
        return lp


class TwoPL(torch.nn.Module):
    """2PL IRT: P = sigmoid(a_j * (theta_i - b_j)), a_j > 0."""

    def __init__(self, n_subjects: int, n_items: int, device: torch.device) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(n_subjects, device=device))
        self.b = torch.nn.Parameter(torch.zeros(n_items, device=device))
        # raw_a passed through softplus to enforce a > 0; initialise near a=1
        init_raw_a = _softplus_inv(1.0)
        self.raw_a = torch.nn.Parameter(
            torch.full((n_items,), init_raw_a, device=device)
        )

    @property
    def a(self) -> torch.Tensor:
        return F.softplus(self.raw_a)

    def log_likelihood(
        self, subjects: torch.Tensor, items: torch.Tensor, responses: torch.Tensor
    ) -> torch.Tensor:
        logits = self.a[items] * (self.theta[subjects] - self.b[items])
        return -F.binary_cross_entropy_with_logits(logits, responses, reduction="sum")

    def log_prior(self) -> torch.Tensor:
        # theta ~ N(0,1), b ~ N(0,1), log(a) ~ N(0, 0.5)
        # The tighter sigma=0.5 prior on log(a) prevents boundary solutions
        # (disc→∞) on items that perfectly separate the model population.
        log_a = torch.log(self.a)
        lp = -0.5 * (self.theta ** 2).sum()
        lp += -0.5 * (self.b ** 2).sum()
        lp += -0.5 * ((log_a / 0.5) ** 2).sum() - log_a.sum()  # LogNormal(0, 0.5)
        return lp


class ThreePL(torch.nn.Module):
    """3PL IRT: P = c_j + (1-c_j)*sigmoid(a_j*(theta_i-b_j)), a_j>0, c_j in [0,1]."""

    def __init__(self, n_subjects: int, n_items: int, device: torch.device) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(n_subjects, device=device))
        self.b = torch.nn.Parameter(torch.zeros(n_items, device=device))
        init_raw_a = _softplus_inv(1.0)
        self.raw_a = torch.nn.Parameter(
            torch.full((n_items,), init_raw_a, device=device)
        )
        # logit_c such that c = sigmoid(logit_c); init near 0.25 (4-choice chance)
        init_logit_c = float(np.log(0.25 / 0.75))
        self.logit_c = torch.nn.Parameter(
            torch.full((n_items,), init_logit_c, device=device)
        )

    @property
    def a(self) -> torch.Tensor:
        return F.softplus(self.raw_a)

    @property
    def c(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_c)

    def log_likelihood(
        self, subjects: torch.Tensor, items: torch.Tensor, responses: torch.Tensor
    ) -> torch.Tensor:
        p_star = torch.sigmoid(self.a[items] * (self.theta[subjects] - self.b[items]))
        p = self.c[items] + (1 - self.c[items]) * p_star
        p = p.clamp(1e-7, 1 - 1e-7)
        ll = responses * torch.log(p) + (1 - responses) * torch.log(1 - p)
        return ll.sum()

    def log_prior(self) -> torch.Tensor:
        log_a = torch.log(self.a)
        lp = -0.5 * (self.theta ** 2).sum()
        lp += -0.5 * (self.b ** 2).sum()
        lp += -0.5 * ((log_a / 0.5) ** 2).sum() - log_a.sum()  # LogNormal(0, 0.5)
        # c ~ Beta(2,8) encourages small guessing; approx via logit_c ~ N(-2, 1)
        lp += -0.5 * ((self.logit_c + 2.0) ** 2).sum()
        return lp


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonlines(path: Path) -> tuple[list[str], list[str], torch.Tensor]:
    """Return (subject_ids, item_ids, response_matrix[n_subjects, n_items])."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    all_items: list[str] = []
    for rec in records:
        for k in rec["responses"]:
            if k not in all_items:
                all_items.append(k)
    # sort item ids by numeric suffix for consistency
    all_items = sorted(all_items, key=lambda x: int(x.split("_")[-1]))

    item_idx = {item: i for i, item in enumerate(all_items)}
    subject_ids = [rec["subject_id"] for rec in records]

    n_s, n_i = len(records), len(all_items)
    R = torch.zeros(n_s, n_i)
    for s_idx, rec in enumerate(records):
        for item, resp in rec["responses"].items():
            R[s_idx, item_idx[item]] = float(resp)

    return subject_ids, all_items, R


# ── Training ──────────────────────────────────────────────────────────────────

def train_one(
    model_cls: type,
    subjects: torch.Tensor,
    items: torch.Tensor,
    responses: torch.Tensor,
    n_subjects: int,
    n_items: int,
    epochs: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> tuple[torch.nn.Module, float]:
    """One training run from a given seed. Returns (model, best_neg_elbo)."""
    torch.manual_seed(seed)
    model = model_cls(n_subjects, n_items, device)

    # Randomise initial theta, b slightly; keep a near 1
    with torch.no_grad():
        model.theta.add_(torch.randn_like(model.theta) * 0.5)
        model.b.add_(torch.randn_like(model.b) * 0.5)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = -(model.log_likelihood(subjects, items, responses) + model.log_prior())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        val = loss.item()
        if val < best_loss:
            best_loss = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_loss


def fit(
    path: Path,
    model_name: str,
    epochs: int,
    lr: float,
    n_restarts: int,
    device: torch.device,
    logger: logging.Logger,
) -> dict:
    subject_ids, item_ids, R = load_jsonlines(path)
    n_s, n_i = R.shape

    R = R.to(device)
    s_idx = torch.arange(n_s, device=device).repeat_interleave(n_i)
    i_idx = torch.arange(n_i, device=device).repeat(n_s)
    resp = R.flatten()

    if model_name == "1pl":
        model_cls = OnePL
    elif model_name == "2pl":
        model_cls = TwoPL
    else:
        model_cls = ThreePL

    best_model: torch.nn.Module
    best_loss = float("inf")
    for restart in range(n_restarts):
        m, loss = train_one(
            model_cls, s_idx, i_idx, resp, n_s, n_i,
            epochs=epochs, lr=lr, device=device, seed=restart * 42,
        )
        logger.info("  restart %d/%d  loss=%.4f", restart + 1, n_restarts, loss)
        if loss < best_loss:
            best_loss, best_model = loss, m

    with torch.no_grad():
        ability = best_model.theta.cpu().tolist()  # type: ignore[union-attr]
        diff = best_model.b.cpu().tolist()          # type: ignore[union-attr]

    out: dict = {
        "ability": ability,
        "diff": diff,
        "irt_model": model_name,
        "item_ids": {str(i): v for i, v in enumerate(item_ids)},
        "subject_ids": {str(i): v for i, v in enumerate(subject_ids)},
    }
    if model_name in ("2pl", "3pl"):
        with torch.no_grad():
            out["disc"] = best_model.a.cpu().tolist()  # type: ignore[union-attr]
    if model_name == "3pl":
        with torch.no_grad():
            out["lambdas"] = best_model.c.cpu().tolist()  # type: ignore[union-attr]

    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAP-based 2PL/3PL IRT fitting.")
    parser.add_argument("--model", choices=["1pl", "2pl", "3pl"], required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data/irt"))
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Defaults to results/2PL_map or results/3PL_map")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs (default: 4000 for 2pl, 6000 for 3pl)")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_dir", type=Path, default=Path("logs/irt"))
    args = parser.parse_args()

    def resolve(p: Path) -> Path:
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    args.data_dir = resolve(args.data_dir)
    args.log_dir = resolve(args.log_dir)

    if args.output_dir is None:
        label = {"1pl": "1PL_map", "2pl": "2PL_map", "3pl": "3PL_map"}[args.model]
        args.output_dir = PROJECT_ROOT / "results" / label
    else:
        args.output_dir = resolve(args.output_dir)

    if args.epochs is None:
        args.epochs = {"1pl": 2000, "2pl": 4000, "3pl": 6000}[args.model]

    return args


def setup_logger(log_dir: Path, model_name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    run_date = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    log_path = log_dir / f"irt_map_{model_name}_{run_date}.log"

    logger = logging.getLogger(f"irt_map_{model_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_dir, args.model)
    device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_files = sorted(args.data_dir.rglob("*.jsonlines"))
    if not input_files:
        raise FileNotFoundError(f"No jsonlines files in {args.data_dir}")

    logger.info(
        "MAP %s | %d files | epochs=%d lr=%.4f restarts=%d device=%s",
        args.model.upper(), len(input_files), args.epochs, args.lr, args.restarts, args.device,
    )

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(), transient=True,
    ) as progress:
        task = progress.add_task("Fitting...", total=len(input_files))

        for input_file in input_files:
            slice_name = input_file.stem
            out_path = args.output_dir / f"{slice_name}_{args.model}.json"
            progress.update(task, description=f"{slice_name}")
            logger.info("Fitting %s on %s", args.model.upper(), slice_name)

            params = fit(
                path=input_file,
                model_name=args.model,
                epochs=args.epochs,
                lr=args.lr,
                n_restarts=args.restarts,
                device=device,
                logger=logger,
            )
            out_path.write_text(json.dumps(params, indent=2))
            logger.info("Saved → %s", out_path)
            progress.advance(task)

    logger.info("Done. Results in %s", args.output_dir)


if __name__ == "__main__":
    main()
