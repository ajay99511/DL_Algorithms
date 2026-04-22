import logging
import torch
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FRESH_START: dict[str, Any] = {
    "epoch": 0,
    "step": 0,
    "best_metric": float("inf"),
    "extra": {},
}


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step: int,
    best_metric: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save training state to a .pt checkpoint file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "config": extra or {},
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    """Load checkpoint and restore model/optimizer/scheduler states.

    Returns a dict with keys: epoch, step, best_metric, extra.
    On FileNotFoundError or RuntimeError, logs a warning and returns a
    fresh-start sentinel: {epoch: 0, step: 0, best_metric: inf, extra: {}}.
    """
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except FileNotFoundError:
        logger.warning("Checkpoint not found at '%s' — starting from scratch.", path)
        return dict(_FRESH_START)
    except RuntimeError as exc:
        logger.warning("Failed to load checkpoint at '%s': %s — starting from scratch.", path, exc)
        return dict(_FRESH_START)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "best_metric": checkpoint.get("best_metric", float("inf")),
        "extra": checkpoint.get("config", {}),
    }
