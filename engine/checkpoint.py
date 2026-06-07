import os
from typing import Any, Dict, Optional

import torch


def unwrap_model(model):
    """
    Return the underlying model if DataParallel/DistributedDataParallel is used.
    """
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    path: str,
    model,
    optimizer_g=None,
    optimizer_d=None,
    scheduler_g=None,
    epoch: int = 0,
    best_val: Optional[float] = None,
    cfg: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a full training checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    base_model = unwrap_model(model)

    checkpoint = {
        "epoch": int(epoch),
        "model": base_model.state_dict(),
        "best_val": best_val,
        "config": cfg,
    }

    if optimizer_g is not None:
        checkpoint["optimizer_g"] = optimizer_g.state_dict()

    if optimizer_d is not None:
        checkpoint["optimizer_d"] = optimizer_d.state_dict()

    if scheduler_g is not None:
        checkpoint["scheduler_g"] = scheduler_g.state_dict()

    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    path: str,
    model,
    optimizer_g=None,
    optimizer_d=None,
    scheduler_g=None,
    map_location="cpu",
    strict: bool = True,
):
    """
    Load checkpoint.

    Supports both:
        1. new checkpoint dict: {"model": state_dict, ...}
        2. old style pure model.state_dict()
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)
    base_model = unwrap_model(model)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    missing, unexpected = base_model.load_state_dict(state_dict, strict=strict)

    if optimizer_g is not None and isinstance(checkpoint, dict) and "optimizer_g" in checkpoint:
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])

    if optimizer_d is not None and isinstance(checkpoint, dict) and "optimizer_d" in checkpoint:
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])

    if scheduler_g is not None and isinstance(checkpoint, dict) and "scheduler_g" in checkpoint:
        scheduler_g.load_state_dict(checkpoint["scheduler_g"])

    return {
        "checkpoint": checkpoint,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def get_checkpoint_epoch(checkpoint) -> int:
    """
    Return saved epoch from checkpoint dict.
    """
    if isinstance(checkpoint, dict):
        return int(checkpoint.get("epoch", -1))
    return -1


def get_checkpoint_best_val(checkpoint, default=float("inf")) -> float:
    """
    Return best validation loss from checkpoint dict.
    """
    if isinstance(checkpoint, dict) and checkpoint.get("best_val", None) is not None:
        return float(checkpoint["best_val"])
    return float(default)