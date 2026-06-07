import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .paired_patches_3d import PairedPatches3D


def seed_worker(worker_id):
    """
    Make DataLoader workers deterministic under a fixed torch seed.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _get_data_root(cfg: Dict) -> str:
    if "data" in cfg and "root" in cfg["data"]:
        return cfg["data"]["root"]
    if "data_root" in cfg:
        return cfg["data_root"]
    raise KeyError("data.root or data_root must be specified in config.")


def build_dataset(cfg: Dict, split: str):
    data_cfg = cfg.get("data", {})
    dataset_cfg = cfg.get("dataset", {})
    aug_cfg = cfg.get("augmentation", {})

    root = _get_data_root(cfg)

    split_csv = data_cfg.get("split_csv", None)
    mask_root = data_cfg.get("mask_root", dataset_cfg.get("mask_root", None))
    use_prior = bool(data_cfg.get("use_prior", dataset_cfg.get("use_prior", True)))

    seed = int(cfg.get("seed", 42))

    return PairedPatches3D(
        root=root,
        split=split,
        split_csv=split_csv,
        split_ratios=tuple(dataset_cfg.get("split_ratios", [0.75, 0.125, 0.125])),
        overwrite_split=bool(dataset_cfg.get("overwrite_split", False)),

        mask_root=mask_root,
        use_prior=use_prior,
        allow_missing_prior=bool(dataset_cfg.get("allow_missing_prior", False)),

        patch_size=tuple(dataset_cfg.get("patch_size", [32, 96, 96])),
        hw_target=tuple(dataset_cfg.get("hw_target", [256, 256])),
        stride=tuple(dataset_cfg.get("stride", [16, 48, 48])),
        center_crop_ratio=float(dataset_cfg.get("center_crop_ratio", 1.0)),

        suv_thr=float(dataset_cfg.get("suv_thr", 2.5)),
        balance_positive=bool(dataset_cfg.get("balance_positive", True)),
        pos_ratio=float(dataset_cfg.get("pos_ratio", 0.7)),

        do_aug=bool(aug_cfg.get("enable", True)),
        aug_cfg=aug_cfg,

        return_vis=bool(dataset_cfg.get("return_vis", False)),
        seed=seed,
    )


def build_dataloaders(cfg: Dict, include_test: bool = False):
    train_cfg = cfg.get("train", {})
    seed = int(cfg.get("seed", 42))

    generator = torch.Generator()
    generator.manual_seed(seed)

    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 4))

    train_ds = build_dataset(cfg, split="train")
    val_ds = build_dataset(cfg, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    if not include_test:
        return train_loader, val_loader

    test_ds = build_dataset(cfg, split="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return train_loader, val_loader, test_loader