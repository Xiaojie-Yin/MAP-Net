import os
import sys

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from utils import (
    CSVLogger,
    TRAIN_LOG_FIELDS,
    ct_from_norm,
    ensure_output_dirs,
    load_config,
    make_batch_panel,
    mask_debug_stats,
    pet_from_norm,
    set_seed,
)


def main():
    set_seed(42)

    cfg = load_config("configs/mapnet_3d.yaml")
    dirs = ensure_output_dirs(cfg)
    print("Output dirs:", dirs)

    x = torch.randn(2, 2, 32, 96, 96)
    pet = torch.randn(2, 1, 32, 96, 96)
    pred = torch.randn(2, 1, 32, 96, 96)
    mask_logits = torch.randn(2, 2, 32, 96, 96)

    ct_abs = ct_from_norm(x[:, 0:1])
    pet_abs = pet_from_norm(pet)

    print("ct_abs:", ct_abs.shape, float(ct_abs.min()), float(ct_abs.max()))
    print("pet_abs:", pet_abs.shape, float(pet_abs.min()), float(pet_abs.max()))

    stats = mask_debug_stats(mask_logits)
    print("mask stats:", stats)

    panel = make_batch_panel(
        src=x,
        pet=pet,
        pred=pred,
        mask_logits=mask_logits,
        suv_thr=2.5,
        nrow=2,
        enable_mask=True,
    )
    print("panel:", panel.shape)

    logger = CSVLogger(
        os.path.join(dirs["log_dir"], "check_utils_log.csv"),
        TRAIN_LOG_FIELDS,
    )
    logger.write({
        "epoch": 0,
        "iter": 0,
        "D": 1.0,
        "G": 2.0,
    })

    print("Utils check passed.")


if __name__ == "__main__":
    main()