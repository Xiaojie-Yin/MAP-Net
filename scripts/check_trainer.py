import os
import sys

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from datasets.build import build_dataloaders
from engine.trainer import Trainer
from models.build import build_model
from utils import ensure_output_dirs, load_config, set_seed


def main():
    cfg = load_config("configs/mapnet_3d.yaml")

    # Smoke test settings. Do not affect the original yaml file.
    cfg["out"]["out_dir"] = "outputs/check_trainer"
    cfg["out"]["ckpt_dir"] = "outputs/check_trainer/ckpt"
    cfg["out"]["vis_dir"] = "outputs/check_trainer/vis"
    cfg["out"]["log_dir"] = "outputs/check_trainer/logs"
    cfg["out"]["eval_dir"] = "outputs/check_trainer/eval"

    cfg["train"]["epochs"] = 1
    cfg["train"]["max_train_iters_per_epoch"] = 2
    cfg["out"]["log_interval"] = 1
    cfg["out"]["csv_interval"] = 1
    cfg["out"]["vis_interval"] = 1
    cfg["out"]["eval_interval"] = 0
    cfg["out"]["save_every"] = 1

    # Reduce memory for smoke test if needed.
    # Comment this out when testing the real training setting.
    cfg["train"]["batch_size"] = 2

    system_cfg = cfg.get("system", {})
    set_seed(
        cfg.get("seed", 42),
        deterministic=bool(system_cfg.get("deterministic", False)),
        benchmark=bool(system_cfg.get("cudnn_benchmark", True)),
    )

    ensure_output_dirs(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        resume=None,
    )

    trainer.fit()

    print("Trainer check passed.")


if __name__ == "__main__":
    main()