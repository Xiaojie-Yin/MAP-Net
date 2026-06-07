import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MAP-Net for 3D CT-to-PET synthesis."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/mapnet_3d.yaml",
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training.",
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Visible GPU id(s), e.g., '0' or '0,1'. For now, training uses a single process.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory.",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root.",
    )

    parser.add_argument(
        "--mask-root",
        type=str,
        default=None,
        help="Override esophagus mask / SDM root.",
    )

    parser.add_argument(
        "--split-csv",
        type=str,
        default=None,
        help="Override patient split CSV path.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override DataLoader worker number.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs.",
    )

    parser.add_argument(
        "--overwrite-split",
        action="store_true",
        help="Regenerate train/val/test split even if split CSV exists.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a short smoke-test training: 1 epoch and 2 iterations.",
    )

    return parser.parse_args()


def apply_cli_overrides(cfg, args):
    """
    Apply command-line overrides to config.
    """
    cfg.setdefault("data", {})
    cfg.setdefault("dataset", {})
    cfg.setdefault("train", {})
    cfg.setdefault("out", {})

    if args.output_dir is not None:
        out_dir = args.output_dir
        cfg["out"]["out_dir"] = out_dir
        cfg["out"]["ckpt_dir"] = os.path.join(out_dir, "ckpt")
        cfg["out"]["vis_dir"] = os.path.join(out_dir, "vis")
        cfg["out"]["log_dir"] = os.path.join(out_dir, "logs")
        cfg["out"]["eval_dir"] = os.path.join(out_dir, "eval")

    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root
        cfg["data_root"] = args.data_root

    if args.mask_root is not None:
        cfg["data"]["mask_root"] = args.mask_root
        cfg["dataset"]["mask_root"] = args.mask_root

    if args.split_csv is not None:
        cfg["data"]["split_csv"] = args.split_csv

    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)

    if args.num_workers is not None:
        cfg["train"]["num_workers"] = int(args.num_workers)

    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)

    if args.overwrite_split:
        cfg["dataset"]["overwrite_split"] = True

    if args.debug:
        cfg["train"]["epochs"] = 1
        cfg["train"]["max_train_iters_per_epoch"] = 2

        cfg["out"]["log_interval"] = 1
        cfg["out"]["csv_interval"] = 1
        cfg["out"]["vis_interval"] = 1
        cfg["out"]["eval_interval"] = 0
        cfg["out"]["save_every"] = 1

        cfg["out"]["out_dir"] = args.output_dir or "outputs/debug_train"
        cfg["out"]["ckpt_dir"] = os.path.join(cfg["out"]["out_dir"], "ckpt")
        cfg["out"]["vis_dir"] = os.path.join(cfg["out"]["out_dir"], "vis")
        cfg["out"]["log_dir"] = os.path.join(cfg["out"]["out_dir"], "logs")
        cfg["out"]["eval_dir"] = os.path.join(cfg["out"]["out_dir"], "eval")

    return cfg


def main():
    args = parse_args()

    # Set visible GPU before importing torch.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Make project root importable when running as:
    # python /path/to/project/train.py
    root_dir = os.path.abspath(os.path.dirname(__file__))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    import torch

    from datasets.build import build_dataloaders
    from engine.trainer import Trainer
    from models.build import build_model
    from utils import (
        copy_config,
        ensure_output_dirs,
        load_config,
        save_config,
        set_seed,
    )

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    system_cfg = cfg.get("system", {})
    set_seed(
        seed=int(cfg.get("seed", 42)),
        deterministic=bool(system_cfg.get("deterministic", False)),
        benchmark=bool(system_cfg.get("cudnn_benchmark", True)),
    )

    dirs = ensure_output_dirs(cfg)

    # Save both source and resolved configs.
    copy_config(args.config, dirs["out_dir"], filename="config_source.yaml")
    save_config(cfg, os.path.join(dirs["out_dir"], "config_resolved.yaml"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("MAP-Net training")
    print(f"Config     : {args.config}")
    print(f"Output dir : {dirs['out_dir']}")
    print(f"Device     : {device}")
    if args.gpu is not None:
        print(f"GPU visible: {args.gpu}")
    print("=" * 80)

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        resume=args.resume,
    )

    trainer.fit()


if __name__ == "__main__":
    main()