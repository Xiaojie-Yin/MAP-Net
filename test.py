import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MAP-Net with whole-volume sliding-window prediction."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/mapnet_3d.yaml",
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint, e.g., outputs/mapnet_3d/ckpt/best.pth.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Visible GPU id, e.g., '0'.",
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
        "--max-patients",
        type=int,
        default=None,
        help="Evaluate only first N patients for debugging.",
    )

    parser.add_argument(
        "--compute-perceptual",
        action="store_true",
        help="Compute LPIPS and GMSD. This is slower and requires lpips/piq.",
    )

    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Load checkpoint with strict=False.",
    )

    return parser.parse_args()


def apply_cli_overrides(cfg, args):
    cfg.setdefault("data", {})
    cfg.setdefault("dataset", {})
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

    return cfg


def print_summary(summary, split):
    print("=" * 80)
    print(f"Evaluation summary | split={split}")
    print("=" * 80)

    for key in [
        "SSIM",
        "PSNR",
        "MAE",
        "Dice",
        "HD95",
        "HFEN",
        "GradMAE",
        "LPIPS",
        "GMSD",
    ]:
        if key in summary:
            print(f"{key:>8s}: {summary[key]:.6f}")

    print("=" * 80)


def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    root_dir = os.path.abspath(os.path.dirname(__file__))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    import torch

    from datasets.build import build_dataset
    from engine.checkpoint import load_checkpoint
    from engine.evaluator import SlidingWindowEvaluator
    from models.build import build_model
    from utils import ensure_output_dirs, load_config, save_config, set_seed

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    system_cfg = cfg.get("system", {})
    set_seed(
        seed=int(cfg.get("seed", 42)),
        deterministic=bool(system_cfg.get("deterministic", False)),
        benchmark=bool(system_cfg.get("cudnn_benchmark", True)),
    )

    dirs = ensure_output_dirs(cfg)

    save_config(
        cfg,
        os.path.join(dirs["out_dir"], f"config_test_{args.split}.yaml"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = build_dataset(cfg, split=args.split)

    model = build_model(cfg).to(device)

    load_result = load_checkpoint(
        path=args.checkpoint,
        model=model,
        map_location=device,
        strict=not args.non_strict,
    )

    print(f"[Checkpoint] Loaded: {args.checkpoint}")
    if load_result["missing_keys"]:
        print("[Checkpoint] Missing keys:", load_result["missing_keys"])
    if load_result["unexpected_keys"]:
        print("[Checkpoint] Unexpected keys:", load_result["unexpected_keys"])

    dataset_cfg = cfg.get("dataset", {})
    inference_cfg = cfg.get("inference", {})
    mask_cfg = cfg.get("mask", {})
    data_cfg = cfg.get("data", {})

    patch_size = tuple(
        inference_cfg.get(
            "patch_size",
            dataset_cfg.get("patch_size", [32, 96, 96]),
        )
    )
    stride = tuple(
        inference_cfg.get(
            "stride",
            dataset_cfg.get("stride", [16, 48, 48]),
        )
    )

    suv_thr = float(
        mask_cfg.get("threshold", {}).get(
            "suv",
            dataset_cfg.get("suv_thr", 2.5),
        )
    )

    enable_mask = bool(
        mask_cfg.get(
            "enable",
            cfg.get("model", {}).get("enable_mask", True),
        )
    )

    use_prior = bool(
        data_cfg.get(
            "use_prior",
            dataset_cfg.get("use_prior", True),
        )
    )

    evaluator = SlidingWindowEvaluator(
        model=model,
        device=device,
        patch_size=patch_size,
        stride=stride,
        suv_thr=suv_thr,
        enable_mask=enable_mask,
        use_prior=use_prior,
        compute_perceptual=args.compute_perceptual,
    )

    rows = evaluator.evaluate_dataset(
        dataset,
        max_patients=args.max_patients,
        shuffle=False,
    )

    summary = evaluator.summarize(rows)

    patient_csv = os.path.join(
        dirs["eval_dir"],
        f"{args.split}_patient_metrics.csv",
    )
    summary_csv = os.path.join(
        dirs["eval_dir"],
        f"{args.split}_summary.csv",
    )

    evaluator.save_patient_csv(rows, patient_csv)
    evaluator.append_summary_csv(
        summary=summary,
        csv_path=summary_csv,
        epoch=0,
        iteration=0,
        split=args.split,
    )

    print_summary(summary, args.split)

    print(f"[Saved] Patient metrics: {patient_csv}")
    print(f"[Saved] Summary metrics: {summary_csv}")


if __name__ == "__main__":
    main()