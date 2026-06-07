import argparse
import csv
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MAP-Net inference and save synthesized PET volumes."
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
        help="Path to trained checkpoint.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split for inference.",
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
        help="Directory to save inference outputs.",
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
        help="Run inference on only first N patients.",
    )

    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Save auxiliary high-uptake mask probability map.",
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

    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root
        cfg["data_root"] = args.data_root

    if args.mask_root is not None:
        cfg["data"]["mask_root"] = args.mask_root
        cfg["dataset"]["mask_root"] = args.mask_root

    if args.split_csv is not None:
        cfg["data"]["split_csv"] = args.split_csv

    return cfg


def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    root_dir = os.path.abspath(os.path.dirname(__file__))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    import numpy as np
    import torch

    from datasets.build import build_dataset
    from engine.checkpoint import load_checkpoint
    from engine.evaluator import SlidingWindowEvaluator
    from models.build import build_model
    from utils import load_config, set_seed

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    system_cfg = cfg.get("system", {})
    set_seed(
        seed=int(cfg.get("seed", 42)),
        deterministic=bool(system_cfg.get("deterministic", False)),
        benchmark=bool(system_cfg.get("cudnn_benchmark", True)),
    )

    if args.output_dir is None:
        out_dir = os.path.join(
            cfg.get("out", {}).get("out_dir", "outputs/mapnet_3d"),
            "inference",
            args.split,
        )
    else:
        out_dir = args.output_dir

    os.makedirs(out_dir, exist_ok=True)

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
        compute_perceptual=False,
    )

    pids = list(dataset.volumes.keys())
    if args.max_patients is not None:
        pids = pids[: int(args.max_patients)]

    manifest_path = os.path.join(out_dir, "manifest.csv")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pid", "spet_path", "mask_path"],
        )
        writer.writeheader()

        for i, pid in enumerate(pids):
            print(f"[Inference] {i + 1}/{len(pids)} | pid={pid}")

            pred_pet, pred_mask = evaluator.predict_patient(dataset, pid)

            pid_dir = os.path.join(out_dir, pid)
            os.makedirs(pid_dir, exist_ok=True)

            spet_path = os.path.join(pid_dir, "sPET_SUV.npy")
            np.save(spet_path, pred_pet.astype(np.float32))

            mask_path = ""
            if args.save_mask:
                mask_path = os.path.join(pid_dir, "high_uptake_mask_prob.npy")
                np.save(mask_path, pred_mask.astype(np.float32))

            writer.writerow(
                {
                    "pid": pid,
                    "spet_path": spet_path,
                    "mask_path": mask_path,
                }
            )

    print(f"[Saved] Inference outputs: {out_dir}")
    print(f"[Saved] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()