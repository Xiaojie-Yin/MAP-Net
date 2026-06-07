import os
import sys
import yaml
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from datasets.build import build_dataloaders
from models.build import build_model
from engine.evaluator import SlidingWindowEvaluator


def main():
    with open("configs/mapnet_3d.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_dataloaders(cfg, include_test=True)

    model = build_model(cfg).to(device)
    model.eval()

    dataset_cfg = cfg["dataset"]
    mask_cfg = cfg.get("mask", {})
    data_cfg = cfg.get("data", {})

    evaluator = SlidingWindowEvaluator(
        model=model,
        device=device,
        patch_size=tuple(dataset_cfg.get("patch_size", [32, 96, 96])),
        stride=tuple(dataset_cfg.get("stride", [16, 48, 48])),
        suv_thr=float(mask_cfg.get("threshold", {}).get("suv", dataset_cfg.get("suv_thr", 2.5))),
        enable_mask=bool(mask_cfg.get("enable", True)),
        use_prior=bool(data_cfg.get("use_prior", dataset_cfg.get("use_prior", True))),
        compute_perceptual=False,
    )

    rows = evaluator.evaluate_dataset(
        val_loader.dataset,
        max_patients=1,
        shuffle=False,
    )

    print("Rows:", rows)

    summary = evaluator.summarize(rows)
    print("Summary:", summary)

    assert len(rows) > 0
    assert "SSIM" in summary
    assert "PSNR" in summary
    assert "MAE" in summary

    print("Evaluator check passed.")


if __name__ == "__main__":
    main()