import csv
import os
import random
from typing import Dict, Optional

import numpy as np
import torch

from utils.metrics import compute_pet_metrics, PerceptualMetricComputer


def merge_patches(vol_shape, patch_size, patch_results):
    """
    Average overlapping sliding-window predictions.
    """
    D, H, W = vol_shape
    pd, ph, pw = patch_size

    output = np.zeros(vol_shape, dtype=np.float32)
    counter = np.zeros(vol_shape, dtype=np.float32)

    for z, y, x, patch in patch_results:
        output[z:z + pd, y:y + ph, x:x + pw] += patch
        counter[z:z + pd, y:y + ph, x:x + pw] += 1.0

    counter[counter == 0] = 1.0
    return output / counter


class SlidingWindowEvaluator:
    """
    Whole-volume sliding-window evaluator for MAP-Net.
    """

    def __init__(
        self,
        model,
        device,
        patch_size=(32, 96, 96),
        stride=(16, 48, 48),
        suv_thr=2.5,
        enable_mask=True,
        use_prior=True,
        compute_perceptual=False,
        data_range=20.0,
    ):
        self.model = model
        self.device = device

        self.patch_size = tuple(patch_size)
        self.stride = tuple(stride)

        self.suv_thr = float(suv_thr)
        self.enable_mask = bool(enable_mask)
        self.use_prior = bool(use_prior)
        self.data_range = float(data_range)

        self.perceptual = PerceptualMetricComputer(
            device=device,
            enabled=compute_perceptual,
        )

    @torch.no_grad()
    def predict_patient(self, dataset, pid):
        """
        Sliding-window PET synthesis for one patient.

        Returns:
            pred_pet: synthesized PET in SUV domain, [D, H, W].
            pred_mask: auxiliary mask probability/binary map averaged over patches.
        """
        self.model.eval()

        ct_v = dataset.volumes[pid]["ct"]
        pt_v = dataset.volumes[pid]["pt"]
        df_v = dataset.volumes[pid].get("df", np.zeros_like(ct_v, dtype=np.float32))

        D, H, W = ct_v.shape
        pd, ph, pw = self.patch_size
        sd, sh, sw = self.stride

        zs = dataset._compute_starts(D, pd, sd)
        ys = dataset._compute_starts(H, ph, sh)
        xs = dataset._compute_starts(W, pw, sw)

        ct_t = torch.from_numpy(ct_v).float()[None, None].to(self.device)
        df_t = torch.from_numpy(df_v).float()[None, None].to(self.device)

        pred_pet_patches = []
        pred_mask_patches = []

        model_in_ch = int(getattr(self.model, "in_ch", 2))

        for z in zs:
            for y in ys:
                for x in xs:
                    ct_patch = ct_t[:, :, z:z + pd, y:y + ph, x:x + pw]
                    df_patch = df_t[:, :, z:z + pd, y:y + ph, x:x + pw]

                    ct_norm = dataset._ct_to_norm(ct_patch)

                    if self.use_prior and model_in_ch >= 2:
                        df_norm = dataset.norm_df(df_patch)
                        src = torch.cat([ct_norm, df_norm], dim=1)
                    else:
                        src = ct_norm

                    if self.enable_mask:
                        pred, mask_logits, _ = self.model.forward_with_mask(src)
                        prob = torch.softmax(mask_logits, dim=1)[:, 1:2]
                        mask_patch = prob.squeeze().detach().cpu().numpy().astype(np.float32)
                    else:
                        pred, _ = self.model(src)
                        mask_patch = np.zeros((pd, ph, pw), dtype=np.float32)

                    pred_patch = ((pred + 1.0) * 0.5 * 20.0)
                    pred_patch = pred_patch.squeeze().detach().cpu().numpy().astype(np.float32)

                    pred_pet_patches.append((z, y, x, pred_patch))
                    pred_mask_patches.append((z, y, x, mask_patch))

        pred_pet = merge_patches(
            vol_shape=(D, H, W),
            patch_size=self.patch_size,
            patch_results=pred_pet_patches,
        )

        pred_mask = merge_patches(
            vol_shape=(D, H, W),
            patch_size=self.patch_size,
            patch_results=pred_mask_patches,
        )

        return pred_pet, pred_mask

    @torch.no_grad()
    def evaluate_patient(self, dataset, pid):
        pred_pet, pred_mask = self.predict_patient(dataset, pid)
        gt_pet = dataset.volumes[pid]["pt"]

        metrics = compute_pet_metrics(
            gt=gt_pet,
            pred=pred_pet,
            suv_thr=self.suv_thr,
            data_range=self.data_range,
            pred_mask=pred_mask if self.enable_mask else None,
        )

        metrics.update(
            self.perceptual.compute(
                gt=gt_pet,
                pred=pred_pet,
                data_range=self.data_range,
            )
        )

        metrics["pid"] = pid
        return metrics

    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataset,
        max_patients: Optional[int] = None,
        shuffle: bool = False,
    ):
        pids = list(dataset.volumes.keys())

        if shuffle:
            random.shuffle(pids)

        if max_patients is not None:
            pids = pids[:int(max_patients)]

        rows = []

        for pid in pids:
            try:
                row = self.evaluate_patient(dataset, pid)
                rows.append(row)
            except Exception as e:
                print(f"[Evaluator] pid={pid} failed: {e}")

        return rows

    @staticmethod
    def summarize(rows):
        if len(rows) == 0:
            return {}

        keys = [
            "SSIM",
            "PSNR",
            "MAE",
            "Dice",
            "HD95",
            "HFEN",
            "GradMAE",
            "LPIPS",
            "GMSD",
        ]

        summary = {}
        for k in keys:
            vals = [float(r[k]) for r in rows if k in r]
            summary[k] = float(np.mean(vals)) if vals else 0.0

        return summary

    @staticmethod
    def save_patient_csv(rows, csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        if len(rows) == 0:
            return

        fieldnames = [
            "pid",
            "SSIM",
            "PSNR",
            "MAE",
            "Dice",
            "HD95",
            "HFEN",
            "GradMAE",
            "LPIPS",
            "GMSD",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    @staticmethod
    def append_summary_csv(
        summary: Dict,
        csv_path: str,
        epoch: int,
        iteration: int,
        split: str,
    ):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        fieldnames = [
            "epoch",
            "iter",
            "split",
            "SSIM",
            "PSNR",
            "MAE",
            "Dice",
            "HD95",
            "HFEN",
            "GradMAE",
            "LPIPS",
            "GMSD",
        ]

        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            row = {
                "epoch": epoch,
                "iter": iteration,
                "split": split,
            }
            row.update(summary)
            writer.writerow(row)