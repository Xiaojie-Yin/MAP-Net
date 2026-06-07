import warnings

import numpy as np
import torch
from scipy.ndimage import gaussian_laplace
from scipy.spatial import cKDTree
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def dice_3d(pred, gt, eps=1e-6):
    """
    Binary 3D Dice.
    pred, gt: numpy arrays or bool arrays.
    """
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)

    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()

    return float((2.0 * inter + eps) / (denom + eps))


def hd95(pred, gt):
    """
    Symmetric 95th percentile Hausdorff distance.

    Returns:
        0.0 if both masks are empty.
        999.0 if only one mask is empty.
    """
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)

    pred_pts = np.argwhere(pred)
    gt_pts = np.argwhere(gt)

    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0.0

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 999.0

    tree_gt = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)

    d_pred_to_gt, _ = tree_gt.query(pred_pts, k=1)
    d_gt_to_pred, _ = tree_pred.query(gt_pts, k=1)

    distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(distances, 95))


def hfen_2d(gt, pred, sigma=1.5):
    """
    High-Frequency Error Norm using LoG-filtered 2D images.
    gt, pred should usually be normalized to [0, 1].
    """
    gt_log = gaussian_laplace(gt.astype(np.float32), sigma=sigma)
    pred_log = gaussian_laplace(pred.astype(np.float32), sigma=sigma)
    diff = gt_log - pred_log
    return float(np.sqrt(np.mean(diff * diff)))


def gradient_magnitude_2d(x):
    gx, gy = np.gradient(x.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy)


def grad_mae_2d(gt, pred):
    gt_grad = gradient_magnitude_2d(gt)
    pred_grad = gradient_magnitude_2d(pred)
    return float(np.mean(np.abs(gt_grad - pred_grad)))


def compute_pet_metrics(
    gt,
    pred,
    suv_thr=2.5,
    data_range=20.0,
    pred_mask=None,
):
    """
    Compute basic PET synthesis and high-uptake mask metrics.

    Args:
        gt: ground-truth PET in SUV domain, [D, H, W].
        pred: synthesized PET in SUV domain, [D, H, W].
        suv_thr: threshold for high-uptake mask.
        pred_mask: optional predicted auxiliary mask, [D, H, W].
                   If None, use pred > suv_thr.

    Returns:
        dict with SSIM, PSNR, MAE, Dice, HD95, HFEN, GradMAE.
    """
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)

    D = gt.shape[0]

    psnr_value = float(peak_signal_noise_ratio(gt, pred, data_range=data_range))

    ssim_values = []
    for d in range(D):
        try:
            ssim_values.append(
                structural_similarity(
                    gt[d],
                    pred[d],
                    data_range=data_range,
                )
            )
        except Exception:
            pass

    ssim_value = float(np.mean(ssim_values)) if ssim_values else 0.0
    mae_value = float(np.mean(np.abs(gt - pred)))

    gt_mask = gt > suv_thr
    if pred_mask is None:
        pred_mask = pred > suv_thr
    else:
        pred_mask = np.asarray(pred_mask) > 0.5

    dice_value = dice_3d(pred_mask, gt_mask)
    hd95_value = hd95(pred_mask, gt_mask)

    hfen_values = []
    grad_values = []

    for d in range(D):
        g = np.clip(gt[d] / data_range, 0.0, 1.0)
        p = np.clip(pred[d] / data_range, 0.0, 1.0)

        hfen_values.append(hfen_2d(g, p))
        grad_values.append(grad_mae_2d(g, p))

    return {
        "SSIM": ssim_value,
        "PSNR": psnr_value,
        "MAE": mae_value,
        "Dice": dice_value,
        "HD95": hd95_value,
        "HFEN": float(np.mean(hfen_values)) if hfen_values else 0.0,
        "GradMAE": float(np.mean(grad_values)) if grad_values else 0.0,
    }


class PerceptualMetricComputer:
    """
    Optional LPIPS and GMSD computer.

    For fast validation during training, keep enabled=False.
    For final evaluation, set enabled=True if lpips and piq are installed.
    """

    def __init__(self, device, enabled=False):
        self.device = device
        self.enabled = bool(enabled)
        self.lpips_fn = None
        self.piq = None

        if not self.enabled:
            return

        try:
            import lpips
            import piq

            self.lpips_fn = lpips.LPIPS(net="alex").to(device)
            self.lpips_fn.eval()
            self.piq = piq
        except Exception as e:
            warnings.warn(
                f"LPIPS/PIQ is unavailable. Perceptual metrics will be disabled. Error: {e}"
            )
            self.enabled = False

    @torch.no_grad()
    def compute(self, gt, pred, data_range=20.0):
        """
        Slice-wise LPIPS and GMSD.

        gt, pred: numpy arrays in SUV domain, [D, H, W].
        """
        if not self.enabled:
            return {
                "LPIPS": 0.0,
                "GMSD": 0.0,
            }

        gt = np.asarray(gt, dtype=np.float32)
        pred = np.asarray(pred, dtype=np.float32)

        lpips_values = []
        gmsd_values = []

        for d in range(gt.shape[0]):
            g01 = np.clip(gt[d] / data_range, 0.0, 1.0).astype(np.float32)
            p01 = np.clip(pred[d] / data_range, 0.0, 1.0).astype(np.float32)

            # LPIPS convention: usually [-1, 1].
            g_lp = torch.from_numpy(g01 * 2.0 - 1.0).to(self.device)
            p_lp = torch.from_numpy(p01 * 2.0 - 1.0).to(self.device)

            g_lp = g_lp.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
            p_lp = p_lp.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

            lp = self.lpips_fn(g_lp, p_lp)
            lpips_values.append(float(lp.item()))

            # GMSD convention: [0, 1].
            g_gm = torch.from_numpy(g01).to(self.device).unsqueeze(0).unsqueeze(0)
            p_gm = torch.from_numpy(p01).to(self.device).unsqueeze(0).unsqueeze(0)

            gm = self.piq.gmsd(g_gm, p_gm, data_range=1.0)
            gmsd_values.append(float(gm.item()))

        return {
            "LPIPS": float(np.mean(lpips_values)) if lpips_values else 0.0,
            "GMSD": float(np.mean(gmsd_values)) if gmsd_values else 0.0,
        }