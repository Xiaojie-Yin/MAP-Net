import torch
import torch.nn.functional as F


def _as_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _soft_threshold(x: torch.Tensor, thr: float, tau: float = 0.5):
    """
    Differentiable approximation of x > thr.
    """
    tau = max(float(tau), 1e-6)
    return torch.sigmoid((x - float(thr)) / tau)


def mtv_loss(
    pred_raw: torch.Tensor,
    gt_raw: torch.Tensor,
    thr=4.0,
    tau: float = 0.5,
    mode: str = "relative",
    weights=None,
    eps: float = 1e-6,
):
    """
    Metabolic tumor volume loss using differentiable SUV-threshold masks.

    Args:
        pred_raw: synthesized PET in SUV domain, [B, 1, D, H, W].
        gt_raw: ground-truth PET in SUV domain, [B, 1, D, H, W].
        thr: scalar threshold or list of thresholds.
        tau: softness of threshold.
        mode: "relative" or "absolute".
        weights: optional weights for multiple thresholds.

    Returns:
        Scalar loss.
    """
    thresholds = _as_list(thr)

    if weights is None:
        weights = [1.0 / len(thresholds)] * len(thresholds)
    else:
        weights = _as_list(weights)
        if len(weights) != len(thresholds):
            raise ValueError("weights must have the same length as thresholds.")
        s = sum(float(w) for w in weights)
        weights = [float(w) / max(s, eps) for w in weights]

    total = pred_raw.new_tensor(0.0)

    for t, w in zip(thresholds, weights):
        pred_mask = _soft_threshold(pred_raw, t, tau)
        gt_mask = _soft_threshold(gt_raw, t, tau)

        pred_vol = pred_mask.flatten(1).sum(dim=1)
        gt_vol = gt_mask.flatten(1).sum(dim=1)

        if mode == "relative":
            loss = torch.abs(pred_vol - gt_vol) / (gt_vol.abs() + eps)
        elif mode == "absolute":
            loss = torch.abs(pred_vol - gt_vol)
        else:
            raise ValueError(f"Unsupported MTV loss mode: {mode}")

        total = total + float(w) * loss.mean()

    return total


def tlg_loss(
    pred_raw: torch.Tensor,
    gt_raw: torch.Tensor,
    thr=4.0,
    tau: float = 0.5,
    mode: str = "relative",
    weights=None,
    eps: float = 1e-6,
):
    """
    Total lesion glycolysis loss.

    TLG is approximated as:
        sum(SUV * soft_mask(SUV > threshold))
    """
    thresholds = _as_list(thr)

    if weights is None:
        weights = [1.0 / len(thresholds)] * len(thresholds)
    else:
        weights = _as_list(weights)
        if len(weights) != len(thresholds):
            raise ValueError("weights must have the same length as thresholds.")
        s = sum(float(w) for w in weights)
        weights = [float(w) / max(s, eps) for w in weights]

    total = pred_raw.new_tensor(0.0)

    for t, w in zip(thresholds, weights):
        pred_mask = _soft_threshold(pred_raw, t, tau)
        gt_mask = _soft_threshold(gt_raw, t, tau)

        pred_tlg = (pred_raw * pred_mask).flatten(1).sum(dim=1)
        gt_tlg = (gt_raw * gt_mask).flatten(1).sum(dim=1)

        if mode == "relative":
            loss = torch.abs(pred_tlg - gt_tlg) / (gt_tlg.abs() + eps)
        elif mode == "absolute":
            loss = torch.abs(pred_tlg - gt_tlg)
        else:
            raise ValueError(f"Unsupported TLG loss mode: {mode}")

        total = total + float(w) * loss.mean()

    return total


def global_sum_loss(
    pred_raw: torch.Tensor,
    gt_raw: torch.Tensor,
    thr: float = 4.0,
    tau: float = 0.5,
    mask: str = "gt",
    normalize: str = "sum_gt",
    eps: float = 1e-6,
):
    """
    Global uptake-sum consistency loss over high-uptake regions.

    Args:
        mask:
            "gt"    : use GT soft threshold mask.
            "pred"  : use prediction soft threshold mask.
            "union" : use soft union of GT and prediction masks.
            "none"  : use all voxels.
        normalize:
            "sum_gt": divide by GT uptake sum.
            "count" : divide by mask volume.
            "none"  : no normalization.
    """
    if mask == "gt":
        m = _soft_threshold(gt_raw, thr, tau)
    elif mask == "pred":
        m = _soft_threshold(pred_raw, thr, tau)
    elif mask == "union":
        m_gt = _soft_threshold(gt_raw, thr, tau)
        m_pr = _soft_threshold(pred_raw, thr, tau)
        m = torch.clamp(m_gt + m_pr, 0.0, 1.0)
    elif mask == "none":
        m = torch.ones_like(gt_raw)
    else:
        raise ValueError(f"Unsupported mask option: {mask}")

    pred_sum = (pred_raw * m).flatten(1).sum(dim=1)
    gt_sum = (gt_raw * m).flatten(1).sum(dim=1)

    diff = torch.abs(pred_sum - gt_sum)

    if normalize == "sum_gt":
        denom = gt_sum.abs() + eps
        loss = diff / denom
    elif normalize == "count":
        denom = m.flatten(1).sum(dim=1) + eps
        loss = diff / denom
    elif normalize == "none":
        loss = diff
    else:
        raise ValueError(f"Unsupported normalize option: {normalize}")

    return loss.mean()


def hotspot_sparsity_loss(
    pred_raw: torch.Tensor,
    thr: float = 2.5,
    tau: float = 0.5,
    reduce: str = "mean",
):
    """
    Penalize excessive predicted high-uptake volume.

    This is usually used with a small weight.
    """
    m = _soft_threshold(pred_raw, thr, tau)

    if reduce in ("mean", "mean_all"):
        return m.mean()

    if reduce == "sum_per_sample":
        return m.flatten(1).sum(dim=1).mean()

    if reduce == "mean_over_mask":
        # Kept for compatibility with previous config names.
        return m.mean()

    raise ValueError(f"Unsupported reduce option: {reduce}")


def gradient_loss(
    pred_raw: torch.Tensor,
    gt_raw: torch.Tensor,
):
    """
    3D gradient L1 loss in SUV domain.
    Encourages local uptake transitions and boundaries to be preserved.
    """

    def gradients(x):
        dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        return dz, dy, dx

    pz, py, px = gradients(pred_raw)
    gz, gy, gx = gradients(gt_raw)

    loss = (
        F.l1_loss(pz, gz) +
        F.l1_loss(py, gy) +
        F.l1_loss(px, gx)
    ) / 3.0

    return loss