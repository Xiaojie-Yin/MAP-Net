from typing import Any, Tuple

import torch
import torch.nn.functional as F


def to_cpu(x: Any):
    """
    Detach tensor and move to CPU.
    """
    return x.detach().cpu() if torch.is_tensor(x) else x


def window_to01(x: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    """
    Window tensor values to [0, 1].
    """
    x = torch.clamp(x, float(vmin), float(vmax))
    return (x - float(vmin)) / (float(vmax) - float(vmin) + 1e-8)


def ct_from_norm(ct_norm: torch.Tensor) -> torch.Tensor:
    """
    CT normalized from [-1, 1] back to HU window [-160, 240].
    """
    return (ct_norm + 1.0) * 0.5 * 400.0 - 160.0


def pet_from_norm(pet_norm: torch.Tensor) -> torch.Tensor:
    """
    PET normalized from [-1, 1] back to SUV [0, 20].
    """
    return (pet_norm + 1.0) * 0.5 * 20.0


def ct_to_norm(ct_hu: torch.Tensor) -> torch.Tensor:
    """
    CT HU [-160, 240] to normalized [-1, 1].
    """
    ct_hu = torch.clamp(ct_hu, -160.0, 240.0)
    return (ct_hu + 160.0) / 400.0 * 2.0 - 1.0


def pet_to_norm(pet_suv: torch.Tensor) -> torch.Tensor:
    """
    PET SUV [0, 20] to normalized [-1, 1].
    """
    pet_suv = torch.clamp(pet_suv, 0.0, 20.0)
    return pet_suv / 20.0 * 2.0 - 1.0


def maybe_unpack_batch(batch):
    """
    Support:
        (src, pet)
        (src, pet, ct_raw, pet_raw)
        other tuple/list variants

    Returns:
        src, pet, ct_raw, pet_raw
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 4:
            return batch[0], batch[1], batch[2], batch[3]
        if len(batch) >= 2:
            return batch[0], batch[1], None, None

    return batch, None, None, None


def take_middle_slice_3d(vol_5d: torch.Tensor) -> torch.Tensor:
    """
    Extract middle axial slice from [B, C, D, H, W].

    Returns:
        [B, C, H, W]
    """
    if vol_5d.dim() != 5:
        raise ValueError(f"Expected 5D tensor [B, C, D, H, W], got {vol_5d.shape}.")

    mid = vol_5d.shape[2] // 2
    return vol_5d[:, :, mid, :, :]


def ensure_same_hw_2d(*tensors: torch.Tensor, size: Tuple[int, int] = None):
    """
    Resize multiple 2D tensors [B, C, H, W] to the same H/W.
    """
    if len(tensors) == 0:
        return []

    if size is None:
        size = tensors[0].shape[-2:]

    outputs = []
    for t in tensors:
        if t.shape[-2:] != size:
            t = F.interpolate(
                t,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        outputs.append(t)

    return outputs


@torch.no_grad()
def mask_debug_stats(mask_logits: torch.Tensor):
    """
    Compute foreground-mask debug statistics.

    Args:
        mask_logits: [B, 2, D, H, W]

    Returns:
        dict
    """
    if mask_logits is None:
        return {
            "ml_mean": 0.0,
            "ml_max": 0.0,
            "pf_mean": 0.0,
            "pf_max": 0.0,
            "pf_p50": 0.0,
            "pf_p90": 0.0,
            "pf_p99": 0.0,
            "pf_pos>0.5(%)": 0.0,
        }

    fg_logit = mask_logits[:, 1:2]
    prob_fg = torch.softmax(mask_logits, dim=1)[:, 1:2]

    prob_flat = prob_fg.reshape(prob_fg.shape[0], -1)
    pos_frac = (prob_flat > 0.5).float().mean().item() * 100.0

    stats = {
        "ml_mean": float(fg_logit.mean().item()),
        "ml_max": float(fg_logit.max().item()),
        "pf_mean": float(prob_fg.mean().item()),
        "pf_max": float(prob_fg.max().item()),
        "pf_p50": float(torch.quantile(prob_fg, 0.50).item()),
        "pf_p90": float(torch.quantile(prob_fg, 0.90).item()),
        "pf_p99": float(torch.quantile(prob_fg, 0.99).item()),
        "pf_pos>0.5(%)": float(pos_frac),
    }

    return stats


def make_pet_threshold_onehot(
    pet_norm: torch.Tensor,
    suv_thr: float = 2.5,
) -> torch.Tensor:
    """
    Build two-class threshold mask from normalized PET.

    Args:
        pet_norm: PET tensor in [-1, 1], [B, 1, D, H, W].
        suv_thr: SUV threshold.

    Returns:
        one-hot mask [B, 2, D, H, W], channel 0 background, channel 1 foreground.
    """
    pet_raw = pet_from_norm(pet_norm)
    fg = (pet_raw > float(suv_thr)).float()
    bg = 1.0 - fg
    return torch.cat([bg, fg], dim=1)


def tensor_to_float(x, default: float = 0.0) -> float:
    """
    Convert tensor or numeric value to float safely.
    """
    if x is None:
        return float(default)

    if torch.is_tensor(x):
        return float(x.detach().cpu().item())

    return float(x)