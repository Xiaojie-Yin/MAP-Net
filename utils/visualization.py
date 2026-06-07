import os
import random
from typing import Optional

import torch
import torchvision

from .image_utils import (
    ct_from_norm,
    ensure_same_hw_2d,
    maybe_unpack_batch,
    pet_from_norm,
    take_middle_slice_3d,
    to_cpu,
    window_to01,
)


def make_ct_pet_panel(
    ct_mid: torch.Tensor,
    pet_mid: torch.Tensor,
    pred_mid: torch.Tensor,
    nrow: int = 6,
    ct_raw_mid: Optional[torch.Tensor] = None,
    pet_raw_mid: Optional[torch.Tensor] = None,
    ct_window=(-160.0, 240.0),
    pet_window=(0.0, 20.0),
) -> torch.Tensor:
    """
    Make a 3-row panel:
        CT | ground-truth PET | synthesized PET

    Args:
        ct_mid: [B, 1, H, W], normalized CT.
        pet_mid: [B, 1, H, W], normalized PET.
        pred_mid: [B, 1, H, W], normalized synthesized PET.
    """
    if ct_raw_mid is not None:
        ct_abs = to_cpu(ct_raw_mid)
    else:
        ct_abs = to_cpu(ct_from_norm(ct_mid))

    if pet_raw_mid is not None:
        pet_abs = to_cpu(pet_raw_mid)
    else:
        pet_abs = to_cpu(pet_from_norm(pet_mid))

    pred_abs = to_cpu(pet_from_norm(pred_mid))

    ct_disp = window_to01(ct_abs, *ct_window)
    pet_disp = window_to01(pet_abs, *pet_window)
    pred_disp = window_to01(pred_abs, *pet_window)

    ct_disp, pet_disp, pred_disp = ensure_same_hw_2d(
        ct_disp,
        pet_disp,
        pred_disp,
    )

    panel = torch.cat(
        [
            torchvision.utils.make_grid(ct_disp, nrow=nrow, normalize=False),
            torchvision.utils.make_grid(pet_disp, nrow=nrow, normalize=False),
            torchvision.utils.make_grid(pred_disp, nrow=nrow, normalize=False),
        ],
        dim=1,
    )

    return panel


def make_ct_pet_mask_panel(
    ct_mid: torch.Tensor,
    pet_mid: torch.Tensor,
    pred_mid: torch.Tensor,
    real_mask_mid: torch.Tensor,
    pred_mask_mid: torch.Tensor,
    spet_mask_mid: torch.Tensor,
    nrow: int = 6,
    ct_raw_mid: Optional[torch.Tensor] = None,
    pet_raw_mid: Optional[torch.Tensor] = None,
    ct_window=(-160.0, 240.0),
    pet_window=(0.0, 20.0),
) -> torch.Tensor:
    """
    Make a 6-row panel:
        CT | ground-truth PET | synthesized PET | GT mask | predicted mask | sPET-threshold mask
    """
    if ct_raw_mid is not None:
        ct_abs = to_cpu(ct_raw_mid)
    else:
        ct_abs = to_cpu(ct_from_norm(ct_mid))

    if pet_raw_mid is not None:
        pet_abs = to_cpu(pet_raw_mid)
    else:
        pet_abs = to_cpu(pet_from_norm(pet_mid))

    pred_abs = to_cpu(pet_from_norm(pred_mid))

    ct_disp = window_to01(ct_abs, *ct_window)
    pet_disp = window_to01(pet_abs, *pet_window)
    pred_disp = window_to01(pred_abs, *pet_window)

    real_mask_disp = to_cpu(real_mask_mid.clamp(0, 1))
    pred_mask_disp = to_cpu(pred_mask_mid.clamp(0, 1))
    spet_mask_disp = to_cpu(spet_mask_mid.clamp(0, 1))

    (
        ct_disp,
        pet_disp,
        pred_disp,
        real_mask_disp,
        pred_mask_disp,
        spet_mask_disp,
    ) = ensure_same_hw_2d(
        ct_disp,
        pet_disp,
        pred_disp,
        real_mask_disp,
        pred_mask_disp,
        spet_mask_disp,
    )

    panel = torch.cat(
        [
            torchvision.utils.make_grid(ct_disp, nrow=nrow, normalize=False),
            torchvision.utils.make_grid(pet_disp, nrow=nrow, normalize=False),
            torchvision.utils.make_grid(pred_disp, nrow=nrow, normalize=False),
            torchvision.utils.make_grid(real_mask_disp, nrow=nrow, normalize=False),
            torchvision.utils.make_grid(pred_mask_disp, nrow=nrow, normalize=False),
            torchvision.utils.make_grid(spet_mask_disp, nrow=nrow, normalize=False),
        ],
        dim=1,
    )

    return panel


def make_batch_panel(
    src: torch.Tensor,
    pet: torch.Tensor,
    pred: torch.Tensor,
    mask_logits: Optional[torch.Tensor] = None,
    suv_thr: float = 2.5,
    nrow: int = 6,
    enable_mask: bool = True,
) -> torch.Tensor:
    """
    Build visualization panel from a training batch.

    Args:
        src: [B, C, D, H, W], first channel is CT.
        pet: [B, 1, D, H, W], normalized GT PET.
        pred: [B, 1, D, H, W], normalized predicted PET.
        mask_logits: [B, 2, D, H, W] or None.
    """
    k = min(int(nrow), src.shape[0])

    ct_mid = take_middle_slice_3d(src[:k, 0:1])
    pet_mid = take_middle_slice_3d(pet[:k])
    pred_mid = take_middle_slice_3d(pred[:k])

    if enable_mask and mask_logits is not None:
        pred_raw = pet_from_norm(pred[:k])
        gt_raw = pet_from_norm(pet[:k])

        prob = torch.softmax(mask_logits[:k], dim=1)
        pred_mask = (prob[:, 1:2] > 0.5).float()

        real_mask = (gt_raw > float(suv_thr)).float()
        spet_mask = (pred_raw > float(suv_thr)).float()

        real_mask_mid = take_middle_slice_3d(real_mask)
        pred_mask_mid = take_middle_slice_3d(pred_mask)
        spet_mask_mid = take_middle_slice_3d(spet_mask)

        return make_ct_pet_mask_panel(
            ct_mid.cpu(),
            pet_mid.cpu(),
            pred_mid.cpu(),
            real_mask_mid.cpu(),
            pred_mask_mid.cpu(),
            spet_mask_mid.cpu(),
            nrow=k,
        )

    return make_ct_pet_panel(
        ct_mid.cpu(),
        pet_mid.cpu(),
        pred_mid.cpu(),
        nrow=k,
    )


@torch.no_grad()
def make_val_panel(
    model,
    val_dataset,
    device,
    nrow: int = 6,
    enable_mask: bool = True,
    suv_thr: float = 2.5,
) -> Optional[torch.Tensor]:
    """
    Randomly sample patches from validation dataset and build a visualization panel.
    """
    if val_dataset is None or len(val_dataset) == 0:
        return None

    k = min(int(nrow), len(val_dataset))
    indices = random.sample(range(len(val_dataset)), k)

    src_list = []
    pet_list = []

    for idx in indices:
        item = val_dataset[idx]
        src, pet, *_ = maybe_unpack_batch(item)
        src_list.append(src)
        pet_list.append(pet)

    src = torch.stack(src_list, dim=0).to(device)
    pet = torch.stack(pet_list, dim=0).to(device)

    was_training = model.training
    model.eval()

    if enable_mask:
        pred, mask_logits, _ = model.forward_with_mask(src)
    else:
        pred, _ = model(src)
        mask_logits = None

    if was_training:
        model.train()

    panel = make_batch_panel(
        src=src,
        pet=pet,
        pred=pred,
        mask_logits=mask_logits,
        suv_thr=suv_thr,
        nrow=k,
        enable_mask=enable_mask,
    )

    return panel


def concat_panels_vertical(
    panel_a: torch.Tensor,
    panel_b: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Concatenate two image panels vertically, cropping to shared size if needed.
    """
    if panel_b is None:
        return panel_a

    h = min(panel_a.shape[1], panel_b.shape[1])
    w = min(panel_a.shape[2], panel_b.shape[2])

    panel_a = panel_a[:, :h, :w]
    panel_b = panel_b[:, :h, :w]

    return torch.cat([panel_a, panel_b], dim=1)


@torch.no_grad()
def save_train_val_panel(
    model,
    src: torch.Tensor,
    pet: torch.Tensor,
    pred: torch.Tensor,
    val_dataset,
    device,
    save_dir: str,
    epoch: int,
    iteration: int,
    mask_logits: Optional[torch.Tensor] = None,
    suv_thr: float = 2.5,
    nrow: int = 6,
    enable_mask: bool = True,
    filename: Optional[str] = None,
) -> str:
    """
    Save train + validation visualization panel.
    """
    os.makedirs(save_dir, exist_ok=True)

    train_panel = make_batch_panel(
        src=src,
        pet=pet,
        pred=pred,
        mask_logits=mask_logits,
        suv_thr=suv_thr,
        nrow=nrow,
        enable_mask=enable_mask,
    )

    val_panel = make_val_panel(
        model=model,
        val_dataset=val_dataset,
        device=device,
        nrow=nrow,
        enable_mask=enable_mask,
        suv_thr=suv_thr,
    )

    full_panel = concat_panels_vertical(train_panel, val_panel)

    if filename is None:
        filename = f"train_e{epoch:03d}_i{iteration:05d}.png"

    save_path = os.path.join(save_dir, filename)
    torchvision.utils.save_image(full_panel, save_path)

    return save_path