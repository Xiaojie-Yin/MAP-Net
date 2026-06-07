import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator import MAPNetGenerator3D
from .discriminators import build_discriminator
from .metabolic_losses import (
    mtv_loss,
    tlg_loss,
    global_sum_loss,
    hotspot_sparsity_loss,
    gradient_loss,
)


class MAPNet3D(nn.Module):
    """
    MAP-Net for 3D CT-to-PET synthesis.

    Main components:
        1. shared-encoder multi-task generator
        2. high-uptake mask auxiliary branch
        3. MGCA-based mask-to-PET feature guidance
        4. optional frequency-aware dual discriminator
    """

    def __init__(
        self,
        in_ch: int = 2,
        out_ch: int = 1,
        base_ch: int = 64,
        lambda_rec: float = 1.0,
        lambda_adv: float = 0.01,
        enable_mask: bool = True,
        enable_mgca: bool = True,
        mgca_window=(4, 4, 4),
        mgca_heads: int = 2,
        mgca_alpha: float = 0.05,
        output_activation: str = "none",
        discriminator_type: str = "fdd",
        disc_base_ch: int = 64,
        disc_r0_ratio: float = 0.25,
        disc_alpha: float = 1.0,
        disc_beta: float = 1.0,
    ):
        super().__init__()

        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.enable_mask = bool(enable_mask)

        self.generator = MAPNetGenerator3D(
            in_ch=self.in_ch,
            base_ch=base_ch,
            out_pet_ch=out_ch,
            out_mask_ch=2,
            enable_mask=enable_mask,
            enable_mgca=enable_mgca,
            mgca_window=mgca_window,
            mgca_heads=mgca_heads,
            mgca_alpha=mgca_alpha,
            output_activation=output_activation,
        )

        self.discriminator = build_discriminator(
            disc_type=discriminator_type,
            in_ch=out_ch,
            base_ch=disc_base_ch,
            r0_ratio=disc_r0_ratio,
            alpha=disc_alpha,
            beta=disc_beta,
        )

        self.lambda_rec = float(lambda_rec)
        self.lambda_adv = float(lambda_adv)

    def _align_cond(self, cond: torch.Tensor):
        """
        Align input channel number to self.in_ch.

        This keeps the code robust when running CT-only or CT+SDM settings.
        """
        if cond.dim() != 5:
            raise ValueError(f"cond must be 5D [B, C, D, H, W], got {cond.shape}")

        B, C, D, H, W = cond.shape

        if C == self.in_ch:
            return cond

        if C > self.in_ch:
            return cond[:, :self.in_ch, ...]

        pad_c = self.in_ch - C
        pad = torch.zeros(
            (B, pad_c, D, H, W),
            device=cond.device,
            dtype=cond.dtype,
        )
        return torch.cat([cond, pad], dim=1)

    def forward(self, cond):
        """
        PET-only forward.

        Returns:
            pred_pet, aux_loss

        aux_loss is currently a zero tensor, kept only to avoid changing
        the existing training loop too much.
        """
        cond = self._align_cond(cond)
        pred_pet, _ = self.generator(cond)
        aux_loss = torch.zeros((), device=cond.device, dtype=pred_pet.dtype)
        return pred_pet, aux_loss

    def forward_with_mask(self, cond):
        """
        PET + high-uptake mask forward.

        Returns:
            pred_pet, mask_logits, aux_loss
        """
        if not self.enable_mask:
            raise RuntimeError("forward_with_mask() was called while enable_mask=False.")

        cond = self._align_cond(cond)
        pred_pet, mask_logits = self.generator(cond)
        aux_loss = torch.zeros((), device=cond.device, dtype=pred_pet.dtype)
        return pred_pet, mask_logits, aux_loss

    def d_loss(self, real, fake):
        """
        Hinge loss for discriminator.
        """
        real_logit = self.discriminator(real)
        fake_logit = self.discriminator(fake.detach())

        loss = (
            F.relu(1.0 - real_logit).mean() +
            F.relu(1.0 + fake_logit).mean()
        )

        log = {
            "logits_real": real_logit.mean().detach(),
            "logits_fake": fake_logit.mean().detach(),
        }
        return loss, log

    def g_loss(
        self,
        real,
        fake,
        aux_loss=None,
        pred_raw=None,
        gt_raw=None,
        mtv_cfg=None,
        tlg_cfg=None,
        sum_weight: float = 0.0,
        sparse_cfg=None,
        grad_weight: float = 0.0,
        mask_logits=None,
        mask_target_onehot=None,
        mask_spet_onehot=None,
        mask_cfg=None,
        **legacy_kwargs,
    ):
        """
        Generator loss.

        The argument **legacy_kwargs allows old training scripts that pass
        vq_loss=... to keep running. The current MAP-Net does not use vector
        quantization.
        """
        if aux_loss is None:
            aux_loss = legacy_kwargs.get("vq_loss", None)

        if aux_loss is None:
            aux_loss = torch.zeros((), device=real.device, dtype=real.dtype)

        rec = F.l1_loss(fake, real)

        if self.lambda_adv > 0:
            adv = -self.discriminator(fake).mean()
        else:
            adv = torch.zeros((), device=real.device, dtype=real.dtype)

        loss = self.lambda_rec * rec + self.lambda_adv * adv + aux_loss

        l_mtv = torch.zeros((), device=real.device, dtype=real.dtype)
        l_tlg = torch.zeros((), device=real.device, dtype=real.dtype)
        l_sum = torch.zeros((), device=real.device, dtype=real.dtype)
        l_sp = torch.zeros((), device=real.device, dtype=real.dtype)
        l_gr = torch.zeros((), device=real.device, dtype=real.dtype)

        if pred_raw is not None and gt_raw is not None:
            current_thr = 4.0
            current_tau = 0.5

            if mtv_cfg is not None and float(mtv_cfg.get("weight", 0.0)) > 0:
                current_thr = mtv_cfg.get("thr", 4.0)
                current_tau = float(mtv_cfg.get("tau", 0.5))

                l_mtv = mtv_loss(
                    pred_raw,
                    gt_raw,
                    thr=current_thr,
                    tau=current_tau,
                    mode=mtv_cfg.get("mode", "relative"),
                    weights=mtv_cfg.get("weights", None),
                )
                loss = loss + float(mtv_cfg.get("weight", 1.0)) * l_mtv

            if tlg_cfg is not None and float(tlg_cfg.get("weight", 0.0)) > 0:
                l_tlg = tlg_loss(
                    pred_raw,
                    gt_raw,
                    thr=tlg_cfg.get("thr", current_thr),
                    tau=float(tlg_cfg.get("tau", current_tau)),
                    mode=tlg_cfg.get("mode", "relative"),
                    weights=tlg_cfg.get("weights", None),
                )
                loss = loss + float(tlg_cfg.get("weight", 1.0)) * l_tlg

            if float(sum_weight) > 0:
                l_sum = global_sum_loss(
                    pred_raw,
                    gt_raw,
                    thr=current_thr if not isinstance(current_thr, list) else current_thr[0],
                    tau=current_tau,
                    mask="gt",
                    normalize="sum_gt",
                )
                loss = loss + float(sum_weight) * l_sum

            if sparse_cfg is not None and float(sparse_cfg.get("weight", 0.0)) > 0:
                l_sp = hotspot_sparsity_loss(
                    pred_raw,
                    thr=float(sparse_cfg.get("thr", 2.5)),
                    tau=float(sparse_cfg.get("tau", 0.5)),
                    reduce=sparse_cfg.get("reduce", "mean"),
                )
                loss = loss + float(sparse_cfg.get("weight", 0.0)) * l_sp

            if float(grad_weight) > 0:
                l_gr = gradient_loss(pred_raw, gt_raw)
                loss = loss + float(grad_weight) * l_gr

        l_mask_dice = torch.zeros((), device=real.device, dtype=real.dtype)
        l_mask_ce = torch.zeros((), device=real.device, dtype=real.dtype)
        l_mask_cons = torch.zeros((), device=real.device, dtype=real.dtype)
        l_mask_sparse = torch.zeros((), device=real.device, dtype=real.dtype)

        if mask_logits is not None and mask_cfg is not None:
            prob = F.softmax(mask_logits, dim=1)
            logprob = F.log_softmax(mask_logits, dim=1)

            kappa = float(mask_cfg.get("kappa", 0.01))
            w_bg = float(mask_cfg.get("w_bg", 0.2))
            w_fg = float(mask_cfg.get("w_fg", 0.8))

            lambda_dice = float(mask_cfg.get("lambda_dice", 1.0))
            lambda_ce = float(mask_cfg.get("lambda_ce_all", 0.2))
            lambda_cons = float(mask_cfg.get("lambda_consistency", 0.2))
            lambda_sparse_mask = float(mask_cfg.get("lambda_sparse", 0.0))

            if mask_target_onehot is not None and lambda_dice > 0:
                dice_val = self._dice_3d(
                    prob,
                    mask_target_onehot,
                    kappa=kappa,
                    w_bg=w_bg,
                    w_fg=w_fg,
                )
                l_mask_dice = 1.0 - dice_val
                loss = loss + lambda_dice * l_mask_dice

            if mask_target_onehot is not None and lambda_ce > 0:
                w_bg_ce = float(mask_cfg.get("w_bg_ce", 0.1))
                w_fg_ce = float(mask_cfg.get("w_fg_ce", 0.9))

                ce_map = -mask_target_onehot * logprob
                weights = torch.zeros_like(mask_target_onehot)
                weights[:, 0] = w_bg_ce
                weights[:, 1] = w_fg_ce

                l_mask_ce = (weights * ce_map).sum() / (weights.sum() + 1e-6)
                loss = loss + lambda_ce * l_mask_ce

            if mask_spet_onehot is not None and lambda_cons > 0:
                dice_val = self._dice_3d(
                    prob,
                    mask_spet_onehot,
                    kappa=kappa,
                    w_bg=w_bg,
                    w_fg=w_fg,
                )
                l_mask_cons = 1.0 - dice_val
                loss = loss + lambda_cons * l_mask_cons

            if lambda_sparse_mask > 0:
                target_fg_ratio = float(mask_cfg.get("target_fg_ratio", 0.01))
                pred_fg_ratio = prob[:, 1:2].mean()
                l_mask_sparse = (pred_fg_ratio - target_fg_ratio) ** 2
                loss = loss + lambda_sparse_mask * l_mask_sparse

        log = {
            "rec_loss": rec.detach(),
            "adv_loss": adv.detach(),
            "aux_loss": aux_loss.detach(),
            "mtv": l_mtv.detach(),
            "tlg": l_tlg.detach(),
            "sum_loss": l_sum.detach(),
            "sparse": l_sp.detach(),
            "grad": l_gr.detach(),
            "mask_dice": l_mask_dice.detach(),
            "mask_ce": l_mask_ce.detach(),
            "mask_cons": l_mask_cons.detach(),
            "mask_sparse": l_mask_sparse.detach(),

            # Temporary compatibility with old train.py.
            # In the final public train.py, replace "quant_loss" with "aux_loss".
            "quant_loss": aux_loss.detach(),
        }

        return loss, log

    @staticmethod
    def _dice_3d(
        prob: torch.Tensor,
        target: torch.Tensor,
        kappa: float = 0.01,
        w_bg: float = 0.2,
        w_fg: float = 0.8,
    ):
        """
        Weighted soft Dice for 3D two-class mask prediction.
        """
        if prob.shape != target.shape:
            raise ValueError(
                f"prob and target must have the same shape, got {prob.shape} and {target.shape}."
            )

        _, _, D, H, W = prob.shape
        smooth = float(kappa) * D * H * W

        inter = (prob * target).sum(dim=(0, 2, 3, 4))
        prob_sum = prob.sum(dim=(0, 2, 3, 4))
        target_sum = target.sum(dim=(0, 2, 3, 4))

        dice_c = (2.0 * inter + smooth) / (prob_sum + target_sum + smooth + 1e-6)

        dice = float(w_bg) * dice_c[0] + float(w_fg) * dice_c[1]
        return dice