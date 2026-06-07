import torch
import torch.nn as nn
import torch.nn.functional as F

from .mgca import MGCA3D


def _valid_num_groups(channels: int, max_groups: int = 8) -> int:
    for g in range(max_groups, 0, -1):
        if channels % g == 0:
            return g
    return 1


def _match_spatial_size(x: torch.Tensor, target: torch.Tensor):
    """
    Align x to target spatial size.
    Used as a safety fallback for odd input dimensions.
    """
    if x.shape[2:] == target.shape[2:]:
        return x

    return F.interpolate(
        x,
        size=target.shape[2:],
        mode="trilinear",
        align_corners=False,
    )


class ConvGNAct3D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        gn_groups: int = 8,
    ):
        super().__init__()

        groups = _valid_num_groups(out_ch, gn_groups)
        self.net = nn.Sequential(
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DoubleConv3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, gn_groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct3D(in_ch, out_ch, gn_groups=gn_groups),
            ConvGNAct3D(out_ch, out_ch, gn_groups=gn_groups),
        )

    def forward(self, x):
        return self.net(x)


class Down3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, gn_groups: int = 8):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_ch, out_ch, gn_groups=gn_groups)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up3D(nn.Module):
    """
    3D upsampling block with skip connection.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, gn_groups: int = 8):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_ch,
            in_ch,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv3D(
            in_ch + skip_ch,
            out_ch,
            gn_groups=gn_groups,
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = _match_spatial_size(x, skip)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class MAPNetGenerator3D(nn.Module):
    """
    Multi-task 3D generator for CT-to-PET synthesis.

    Structure:
        shared encoder
        PET synthesis decoder
        high-uptake mask decoder
        optional MGCA from mask decoder to PET decoder
    """

    def __init__(
        self,
        in_ch: int = 2,
        base_ch: int = 32,
        out_pet_ch: int = 1,
        out_mask_ch: int = 2,
        enable_mask: bool = True,
        enable_mgca: bool = True,
        mgca_window=(4, 4, 4),
        mgca_heads: int = 2,
        mgca_alpha: float = 0.05,
        gn_groups: int = 8,
        output_activation: str = "none",
    ):
        super().__init__()

        self.enable_mask = bool(enable_mask)
        self.enable_mgca = bool(enable_mgca) and self.enable_mask
        self.output_activation = str(output_activation).lower()

        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4
        c4 = base_ch * 8

        # Shared encoder
        self.inc = DoubleConv3D(in_ch, c1, gn_groups=gn_groups)
        self.down1 = Down3D(c1, c2, gn_groups=gn_groups)
        self.down2 = Down3D(c2, c3, gn_groups=gn_groups)
        self.down3 = Down3D(c3, c4, gn_groups=gn_groups)

        # High-uptake mask decoder
        if self.enable_mask:
            self.up_mask1 = Up3D(c4, c3, c3, gn_groups=gn_groups)
            self.up_mask2 = Up3D(c3, c2, c2, gn_groups=gn_groups)
            self.up_mask3 = Up3D(c2, c1, c1, gn_groups=gn_groups)
            self.out_mask = nn.Conv3d(c1, out_mask_ch, kernel_size=1)

        # PET decoder
        self.up_pet1 = Up3D(c4, c3, c3, gn_groups=gn_groups)
        self.up_pet2 = Up3D(c3, c2, c2, gn_groups=gn_groups)
        self.up_pet3 = Up3D(c2, c1, c1, gn_groups=gn_groups)
        self.out_pet = nn.Conv3d(c1, out_pet_ch, kernel_size=1)

        # MGCA modules: mask features guide PET features
        if self.enable_mgca:
            self.mgca_l1 = MGCA3D(
                dim=c3,
                window_size=mgca_window,
                num_heads=mgca_heads,
                alpha_init=mgca_alpha,
                gn_groups=gn_groups,
            )
            self.mgca_l2 = MGCA3D(
                dim=c2,
                window_size=mgca_window,
                num_heads=max(1, mgca_heads // 2),
                alpha_init=mgca_alpha,
                gn_groups=gn_groups,
            )

    def _apply_output_activation(self, x):
        if self.output_activation == "tanh":
            return torch.tanh(x)
        if self.output_activation in ("none", "identity", ""):
            return x
        raise ValueError(f"Unsupported output_activation: {self.output_activation}")

    def forward(self, x):
        """
        Args:
            x: [B, C, D, H, W], usually CT or CT+SDM.

        Returns:
            pred_pet: [B, 1, D, H, W]
            mask_logits: [B, 2, D, H, W] or None
        """
        # Shared encoder
        x1 = self.inc(x)       # /1
        x2 = self.down1(x1)    # /2
        x3 = self.down2(x2)    # /4
        x4 = self.down3(x3)    # /8

        # Mask decoder first, because its features guide PET decoder
        mask_logits = None
        mask_l1 = None
        mask_l2 = None

        if self.enable_mask:
            m = self.up_mask1(x4, x3)   # /4
            mask_l1 = m

            m = self.up_mask2(m, x2)    # /2
            mask_l2 = m

            m = self.up_mask3(m, x1)    # /1
            mask_logits = self.out_mask(m)

        # PET decoder
        p = self.up_pet1(x4, x3)        # /4
        if self.enable_mgca and mask_l1 is not None:
            p = self.mgca_l1(p, mask_l1)

        p = self.up_pet2(p, x2)         # /2
        if self.enable_mgca and mask_l2 is not None:
            p = self.mgca_l2(p, mask_l2)

        p = self.up_pet3(p, x1)         # /1
        pred_pet = self.out_pet(p)
        pred_pet = self._apply_output_activation(pred_pet)

        return pred_pet, mask_logits