import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_num_groups(channels: int, max_groups: int = 8) -> int:
    """Return the largest group number <= max_groups that divides channels."""
    for g in range(max_groups, 0, -1):
        if channels % g == 0:
            return g
    return 1


def window_partition_3d(x: torch.Tensor, window_size):
    """
    Partition a 3D feature map into non-overlapping local windows.

    Args:
        x: Tensor of shape [B, C, D, H, W].
        window_size: tuple/list of (wd, wh, ww).

    Returns:
        Tensor of shape [num_windows * B, C, wd, wh, ww].
    """
    B, C, D, H, W = x.shape
    wd, wh, ww = window_size

    if D % wd != 0 or H % wh != 0 or W % ww != 0:
        raise ValueError(
            f"Input size {(D, H, W)} must be divisible by window size {window_size}."
        )

    x = x.view(
        B, C,
        D // wd, wd,
        H // wh, wh,
        W // ww, ww,
    )
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    windows = x.view(-1, C, wd, wh, ww)
    return windows


def window_reverse_3d(
    windows: torch.Tensor,
    window_size,
    batch_size: int,
    depth: int,
    height: int,
    width: int,
):
    """
    Reverse local windows back to a 3D feature map.

    Args:
        windows: Tensor of shape [num_windows * B, C, wd, wh, ww].
        window_size: tuple/list of (wd, wh, ww).
        batch_size: original batch size.
        depth, height, width: padded spatial size.

    Returns:
        Tensor of shape [B, C, D, H, W].
    """
    wd, wh, ww = window_size

    x = windows.view(
        batch_size,
        depth // wd,
        height // wh,
        width // ww,
        -1,
        wd,
        wh,
        ww,
    )
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    x = x.view(batch_size, -1, depth, height, width)
    return x


class MGCA3D(nn.Module):
    """
    Metabolic-Guided Cross-Attention module.

    Query is generated from PET-decoder features.
    Key and value are generated from high-uptake mask-decoder features.

    Args:
        dim: feature channel number.
        window_size: local 3D attention window size.
        num_heads: number of attention heads.
        alpha_init: initial value of the learnable residual gate.
    """

    def __init__(
        self,
        dim: int,
        window_size=(4, 4, 4),
        num_heads: int = 2,
        alpha_init: float = 0.05,
        gn_groups: int = 8,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.dim = int(dim)
        self.window_size = tuple(window_size)
        self.num_heads = int(num_heads)

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Conv3d(dim, dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv3d(dim, dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv3d(dim, dim, kernel_size=1, bias=False)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1)

        groups = _valid_num_groups(dim, gn_groups)
        self.norm1 = nn.GroupNorm(groups, dim)
        self.norm2 = nn.GroupNorm(groups, dim)

        self.ffn = nn.Sequential(
            nn.Conv3d(dim, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(dim * 4, dim, kernel_size=1),
        )

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, pet_feat: torch.Tensor, mask_feat: torch.Tensor):
        """
        Args:
            pet_feat:  [B, C, D, H, W], features from PET decoder.
            mask_feat: [B, C, D, H, W], features from high-uptake mask decoder.

        Returns:
            Refined PET feature map with the same shape as pet_feat.
        """
        if pet_feat.shape != mask_feat.shape:
            raise ValueError(
                f"pet_feat and mask_feat must have the same shape, "
                f"got {pet_feat.shape} and {mask_feat.shape}."
            )

        B, C, D, H, W = pet_feat.shape
        wd, wh, ww = self.window_size

        pad_d = (wd - D % wd) % wd
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww

        if pad_d or pad_h or pad_w:
            pet_feat = F.pad(pet_feat, (0, pad_w, 0, pad_h, 0, pad_d))
            mask_feat = F.pad(mask_feat, (0, pad_w, 0, pad_h, 0, pad_d))

        _, _, Dp, Hp, Wp = pet_feat.shape

        pet_w = window_partition_3d(pet_feat, self.window_size)
        mask_w = window_partition_3d(mask_feat, self.window_size)

        Bw = pet_w.shape[0]
        N = wd * wh * ww
        head_dim = C // self.num_heads

        q = self.q_proj(pet_w)
        k = self.k_proj(mask_w)
        v = self.v_proj(mask_w)

        def to_heads(t):
            t = t.view(Bw, C, N).transpose(1, 2).contiguous()
            t = t.view(Bw, N, self.num_heads, head_dim)
            t = t.permute(0, 2, 1, 3).contiguous()
            return t

        q = to_heads(q)
        k = to_heads(k)
        v = to_heads(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.permute(0, 2, 1, 3).contiguous().view(Bw, N, C)
        out = out.transpose(1, 2).contiguous().view(Bw, C, wd, wh, ww)
        out = self.proj(out)

        x = pet_w + self.alpha * out
        x = self.norm1(x)

        x = x + self.ffn(x)
        x = self.norm2(x)

        x = window_reverse_3d(x, self.window_size, B, Dp, Hp, Wp)
        x = x[:, :, :D, :H, :W].contiguous()
        return x