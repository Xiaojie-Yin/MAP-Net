import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class PatchDiscriminator3D(nn.Module):
    """
    Standard 3D PatchGAN discriminator.
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 64, n_layers: int = 3):
        super().__init__()

        layers = [
            nn.Conv3d(in_ch, base_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        ch = base_ch
        for _ in range(1, n_layers):
            layers += [
                nn.Conv3d(ch, ch * 2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm3d(ch * 2, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch *= 2

        layers.append(nn.Conv3d(ch, 1, kernel_size=3, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def gaussian_kernel_3d(
    kernel_size: int = 3,
    sigma: float = 0.8,
    channels: int = 1,
):
    """
    Construct a 3D Gaussian kernel for grouped convolution.
    """
    ax = torch.arange(kernel_size) - kernel_size // 2
    zz, yy, xx = torch.meshgrid(ax, ax, ax, indexing="ij")

    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1, 1)
    return kernel


class FDD3D(nn.Module):
    """
    Frequency-aware Dual Discriminator.

    It contains:
        1. high-frequency spatial branch:
           Gaussian high-pass residual -> 3D PatchGAN
        2. spectral branch:
           FFT amplitude descriptor -> MLP

    The final score is:
        alpha * spatial_score + beta * spectral_score
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 64,
        r0_ratio: float = 0.25,
        alpha: float = 1.0,
        beta: float = 1.0,
        spectral_bins: int = 128,
        kernel_size: int = 3,
        sigma: float = 0.8,
    ):
        super().__init__()

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.r0_ratio = float(r0_ratio)
        self.spectral_bins = int(spectral_bins)

        self.spatial_d = PatchDiscriminator3D(
            in_ch=in_ch,
            base_ch=base_ch,
            n_layers=3,
        )

        self.spectral_d = nn.Sequential(
            nn.Linear(self.spectral_bins, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)

        kernel = gaussian_kernel_3d(
            kernel_size=self.kernel_size,
            sigma=self.sigma,
            channels=in_ch,
        )
        self.register_buffer("gaussian_kernel", kernel)

    def high_pass_3d(self, x):
        """
        Soft high-pass filtering:
            high = x - GaussianBlur3D(x)
        """
        padding = self.kernel_size // 2

        low = F.conv3d(
            x,
            self.gaussian_kernel,
            padding=padding,
            groups=x.shape[1],
        )
        high = x - low
        return high

    def spectrum_descriptor(self, x):
        """
        Extract a compact high-frequency spectral descriptor.

        Args:
            x: [B, C, D, H, W]

        Returns:
            [B, spectral_bins]
        """
        B = x.shape[0]

        freq = torch.fft.fftn(x, dim=(-3, -2, -1))
        amp = torch.abs(freq).view(B, -1)

        amp_sorted, _ = torch.sort(amp, dim=1)

        cutoff = int(amp_sorted.shape[1] * self.r0_ratio)
        cutoff = max(0, min(cutoff, amp_sorted.shape[1] - 1))

        hf = amp_sorted[:, cutoff:]
        hf = F.adaptive_avg_pool1d(
            hf.unsqueeze(1),
            self.spectral_bins,
        ).squeeze(1)

        return hf

    def forward(self, x):
        high = self.high_pass_3d(x)
        spatial_score = self.spatial_d(high)

        spectral_feat = self.spectrum_descriptor(x)
        spectral_score = self.spectral_d(spectral_feat)

        spatial_global = spatial_score.mean(dim=(2, 3, 4))
        final_score = self.alpha * spatial_global + self.beta * spectral_score

        return final_score


def build_discriminator(
    disc_type: str = "fdd",
    in_ch: int = 1,
    base_ch: int = 64,
    r0_ratio: float = 0.25,
    alpha: float = 1.0,
    beta: float = 1.0,
):
    disc_type = str(disc_type).lower()

    if disc_type in ("fdd", "hfss", "frequency", "frequency_aware"):
        return FDD3D(
            in_ch=in_ch,
            base_ch=base_ch,
            r0_ratio=r0_ratio,
            alpha=alpha,
            beta=beta,
        )

    if disc_type in ("patchgan", "patch", "standard"):
        return PatchDiscriminator3D(
            in_ch=in_ch,
            base_ch=base_ch,
            n_layers=3,
        )

    raise ValueError(f"Unsupported discriminator type: {disc_type}")