import os
import csv
import random
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import (
    distance_transform_edt,
    rotate,
    zoom,
    gaussian_filter,
    map_coordinates,
)
from torch.utils.data import Dataset


def compute_starts(full: int, patch: int, stride: int) -> List[int]:
    """
    Compute sliding-window start indices.
    Always includes the last valid start position.
    """
    if full <= patch:
        return [0]

    starts = list(range(0, full - patch + 1, stride))
    if starts[-1] != full - patch:
        starts.append(full - patch)
    return starts


def normalize_split_name(split: str) -> str:
    split = str(split).lower()
    if split in ("val", "valid", "validation"):
        return "val"
    if split in ("test", "internal_test"):
        return "test"
    if split == "train":
        return "train"
    raise ValueError(f"Unsupported split: {split}")


def load_volume(path: str, rotate_k: int = 1) -> np.ndarray:
    """
    Load NIfTI volume and return [D, H, W].

    The transpose and rotation follow the original project convention.
    """
    img = nib.load(path)
    arr = img.get_fdata().astype(np.float32)      # [H, W, D]
    arr = np.transpose(arr, (2, 0, 1))            # [D, H, W]

    if rotate_k != 0:
        arr = np.rot90(arr, k=rotate_k, axes=(1, 2)).copy()

    return arr


def compute_distance_field(
    mask_arr: np.ndarray,
    max_dist_vox: float = 64.0,
) -> np.ndarray:
    """
    Compute signed distance field from an esophagus mask.

    Args:
        mask_arr: [D, H, W], foreground > 0.5.
        max_dist_vox: clipping distance in voxels.

    Returns:
        Signed distance field normalized to approximately [-1, 1].
        Inside the mask is negative; outside is positive.
    """
    mask_bin = (mask_arr > 0.5).astype(np.uint8)

    if mask_bin.sum() == 0:
        return np.ones_like(mask_bin, dtype=np.float32)

    dist_out = distance_transform_edt(1 - mask_bin)
    dist_in = distance_transform_edt(mask_bin)

    sdf = dist_out - dist_in
    sdf = np.clip(sdf, -max_dist_vox, max_dist_vox)
    sdf = sdf / max_dist_vox

    return sdf.astype(np.float32)


def center_crop_hw(
    arr: np.ndarray,
    crop_ratio: float = 1.0,
) -> np.ndarray:
    """
    Center crop H/W dimensions while keeping depth unchanged.
    """
    if crop_ratio >= 1.0:
        return arr

    D, H, W = arr.shape
    side_h = int(round(H * crop_ratio))
    side_w = int(round(W * crop_ratio))

    side_h = max(1, min(side_h, H))
    side_w = max(1, min(side_w, W))

    y0 = (H - side_h) // 2
    x0 = (W - side_w) // 2

    return arr[:, y0:y0 + side_h, x0:x0 + side_w]


def resize_hw(
    arr: np.ndarray,
    hw_target: Tuple[int, int],
    mode: str = "trilinear",
) -> np.ndarray:
    """
    Resize only H/W while keeping depth unchanged.
    """
    D = arr.shape[0]
    new_h, new_w = hw_target

    t = torch.from_numpy(arr).float()[None, None]  # [1, 1, D, H, W]
    t = F.interpolate(
        t,
        size=(D, new_h, new_w),
        mode=mode,
        align_corners=False if mode in ("trilinear", "bilinear") else None,
    )
    return t[0, 0].cpu().numpy().astype(np.float32)


def pad_to_min_shape(
    arr: np.ndarray,
    min_shape: Tuple[int, int, int],
    mode: str = "edge",
) -> np.ndarray:
    """
    Pad [D, H, W] to at least min_shape.
    """
    D, H, W = arr.shape
    min_d, min_h, min_w = min_shape

    pad_d = max(0, min_d - D)
    pad_h = max(0, min_h - H)
    pad_w = max(0, min_w - W)

    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return arr

    pad_width = (
        (pad_d // 2, pad_d - pad_d // 2),
        (pad_h // 2, pad_h - pad_h // 2),
        (pad_w // 2, pad_w - pad_w // 2),
    )
    return np.pad(arr, pad_width, mode=mode).astype(np.float32)


def center_crop_or_pad_to_shape(
    arr: np.ndarray,
    target_shape: Tuple[int, int, int],
    mode: str = "edge",
) -> np.ndarray:
    """
    Center crop or pad a [D, H, W] array to target_shape.
    """
    target_d, target_h, target_w = target_shape
    D, H, W = arr.shape

    # Crop
    z0 = max(0, (D - target_d) // 2)
    y0 = max(0, (H - target_h) // 2)
    x0 = max(0, (W - target_w) // 2)

    arr = arr[
        z0:z0 + min(D, target_d),
        y0:y0 + min(H, target_h),
        x0:x0 + min(W, target_w),
    ]

    # Pad
    D, H, W = arr.shape
    pad_d = max(0, target_d - D)
    pad_h = max(0, target_h - H)
    pad_w = max(0, target_w - W)

    if pad_d or pad_h or pad_w:
        pad_width = (
            (pad_d // 2, pad_d - pad_d // 2),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
        )
        arr = np.pad(arr, pad_width, mode=mode)

    return arr.astype(np.float32)


def scale_volume(
    arr: np.ndarray,
    scale_factor: float,
    target_shape: Tuple[int, int, int],
    order: int = 1,
) -> np.ndarray:
    """
    In-plane scaling with center crop/pad back to target shape.
    Depth is not scaled.
    """
    scaled = zoom(
        arr,
        zoom=(1.0, scale_factor, scale_factor),
        order=order,
        mode="nearest",
    )
    return center_crop_or_pad_to_shape(scaled, target_shape, mode="edge")


def elastic_deform_inplane(
    arr: np.ndarray,
    alpha: float,
    sigma: float,
    order: int = 1,
) -> np.ndarray:
    """
    In-plane elastic deformation applied slice-by-slice with one shared
    displacement field for all slices.
    """
    D, H, W = arr.shape

    dx = gaussian_filter((np.random.rand(H, W) * 2.0 - 1.0), sigma) * alpha
    dy = gaussian_filter((np.random.rand(H, W) * 2.0 - 1.0), sigma) * alpha

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = np.array([yy + dy, xx + dx])

    out = np.empty_like(arr, dtype=np.float32)
    for z in range(D):
        out[z] = map_coordinates(
            arr[z],
            coords,
            order=order,
            mode="nearest",
        )

    return out.astype(np.float32)


def collect_patient_ids(root: str) -> List[str]:
    """
    Collect patient folders that contain CT.nii.gz and PT.nii.gz.
    """
    ids = []
    for pid in sorted(os.listdir(root)):
        pdir = os.path.join(root, pid)
        if not os.path.isdir(pdir):
            continue

        ct_path = os.path.join(pdir, "CT.nii.gz")
        pt_path = os.path.join(pdir, "PT.nii.gz")

        if os.path.exists(ct_path) and os.path.exists(pt_path):
            ids.append(pid)

    if len(ids) == 0:
        raise RuntimeError(f"No valid patient folders found under: {root}")

    return ids


def write_split_csv(
    path: str,
    patient_to_split: Dict[str, str],
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pid", "split"])
        for pid in sorted(patient_to_split.keys()):
            writer.writerow([pid, patient_to_split[pid]])


def read_split_csv(path: str) -> Dict[str, str]:
    patient_to_split = {}

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "pid" not in reader.fieldnames or "split" not in reader.fieldnames:
            raise ValueError(f"Split csv must contain columns: pid, split. Got {reader.fieldnames}")

        for row in reader:
            pid = row["pid"]
            split = normalize_split_name(row["split"])
            patient_to_split[pid] = split

    return patient_to_split


def create_patient_split(
    patient_ids: List[str],
    split_ratios: Tuple[float, float, float] = (0.75, 0.125, 0.125),
    seed: int = 42,
) -> Dict[str, str]:
    """
    Create patient-level train/val/test split.
    """
    train_ratio, val_ratio, test_ratio = split_ratios

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total

    ids = list(patient_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    # Ensure at least one case in val/test when possible.
    if n >= 3:
        n_train = min(max(1, n_train), n - 2)
        n_val = min(max(1, n_val), n - n_train - 1)

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    patient_to_split = {}
    for pid in train_ids:
        patient_to_split[pid] = "train"
    for pid in val_ids:
        patient_to_split[pid] = "val"
    for pid in test_ids:
        patient_to_split[pid] = "test"

    return patient_to_split


def get_or_create_split(
    root: str,
    split_csv: str,
    split_ratios: Tuple[float, float, float],
    seed: int,
    overwrite: bool = False,
) -> Dict[str, str]:
    """
    Load existing split csv or create a new train/val/test split.
    """
    if os.path.exists(split_csv) and not overwrite:
        patient_to_split = read_split_csv(split_csv)
        split_values = set(patient_to_split.values())

        if {"train", "val", "test"}.issubset(split_values):
            return patient_to_split

        print(
            f"[WARN] Existing split csv does not contain train/val/test splits: {split_csv}. "
            f"A new split will be created."
        )

    patient_ids = collect_patient_ids(root)
    patient_to_split = create_patient_split(
        patient_ids=patient_ids,
        split_ratios=split_ratios,
        seed=seed,
    )
    write_split_csv(split_csv, patient_to_split)

    print(f"[Split] New patient split saved to: {split_csv}")
    return patient_to_split


class PairedPatches3D(Dataset):
    """
    3D paired CT-to-PET patch dataset with optional esophagus signed distance map.

    Expected data layout:
        root/
          patient_001/
            CT.nii.gz
            PT.nii.gz
          patient_002/
            CT.nii.gz
            PT.nii.gz

        mask_root/
          patient_001/
            esophagus.nii.gz
          patient_002/
            esophagus.nii.gz
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        split_csv: Optional[str] = None,
        split_ratios: Tuple[float, float, float] = (0.75, 0.125, 0.125),
        overwrite_split: bool = False,

        mask_root: Optional[str] = None,
        use_prior: bool = True,
        allow_missing_prior: bool = False,

        patch_size: Tuple[int, int, int] = (32, 96, 96),
        hw_target: Tuple[int, int] = (256, 256),
        stride: Tuple[int, int, int] = (16, 48, 48),
        center_crop_ratio: float = 1.0,

        suv_thr: float = 2.5,
        balance_positive: bool = True,
        pos_ratio: float = 0.7,

        do_aug: bool = True,
        aug_cfg: Optional[Dict] = None,

        return_vis: bool = False,
        seed: int = 42,
    ):
        super().__init__()

        self.root = root
        self.split = normalize_split_name(split)
        self.split_ratios = tuple(split_ratios)
        self.seed = int(seed)

        self.mask_root = mask_root
        self.use_prior = bool(use_prior)
        self.allow_missing_prior = bool(allow_missing_prior)

        self.patch_size = tuple(patch_size)
        self.patch_D, self.patch_H, self.patch_W = self.patch_size

        self.hw_target = tuple(hw_target)
        self.stride = tuple(stride)
        self.stride_D, self.stride_H, self.stride_W = self.stride
        self.center_crop_ratio = float(center_crop_ratio)

        self.suv_thr = float(suv_thr)
        self.balance_positive = bool(balance_positive) and self.split == "train"
        self.pos_ratio = float(pos_ratio)

        self.do_aug = bool(do_aug) and self.split == "train"
        self.aug_cfg = aug_cfg or {}
        self.return_vis = bool(return_vis)

        if split_csv is None:
            split_csv = os.path.join(root, "split_mapnet_3d.csv")
        self.split_csv = split_csv

        patient_to_split = get_or_create_split(
            root=root,
            split_csv=self.split_csv,
            split_ratios=self.split_ratios,
            seed=self.seed,
            overwrite=overwrite_split,
        )

        self.chosen_ids = [
            pid for pid, sp in patient_to_split.items()
            if sp == self.split
        ]

        if len(self.chosen_ids) == 0:
            raise RuntimeError(f"No patients found for split={self.split}.")

        self.volumes: Dict[str, Dict[str, np.ndarray]] = {}
        self.patch_indices: List[Dict] = []
        self.pos_list: List[Dict] = []
        self.neg_list: List[Dict] = []

        self._load_all_patients()

        tag = "+SDM" if self.use_prior else "(CT-only)"
        print(
            f"[PairedPatches3D{tag}] split={self.split} | "
            f"patients={len(self.volumes)} | patches={len(self.patch_indices)} | "
            f"pos={len(self.pos_list)} | neg={len(self.neg_list)}"
        )

        if len(self.patch_indices) == 0:
            raise RuntimeError(f"No patches found for split={self.split}.")

    def _load_all_patients(self):
        for pid in self.chosen_ids:
            ct_path = os.path.join(self.root, pid, "CT.nii.gz")
            pt_path = os.path.join(self.root, pid, "PT.nii.gz")

            if not (os.path.exists(ct_path) and os.path.exists(pt_path)):
                print(f"[WARN] Missing CT/PT for {pid}, skipped.")
                continue

            ct_arr = load_volume(ct_path)
            pt_arr = load_volume(pt_path)

            if ct_arr.shape != pt_arr.shape:
                raise ValueError(
                    f"CT/PT shape mismatch for {pid}: "
                    f"{ct_arr.shape} vs {pt_arr.shape}"
                )

            if self.use_prior:
                if self.mask_root is None:
                    raise ValueError("use_prior=True but mask_root is None.")

                mask_path = os.path.join(self.mask_root, pid, "esophagus.nii.gz")
                if not os.path.exists(mask_path):
                    msg = f"Mask for {pid} not found at {mask_path}."
                    if self.allow_missing_prior:
                        print(f"[WARN] {msg} Use zero SDM.")
                        df_arr = np.zeros_like(ct_arr, dtype=np.float32)
                    else:
                        print(f"[WARN] {msg} Skipped.")
                        continue
                else:
                    mask_arr = load_volume(mask_path)
                    if mask_arr.shape != ct_arr.shape:
                        raise ValueError(
                            f"CT/MASK shape mismatch for {pid}: "
                            f"{ct_arr.shape} vs {mask_arr.shape}"
                        )
                    df_arr = compute_distance_field(mask_arr)
            else:
                df_arr = np.zeros_like(ct_arr, dtype=np.float32)

            ct_arr = center_crop_hw(ct_arr, self.center_crop_ratio)
            pt_arr = center_crop_hw(pt_arr, self.center_crop_ratio)
            df_arr = center_crop_hw(df_arr, self.center_crop_ratio)

            ct_arr = resize_hw(ct_arr, self.hw_target, mode="trilinear")
            pt_arr = resize_hw(pt_arr, self.hw_target, mode="trilinear")
            df_arr = resize_hw(df_arr, self.hw_target, mode="trilinear")

            ct_arr = pad_to_min_shape(ct_arr, self.patch_size, mode="edge")
            pt_arr = pad_to_min_shape(pt_arr, self.patch_size, mode="edge")
            df_arr = pad_to_min_shape(df_arr, self.patch_size, mode="edge")

            self.volumes[pid] = {
                "ct": ct_arr.astype(np.float32),
                "pt": pt_arr.astype(np.float32),
                "df": df_arr.astype(np.float32),
            }

            self._index_patches_for_patient(pid)

    def _index_patches_for_patient(self, pid: str):
        pt = self.volumes[pid]["pt"]
        D, H, W = pt.shape

        zs = compute_starts(D, self.patch_D, self.stride_D)
        ys = compute_starts(H, self.patch_H, self.stride_H)
        xs = compute_starts(W, self.patch_W, self.stride_W)

        for z0 in zs:
            for y0 in ys:
                for x0 in xs:
                    patch_pt = pt[
                        z0:z0 + self.patch_D,
                        y0:y0 + self.patch_H,
                        x0:x0 + self.patch_W,
                    ]

                    is_pos = bool((patch_pt > self.suv_thr).any())

                    entry = {
                        "pid": pid,
                        "z": z0,
                        "y": y0,
                        "x": x0,
                        "is_pos": is_pos,
                    }

                    self.patch_indices.append(entry)
                    if is_pos:
                        self.pos_list.append(entry)
                    else:
                        self.neg_list.append(entry)

    @staticmethod
    def norm_ct(x: torch.Tensor) -> torch.Tensor:
        """
        CT HU [-160, 240] -> [-1, 1].
        """
        x = torch.clamp(x, -160.0, 240.0)
        return (x + 160.0) / 400.0 * 2.0 - 1.0

    @staticmethod
    def norm_pt(x: torch.Tensor) -> torch.Tensor:
        """
        PET SUV [0, 20] -> [-1, 1].
        """
        x = torch.clamp(x, 0.0, 20.0)
        return x / 20.0 * 2.0 - 1.0

    @staticmethod
    def norm_df(x: torch.Tensor) -> torch.Tensor:
        """
        SDM is already approximately in [-1, 1].
        """
        return torch.clamp(x, -1.0, 1.0)

    # Compatibility with old evaluation code.
    def _ct_to_norm(self, x):
        return self.norm_ct(x)

    def _pt_to_norm(self, x):
        return self.norm_pt(x)
    
    def _compute_starts(self, full, patch, stride):
        return compute_starts(full, patch, stride)

    def _sample_entry(self, idx: int):
        if not self.balance_positive:
            return self.patch_indices[idx]

        use_pos = random.random() < self.pos_ratio

        if use_pos and len(self.pos_list) > 0:
            return random.choice(self.pos_list)

        if len(self.neg_list) > 0:
            return random.choice(self.neg_list)

        return random.choice(self.pos_list)

    def _extract_patch(self, arr: np.ndarray, ent: Dict) -> np.ndarray:
        z0, y0, x0 = ent["z"], ent["y"], ent["x"]
        return arr[
            z0:z0 + self.patch_D,
            y0:y0 + self.patch_H,
            x0:x0 + self.patch_W,
        ].astype(np.float32)

    def _apply_geometric_aug(
        self,
        patch_ct: np.ndarray,
        patch_pt: np.ndarray,
        patch_df: np.ndarray,
    ):
        cfg = self.aug_cfg
        target_shape = patch_ct.shape

        # Left-right flip, W-axis.
        if random.random() < float(cfg.get("flip_lr_prob", 0.5)):
            patch_ct = np.flip(patch_ct, axis=2).copy()
            patch_pt = np.flip(patch_pt, axis=2).copy()
            patch_df = np.flip(patch_df, axis=2).copy()

        # Optional anterior-posterior flip, disabled by default.
        if random.random() < float(cfg.get("flip_ap_prob", 0.0)):
            patch_ct = np.flip(patch_ct, axis=1).copy()
            patch_pt = np.flip(patch_pt, axis=1).copy()
            patch_df = np.flip(patch_df, axis=1).copy()

        # In-plane rotation.
        if random.random() < float(cfg.get("rotation_prob", 0.3)):
            max_angle = float(cfg.get("rotation_degrees", 10.0))
            angle = random.uniform(-max_angle, max_angle)

            patch_ct = rotate(patch_ct, angle, axes=(1, 2), reshape=False, order=1, mode="nearest")
            patch_pt = rotate(patch_pt, angle, axes=(1, 2), reshape=False, order=1, mode="nearest")
            patch_df = rotate(patch_df, angle, axes=(1, 2), reshape=False, order=1, mode="nearest")

        # In-plane scaling.
        if random.random() < float(cfg.get("scale_prob", 0.3)):
            scale_min, scale_max = cfg.get("scale_range", [0.9, 1.1])
            scale_factor = random.uniform(float(scale_min), float(scale_max))

            patch_ct = scale_volume(patch_ct, scale_factor, target_shape, order=1)
            patch_pt = scale_volume(patch_pt, scale_factor, target_shape, order=1)
            patch_df = scale_volume(patch_df, scale_factor, target_shape, order=1)

        # In-plane elastic deformation.
        if random.random() < float(cfg.get("elastic_prob", 0.1)):
            alpha = float(cfg.get("elastic_alpha", 4.0))
            sigma = float(cfg.get("elastic_sigma", 8.0))

            patch_ct = elastic_deform_inplane(patch_ct, alpha=alpha, sigma=sigma, order=1)
            patch_pt = elastic_deform_inplane(patch_pt, alpha=alpha, sigma=sigma, order=1)
            patch_df = elastic_deform_inplane(patch_df, alpha=alpha, sigma=sigma, order=1)

        return patch_ct.astype(np.float32), patch_pt.astype(np.float32), patch_df.astype(np.float32)

    def _apply_intensity_aug(self, ct_norm: torch.Tensor) -> torch.Tensor:
        """
        Apply CT intensity augmentation only to the input CT channel.
        PET target is not intensity-augmented.
        """
        cfg = self.aug_cfg

        if random.random() < float(cfg.get("gaussian_noise_prob", 0.15)):
            std = float(cfg.get("gaussian_noise_std", 0.02))
            ct_norm = ct_norm + torch.randn_like(ct_norm) * std

        if random.random() < float(cfg.get("gamma_prob", 0.15)):
            gamma_min, gamma_max = cfg.get("gamma_range", [0.8, 1.2])
            gamma = random.uniform(float(gamma_min), float(gamma_max))

            ct01 = torch.clamp((ct_norm + 1.0) * 0.5, 0.0, 1.0)
            ct01 = torch.pow(ct01, gamma)
            ct_norm = ct01 * 2.0 - 1.0

        return torch.clamp(ct_norm, -1.0, 1.0)

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):
        ent = self._sample_entry(idx)
        pid = ent["pid"]

        volume = self.volumes[pid]

        patch_ct = self._extract_patch(volume["ct"], ent)
        patch_pt = self._extract_patch(volume["pt"], ent)
        patch_df = self._extract_patch(volume["df"], ent)

        if self.do_aug:
            patch_ct, patch_pt, patch_df = self._apply_geometric_aug(
                patch_ct,
                patch_pt,
                patch_df,
            )

        patch_ct = torch.from_numpy(patch_ct).unsqueeze(0).float()
        patch_pt = torch.from_numpy(patch_pt).unsqueeze(0).float()
        patch_df = torch.from_numpy(patch_df).unsqueeze(0).float()

        ct_norm = self.norm_ct(patch_ct)
        pt_norm = self.norm_pt(patch_pt)
        df_norm = self.norm_df(patch_df)

        if self.do_aug:
            ct_norm = self._apply_intensity_aug(ct_norm)

        if self.use_prior:
            src = torch.cat([ct_norm, df_norm], dim=0)  # [2, D, H, W]
        else:
            src = ct_norm                               # [1, D, H, W]

        if self.return_vis:
            return src, pt_norm, patch_ct, patch_pt, patch_df

        return src, pt_norm