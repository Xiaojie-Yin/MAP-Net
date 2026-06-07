# MAP-Net: Metabolism-Preserving CT-to-PET Synthesis for Esophageal Cancer

Official PyTorch implementation of **MAP-Net**, a metabolism-preserving multi-task framework for 3D CT-to-PET synthesis in esophageal cancer.

MAP-Net aims to synthesize FDG PET from CT while preserving tumor-relevant metabolic uptake patterns. The framework integrates high-uptake region supervision, esophagus-aware structural guidance, Metabolic-Guided Cross-Attention (MGCA), and a Frequency-aware Dual Discriminator (FDD).

---

## Highlights

- **3D CT-to-PET synthesis** with patch-based training and sliding-window whole-volume inference.
- **Multi-task generator** with a shared encoder, PET synthesis decoder, and high-uptake mask decoder.
- **Esophagus-aware structural prior** using a signed distance map (SDM) derived from esophagus segmentation.
- **Metabolic-Guided Cross-Attention (MGCA)** to transfer uptake-aware features from the mask branch to the PET synthesis branch.
- **Frequency-aware Dual Discriminator (FDD)** combining high-frequency spatial discrimination and spectral-domain discrimination.
- **Whole-volume inference** with overlap averaging.
- **Evaluation metrics** including PSNR, SSIM, MAE, Dice, HD95, HFEN, GradMAE, and optional LPIPS/GMSD.

---

## Repository Structure

```text
MAP-Net/
├── train.py
├── test.py
├── inference.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── configs/
│   └── mapnet_3d.yaml
│
├── datasets/
│   ├── __init__.py
│   ├── paired_patches_3d.py
│   └── build.py
│
├── models/
│   ├── __init__.py
│   ├── mapnet.py
│   ├── generator.py
│   ├── mgca.py
│   ├── discriminators.py
│   ├── metabolic_losses.py
│   └── build.py
│
├── engine/
│   ├── __init__.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── checkpoint.py
│
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── seed.py
│   ├── image_utils.py
│   ├── visualization.py
│   ├── logger.py
│   └── metrics.py
│
└── scripts/
    ├── check_model.py
    ├── check_dataset.py
    ├── check_utils.py
    ├── check_evaluator.py
    └── check_trainer.py
```

---

## Method Overview

MAP-Net consists of the following modules.

### 1. Multi-task 3D Generator

The generator uses a shared encoder and two task-specific decoders:

- PET synthesis decoder
- High-uptake region mask decoder

The auxiliary high-uptake branch guides the model to focus on metabolically active regions rather than only optimizing global image similarity.

### 2. Esophagus-aware Structural Prior

An esophagus signed distance map (SDM) is used as an additional input channel. The default input is:

```text
Input channels = [CT, Esophagus SDM]
```

The SDM is computed from an esophagus mask. Inside the esophagus mask has negative distance values, and outside has positive distance values. The SDM is clipped and normalized to approximately `[-1, 1]`.

### 3. Metabolic-Guided Cross-Attention

MGCA transfers uptake-aware features from the high-uptake mask decoder to the PET decoder at two decoder scales.

In the released implementation:

```text
Query: PET decoder feature
Key/Value: high-uptake mask decoder feature
```

The feature injection strength is controlled by:

```yaml
model:
  mgca_alpha: 0.2
```

### 4. Frequency-aware Dual Discriminator

FDD contains two branches:

- High-frequency spatial branch
- Spectral branch based on FFT amplitude descriptors

The final discriminator score is:

```text
score = alpha * spatial_score + beta * spectral_score
```

Default setting:

```yaml
discriminator:
  type: fdd
  r0_ratio: 0.4
  alpha: 1.0
  beta: 1.0
```

---

## Installation

Create a new environment:

```bash
conda create -n mapnet python=3.10 -y
conda activate mapnet
```

Install PyTorch according to your CUDA version. Follow the official PyTorch installation instructions for your system.

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` is provided:

```text
numpy>=1.23
scipy>=1.9
PyYAML>=6.0
nibabel>=5.0
scikit-image>=0.20
torch>=2.0
torchvision>=0.15
lpips>=0.1.4
piq>=0.8.0
tqdm>=4.64
```

`lpips` and `piq` are only required when computing optional perceptual metrics.

---

## Data Preparation

### Expected Data Structure

The preprocessed CT/PET data should be organized as:

```text
data_root/
├── patient_001/
│   ├── CT.nii.gz
│   └── PT.nii.gz
├── patient_002/
│   ├── CT.nii.gz
│   └── PT.nii.gz
└── ...
```

If the esophagus-aware structural prior is used, the esophagus mask directory should be organized as:

```text
mask_root/
├── patient_001/
│   └── esophagus.nii.gz
├── patient_002/
│   └── esophagus.nii.gz
└── ...
```

The patient folder names must match between `data_root` and `mask_root`.

### Image Convention

The dataset loader expects NIfTI files:

```text
CT.nii.gz
PT.nii.gz
esophagus.nii.gz
```

The loader reads NIfTI files using `nibabel`, transposes the array to `[D, H, W]`, and follows the rotation convention used in the original project.

### Intensity Normalization

CT is windowed and normalized as:

```text
HU [-160, 240] -> [-1, 1]
```

PET is clipped and normalized as:

```text
SUV [0, 20] -> [-1, 1]
```

The model predicts normalized PET. During inference and evaluation, predictions are converted back to SUV domain:

```text
[-1, 1] -> SUV [0, 20]
```

---

## Dataset Split

By default, patient-level splitting is:

```text
train / validation / test = 75% / 12.5% / 12.5%
```

The split file is controlled by:

```yaml
data:
  split_csv: "/path/to/split_mapnet_3d.csv"
```

If the split file does not exist, it will be generated automatically. To force regeneration:

```bash
PYTHONPATH=. python train.py \
  --config configs/mapnet_3d.yaml \
  --overwrite-split \
  --debug
```

The split CSV contains:

```text
pid,split
patient_001,train
patient_002,val
patient_003,test
```

---

## Configuration

The default configuration file is:

```text
configs/mapnet_3d.yaml
```

Important settings:

```yaml
dataset:
  split_ratios: [0.75, 0.125, 0.125]
  patch_size: [32, 96, 96]
  stride: [16, 48, 48]
  balance_positive: true
  pos_ratio: 0.7
  suv_thr: 2.5

data:
  use_prior: true

model:
  name: MAPNet3D
  in_ch: 2
  out_ch: 1
  base_ch: 32
  enable_mask: true
  enable_mgca: true
  mgca_window: [4, 4, 4]
  mgca_heads: 2
  mgca_alpha: 0.2

discriminator:
  type: fdd
  base_ch: 64
  r0_ratio: 0.4
  alpha: 1.0
  beta: 1.0

loss:
  lambda_rec: 1.0
  lambda_adv: 0.1

mask:
  threshold:
    suv: 2.5
  loss:
    lambda_dice: 0.25
    lambda_ce_all: 0.25
    lambda_consistency: 0.2

train:
  batch_size: 8
  epochs: 200
  lr: 2.0e-4
  lr_d: 2.0e-4
  beta1: 0.5
  beta2: 0.999
```

---

## Data Augmentation

Online data augmentation is applied during training.

Current implementation includes:

- Random left-right flipping
- Optional anterior-posterior flipping
- In-plane rotation
- In-plane scaling
- Elastic deformation
- Gaussian noise on CT
- Gamma correction on CT

Geometric transformations are applied synchronously to CT, PET, and SDM to preserve spatial alignment.

Example configuration:

```yaml
augmentation:
  enable: true

  flip_lr_prob: 0.5
  flip_ap_prob: 0.0

  rotation_prob: 0.3
  rotation_degrees: 10.0

  scale_prob: 0.3
  scale_range: [0.9, 1.1]

  elastic_prob: 0.1
  elastic_alpha: 4.0
  elastic_sigma: 8.0

  gaussian_noise_prob: 0.15
  gaussian_noise_std: 0.02

  gamma_prob: 0.15
  gamma_range: [0.8, 1.2]
```

---

## Quick Checks

Several smoke-test scripts are provided to verify that the main modules work correctly.

### Check Model

```bash
PYTHONPATH=. python scripts/check_model.py
```

This checks:

- model construction
- forward pass
- PET output shape
- mask output shape
- generator loss
- discriminator loss
- backward pass
- MGCA existence
- FDD spectral branch existence

### Check Dataset

```bash
PYTHONPATH=. python scripts/check_dataset.py
```

This checks:

- train / validation / test split
- CT + SDM input loading
- PET target loading
- patch extraction
- DataLoader output shape

Expected batch shape:

```text
src: [B, 2, 32, 96, 96]
pet: [B, 1, 32, 96, 96]
```

### Check Utilities

```bash
PYTHONPATH=. python scripts/check_utils.py
```

### Check Evaluator

```bash
PYTHONPATH=. python scripts/check_evaluator.py
```

This checks whole-volume sliding-window prediction and metric computation.

### Check Trainer

```bash
PYTHONPATH=. python scripts/check_trainer.py
```

This runs a short training smoke test.

---

## Training

Run training:

```bash
PYTHONPATH=. python train.py \
  --config configs/mapnet_3d.yaml \
  --gpu 0 \
  --output-dir outputs/mapnet_3d
```

Run a short debug training:

```bash
PYTHONPATH=. python train.py \
  --config configs/mapnet_3d.yaml \
  --gpu 0 \
  --debug \
  --batch-size 2 \
  --output-dir outputs/debug_train
```

Resume from a checkpoint:

```bash
PYTHONPATH=. python train.py \
  --config configs/mapnet_3d.yaml \
  --gpu 0 \
  --resume outputs/mapnet_3d/ckpt/best.pth \
  --output-dir outputs/mapnet_3d
```

Training outputs are saved to:

```text
outputs/mapnet_3d/
├── ckpt/
│   ├── best.pth
│   └── epoch*.pth
├── logs/
│   └── train_log.csv
├── vis/
│   └── train_e*_i*.png
├── config_source.yaml
└── config_resolved.yaml
```

The best checkpoint is selected according to the lowest validation L1 reconstruction loss.

---

## Evaluation

Evaluate a trained checkpoint:

```bash
PYTHONPATH=. python test.py \
  --config configs/mapnet_3d.yaml \
  --checkpoint outputs/mapnet_3d/ckpt/best.pth \
  --split test \
  --gpu 0 \
  --output-dir outputs/mapnet_3d
```

Evaluate only a small number of patients for debugging:

```bash
PYTHONPATH=. python test.py \
  --config configs/mapnet_3d.yaml \
  --checkpoint outputs/mapnet_3d/ckpt/best.pth \
  --split val \
  --gpu 0 \
  --output-dir outputs/debug_test \
  --max-patients 1
```

Compute optional perceptual metrics:

```bash
PYTHONPATH=. python test.py \
  --config configs/mapnet_3d.yaml \
  --checkpoint outputs/mapnet_3d/ckpt/best.pth \
  --split test \
  --gpu 0 \
  --output-dir outputs/mapnet_3d \
  --compute-perceptual
```

Evaluation results are saved as:

```text
outputs/mapnet_3d/eval/
├── test_patient_metrics.csv
└── test_summary.csv
```

Metrics include:

```text
SSIM
PSNR
MAE
Dice
HD95
HFEN
GradMAE
LPIPS
GMSD
```

`LPIPS` and `GMSD` are reported as `0.0` unless `--compute-perceptual` is used.

---

## Inference

Run whole-volume PET synthesis:

```bash
PYTHONPATH=. python inference.py \
  --config configs/mapnet_3d.yaml \
  --checkpoint outputs/mapnet_3d/ckpt/best.pth \
  --split test \
  --gpu 0 \
  --output-dir outputs/mapnet_3d/inference/test \
  --save-mask
```

Outputs are saved as NumPy arrays:

```text
outputs/mapnet_3d/inference/test/
├── manifest.csv
└── patient_001/
    ├── sPET_SUV.npy
    └── high_uptake_mask_prob.npy
```

The synthesized PET is saved in SUV domain.

---

## Whole-volume Sliding-window Prediction

During inference, the model predicts 3D patches using the same patch size and stride as training:

```yaml
inference:
  patch_size: [32, 96, 96]
  stride: [16, 48, 48]
  overlap_mode: average
```

Overlapping predictions are averaged to obtain the final synthesized PET volume.

---

## Running Without Structural Prior

To run CT-only experiments, set:

```yaml
data:
  use_prior: false

dataset:
  use_prior: false

model:
  in_ch: 1
```

The dataset will return CT only as the model input.

---

## Running Without MGCA

For MGCA ablation:

```yaml
model:
  enable_mgca: false
```

---

## Running Without Mask Branch

For a PET-only synthesis baseline:

```yaml
model:
  enable_mask: false

mask:
  enable: false
```

When the mask branch is disabled, MGCA is also disabled automatically.

---

## Running With Standard PatchGAN

For discriminator ablation:

```yaml
discriminator:
  type: patchgan
```

For full FDD:

```yaml
discriminator:
  type: fdd
```

---

## Checkpoint Format

Checkpoints are saved as dictionaries containing:

```text
epoch
model
optimizer_g
optimizer_d
scheduler_g
best_val
config
```

Example:

```python
checkpoint = torch.load("outputs/mapnet_3d/ckpt/best.pth", map_location="cpu")
model_state = checkpoint["model"]
```

---

## Important Notes

### 1. Clinical Data Are Not Included

The clinical imaging data used in the study are not released in this repository due to institutional and ethical restrictions.

### 2. Patient Identifiers

Do not upload real patient identifiers, split files containing patient IDs, raw NIfTI files, or model outputs to a public repository.

### 3. Old Checkpoints

Old checkpoints trained with previous experimental code may use different module names, such as `VQGAN3D` or `unet.*`. These checkpoints may require key conversion before being loaded into the cleaned MAP-Net implementation.

### 4. Torchvision Warning

Some environments may show a warning similar to:

```text
Failed to load image Python extension
```

If the code only uses `torchvision.utils.save_image` and `make_grid`, this warning can usually be ignored.

---

## Data Availability

The clinical CT and PET data used in this study cannot be publicly released due to institutional and ethical restrictions. The source code is provided to support methodological reproducibility. Researchers may run the pipeline on their own preprocessed CT/PET data following the directory structure described above.

---

## Model Weights

Trained model weights will be made available upon publication, subject to institutional policy.

If weights are not included, users may train MAP-Net using their own paired CT/PET data.

---

## Citation

If you use this code, please cite:

```bibtex
@article{yin2026mapnet,
  title={MAP-Net: A Metabolism-Preserving Multi-Task Framework with Anatomy-Guided Structural Priors for CT-to-PET Synthesis in Esophageal Cancer},
  author={Yin, Xiaojie and others},
  journal={},
  year={2026}
}
```

---

## License

This repository is released for research use. Please refer to the license file for details.

---

## Contact

For questions about the implementation, please open an issue on GitHub.
