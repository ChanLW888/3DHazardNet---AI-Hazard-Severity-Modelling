#!/usr/bin/env python3
"""
V10.1 — 5 cm Dense Latent + Full-Grid Hard-Empty Occupancy + Hazard-Focused Training

Purpose
-------
Train a 3D voxel model that predicts:

1. Binary target occupancy for the four supported RF-DETR semantic classes.
2. A four-class semantic label for occupied target voxels.

There is no semantic background class. A voxel is a positive occupancy
target only when its ground-truth label is one of:

    0 loose_trash
    1 mud_spill
    2 public_greenway
    3 sidewalk

All other voxels, including original background, fallen_fence, and
unobserved/no-MapAnything voxels, are explicit EMPTY occupancy targets and
receive occupancy supervision. Semantic classification remains foreground-only. Occupancy is supervised over the full fixed
target grid.

Inputs
------
Deployment/student:
    RGB + four foreground RF-DETR prior channels

Training:
    RGB + four foreground RF-DETR prior channels

The cached prior files may contain additional channels, but this ablation uses
only the four foreground RF-DETR channels. No LiDAR information is used.
The semantic target dataset is the new 5 cm per-image voxel dataset.

3D target ROI
-------------
X: right,    [-8.0, 8.0] m
Y: down,     [0.0, 2.0] m
Z: forward,  [0.25, 10.0] m

Voxel size: 0.05 m
Target voxel shape: (320, 40, 195)

The semantic voxel NPZs remain on the full source grid
[-8,8] x [0,2] x [0.25,10]. The trainer crops Y=0..1.5 m in memory,
so the model sees a target grid of (320,30,195).

Geometry
--------
- Fixed rectified CAM2/CAM6 camera calibration.
- Letterbox-aware image projection.
- Metric voxel-coordinate encoding.
- Explicit RF-DETR 2D-to-3D semantic lift.
- Multi-scale ResNet-50 image features.
- RGB + RF-DETR is the deployment path.

Training
--------
- Separate binary occupancy and four-class semantic heads.
- Occupancy uses asymmetric hard-negative focal loss.
- Semantic supervision is applied only on target-foreground voxels.
- Rare target classes receive additional sampling/weighting.
- Best checkpoint is selected using foreground hazard mIoU.
"""


import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/hazard_test_101"
)
NPZ_ROOT = DATASET_ROOT / "semantic_voxels_new_roi_5cm_filtered"
OUTPUT_ROOT = DATASET_ROOT / "rgb_voxel_training_v10_1_5cm_xm8_8_y0_15_z025_10_dense_latent_hazard_empty_no_lidar_ablation"
PRIOR_ROOT = DATASET_ROOT / "v5_priors_letterboxed"
# v5_priors is the sample index; only samples with valid RF-DETR priors are used.

NUM_CLASSES = 4
IGNORE_INDEX = -1

# New 5 cm semantic voxel grid stored directly in the NPZ files.
# SOURCE NPZ GRID — stored at full resolution. The trainer crops this in memory.
SOURCE_VOXEL_SHAPE = (320, 40, 195)
SOURCE_BOUNDS = (-8.0, 8.0, 0.0, 2.0, 0.25, 10.0)

# V10 target ROI — cropped from the source grid in memory.
VOXEL_SHAPE = (320, 30, 195)
BOUNDS = (-8.0, 8.0, 0.0, 1.5, 0.25, 10.0)
VOXEL_SIZE = 0.05

IMAGE_SIZE = 384

BATCH_SIZE = 1
NUM_WORKERS = 4
GRAD_ACCUM_STEPS = 4
# Effective batch size is BATCH_SIZE * GRAD_ACCUM_STEPS.

EPOCHS = 50
WARMUP_EPOCHS = 3
LEARNING_RATE = 2e-4
BACKBONE_LR_MULT = 0.08
WEIGHT_DECAY = 1e-4
MIN_LR = 2e-6

LATENT_SHAPE = (40, 8, 20)
LATENT_CHANNELS = 160
DECODER_BASE = 128

USE_PRETRAINED = True
USE_AMP = True

# Loss weights
SEM_CE_WEIGHT = 1.0
SEM_FOCAL_WEIGHT = 0.75
SEM_DICE_WEIGHT = 1.0
OCC_WEIGHT = 1.00
COARSE_WEIGHT = 0.35

# Empty space is a HARD NEGATIVE. We still allow more recall than V10,
# because missing a real mud spill / loose-trash voxel is costly.
# Every voxel in the fixed ROI has an occupancy target: 1=target foreground,
# 0=fully empty. There is no unknown occupancy state.
OCC_FALSE_POSITIVE_WEIGHT = 6.0
OCC_FALSE_NEGATIVE_WEIGHT = 2.0
OCC_FOCAL_GAMMA = 1.0
OCC_HARD_NEGATIVE_START = 0.02
# V10.1: retain both hard-negative and easy-negative gradients so empty space remains strongly supervised.
OCC_HARD_NEGATIVE_MIX = 0.50
OCC_EASY_NEGATIVE_MIX = 0.50

# Rare class emphasis on top of data-derived weights.
# 1 = loose_trash.
RARE_CLASS_BOOST = {
    0: 2.75,   # loose_trash: targeted emphasis
    1: 1.25,   # mud_spill: unchanged
    2: 1.0,    # public_greenway: unchanged
    3: 1.0,    # sidewalk: unchanged
}
FOCAL_GAMMA = 2.0
# Legacy occupancy positive-weight limits retained for configuration
# compatibility; the active V8 objective uses asymmetric focal weighting.
OCC_POS_WEIGHT_MIN = 10.0
OCC_POS_WEIGHT_MAX = 400.0

SEED = 42

# ============================================================
# LOGGING / PERIODIC TEST INFERENCE
# ============================================================
#
# The test set is NEVER used for checkpoint selection or early
# stopping. It is only sampled periodically to monitor how the
# RGB-only model behaves on genuinely held-out data.
#
# Every N epochs, a small fixed set of test images is inferred
# and saved as NPZ files containing semantic + occupancy outputs.
#
TEST_INFERENCE_EVERY = 10
TEST_SAMPLES_PER_CAMERA = 2
TEST_OCCUPANCY_THRESHOLD = 0.50

CLASS_NAMES = {
    0: "loose_trash",
    1: "mud_spill",
    2: "public_greenway",
    3: "sidewalk",
}

# Cached priors contain 6 RF-DETR channels followed by legacy LiDAR channels.
# This ablation uses ONLY the four foreground RF-DETR channels.
CACHED_RFDETR_PRIOR_CHANNELS = 6
RFDETR_PRIOR_CHANNELS = 4
PRIOR_CHANNELS = 6
PRIOR_SIZE = 192

# RF-DETR semantic prior is now an EXPLICIT 2D -> 3D semantic anchor.
# The projected RF-DETR class probabilities are converted to log-priors
# and added directly to the final 3D semantic logits. This prevents the
# network from simply learning to ignore the already-available 2D labels.
SEMANTIC_PRIOR_WEIGHT = 1.00
SEMANTIC_PRIOR_TEMPERATURE = 0.8





# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def letterbox_image(image, size):
    """
    Resize while preserving aspect ratio and pad to size x size.

    Returns:
        tensor [3,size,size]
        scale
        pad_x
        pad_y
        original_width
        original_height
    """
    image = image.convert("RGB")
    ow, oh = image.size

    scale = min(size / ow, size / oh)
    nw = max(1, int(round(ow * scale)))
    nh = max(1, int(round(oh * scale)))

    image = image.resize((nw, nh), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas.paste(image, (pad_x, pad_y))

    arr = np.asarray(canvas).astype(np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32))
    arr = arr / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))

    return (
        torch.from_numpy(arr),
        scale,
        pad_x,
        pad_y,
        ow,
        oh,
    )


def augment_rgb_pil(image):
    """Mild photometric augmentation; geometry is untouched."""
    if random.random() < 0.5:
        image = ImageEnhance.Brightness(
            image
        ).enhance(random.uniform(0.85, 1.15))

    if random.random() < 0.5:
        image = ImageEnhance.Contrast(
            image
        ).enhance(random.uniform(0.85, 1.20))

    if random.random() < 0.35:
        image = ImageEnhance.Color(
            image
        ).enhance(random.uniform(0.85, 1.15))

    return image


# ============================================================
# DATASET DISCOVERY
# ============================================================

def find_samples():
    """
    Build the sample list from cached V5 priors.

    The cached prior files are used only as the RF-DETR source/index.
    Any LiDAR channels present in those files are ignored completely.
    The corresponding semantic voxel target is matched using camera + timestamp.
    """
    samples = []

    if not PRIOR_ROOT.exists():
        raise FileNotFoundError(f"V5 prior root does not exist: {PRIOR_ROOT}")

    if not NPZ_ROOT.exists():
        raise FileNotFoundError(
            f"Semantic voxel root does not exist: {NPZ_ROOT}"
        )

    semantic_total = 0
    prior_files = []

    for camera in ("CAM2", "CAM6"):
        semantic_dir = NPZ_ROOT / camera
        prior_dir = PRIOR_ROOT / camera

        if semantic_dir.exists():
            semantic_total += len(list(semantic_dir.glob("*.npz")))

        if not prior_dir.exists():
            print(f"[WARNING] Missing prior directory: {prior_dir}")
            continue

        prior_files.extend(sorted(prior_dir.glob("*.npz")))

    print("\n" + "=" * 80)
    print("DATASET DISCOVERY")
    print("=" * 80)
    print(f"Semantic voxel samples available: {semantic_total:,}")
    print(f"V5 priors available:             {len(prior_files):,}")

    missing_targets = []

    for prior_path in prior_files:
        camera = prior_path.parent.name
        timestamp = int(prior_path.stem)

        semantic_path = NPZ_ROOT / camera / f"{timestamp}.npz"

        if not semantic_path.exists():
            missing_targets.append((camera, timestamp))
            continue

        try:
            with np.load(semantic_path, allow_pickle=True) as data:
                image_path = data["image_path"]

                if isinstance(image_path, np.ndarray):
                    if image_path.ndim == 0:
                        image_path = image_path.item()
                    else:
                        image_path = image_path.tolist()
                        if isinstance(image_path, list):
                            image_path = image_path[0]

                image_path = Path(str(image_path))

        except Exception as e:
            print(
                f"[WARNING] Failed to read semantic target "
                f"{semantic_path}: {e}"
            )
            continue

        samples.append({
            "npz_path": semantic_path,
            "prior_path": prior_path,
            "camera": camera,
            "timestamp": timestamp,
            "image_path": image_path,
        })

    if missing_targets:
        print(
            f"[WARNING] {len(missing_targets):,} priors have no matching "
            f"semantic target and were skipped."
        )

    print(f"Usable V5 samples:                {len(samples):,}")
    print(f"Skipped priors without targets:   {len(missing_targets):,}")

    for camera in ("CAM2", "CAM6"):
        n = sum(s["camera"] == camera for s in samples)
        print(f"  {camera}: {n:,}")

    print("=" * 80)

    if not samples:
        raise RuntimeError("No usable V5 samples found.")

    return samples
def temporal_split(samples, train_fraction=0.8, val_fraction=0.1):
    train, val, test = [], [], []

    for camera in ("CAM2", "CAM6"):
        camera_samples = [
            s for s in samples if s["camera"] == camera
        ]
        camera_samples.sort(key=lambda x: x["timestamp"])

        n = len(camera_samples)
        n_train = int(n * train_fraction)
        n_val = int(n * val_fraction)

        train.extend(camera_samples[:n_train])
        val.extend(camera_samples[n_train:n_train + n_val])
        test.extend(camera_samples[n_train + n_val:])

    return train, val, test


# ============================================================
# 5 CM TARGET GEOMETRY
# ============================================================

# The semantic voxel NPZs remain on the FULL 320 x 40 x 195 source grid.
# V10 crops the Y axis in memory from 0.0..2.0 m to 0.0..1.5 m.

def validate_5cm_voxel_grid(array=None):
    """Validate the stored FULL 5 cm source voxel geometry."""
    expected_shape = SOURCE_VOXEL_SHAPE
    expected_bounds = SOURCE_BOUNDS

    expected_from_bounds = tuple(
        round((hi - lo) / VOXEL_SIZE)
        for lo, hi in (
            (expected_bounds[0], expected_bounds[1]),
            (expected_bounds[2], expected_bounds[3]),
            (expected_bounds[4], expected_bounds[5]),
        )
    )

    if expected_from_bounds != expected_shape:
        raise RuntimeError(
            f"Source 5 cm geometry mismatch: bounds imply {expected_from_bounds}, "
            f"expected {expected_shape}"
        )

    if array is not None and tuple(array.shape) != expected_shape:
        raise RuntimeError(
            f"Source 5 cm voxel shape {array.shape}; expected {expected_shape}"
        )


def crop_to_target_grid(array):
    """Validate the full source grid, then crop Y=0..1.5 m in memory."""
    validate_5cm_voxel_grid(array)

    # Source Y: 0.0..2.0 m at 5 cm -> 40 slices.
    # Target Y: 0.0..1.5 m at 5 cm -> first 30 slices.
    y_start = round((BOUNDS[2] - SOURCE_BOUNDS[2]) / VOXEL_SIZE)
    y_end = y_start + VOXEL_SHAPE[1]

    if y_start < 0 or y_end > SOURCE_VOXEL_SHAPE[1]:
        raise RuntimeError(
            f"Invalid in-memory Y crop: {y_start}:{y_end} from source "
            f"shape {SOURCE_VOXEL_SHAPE}"
        )

    cropped = array[:, y_start:y_end, :]

    if tuple(cropped.shape) != VOXEL_SHAPE:
        raise RuntimeError(
            f"Cropped voxel shape {cropped.shape}; expected {VOXEL_SHAPE}"
        )

    return cropped


# ============================================================
# RARE-CLASS SAMPLE WEIGHTS
# ============================================================

def compute_sample_weights(samples):
    """
    Weight samples containing rare classes more heavily.

    This is intentionally based on class PRESENCE rather than raw voxel count,
    because raw voxel count would make large sidewalk/greenway regions dominate.
    """
    weights = []

    for sample in samples:
        with np.load(sample["npz_path"], allow_pickle=True) as data:
            semantic = data["semantic"].astype(np.int64)
            semantic = crop_to_target_grid(semantic)
            # Only the four target foreground classes affect sampling.
            semantic = semantic[(semantic >= 2) & (semantic <= 5)] - 2

        present = set(np.unique(semantic).tolist())

        w = 1.0

        if 0 in present:
            w *= 1.0 + 2.5
        if 1 in present:
            w *= 1.0 + 0.5

        weights.append(w)

    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.mean()

    print(
        "[SAMPLER] weight range:",
        f"{weights.min():.3f} -> {weights.max():.3f}",
    )

    return torch.as_tensor(weights, dtype=torch.double)


# ============================================================
# DATASET
# ============================================================

class SemanticVoxelDataset(Dataset):

    def __init__(self, samples, train=False):
        self.samples = samples
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = sample["image_path"]
        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing RGB image:\n{image_path}\n"
                f"Referenced by:\n{sample['npz_path']}"
            )

        image = Image.open(image_path).convert("RGB")

        if self.train:
            image = augment_rgb_pil(image)

        image, scale, pad_x, pad_y, ow, oh = letterbox_image(
            image, IMAGE_SIZE
        )

        with np.load(sample["npz_path"], allow_pickle=True) as data:
            semantic = data["semantic"].astype(np.int64)
            occupancy_observed = data["occupancy"].astype(np.uint8)

            # NPZs remain on the full 320 x 40 x 195 source grid; crop the
            # labels and observed-point mask to the V10 ROI in memory.
            semantic = crop_to_target_grid(semantic)
            occupancy_observed = crop_to_target_grid(occupancy_observed)

            # The 5 cm generator stores:
            #   semantic=-1 : unknown/unobserved
            #   semantic=0  : observed non-hazard
            #   semantic=1..5 : observed RF-DETR classes
            # and an explicit occupancy mask indicating where MapAnything
            # produced a valid 3D point.
            #
            # V10.1 OCCUPANCY DEFINITION:
            #   - foreground semantic voxel -> occupied (1)
            #   - observed non-target/background/fallen-fence -> EMPTY (0)
            #   - no MapAnything point / unobserved -> EMPTY (0)
            #
            # This is intentional: the model is NOT allowed to interpret
            # missing GP/MapAnything coverage as "maybe occupied". The complete
            # fixed ROI is supervised as either target foreground or EMPTY.
            observed = occupancy_observed > 0
            foreground_occupancy = (
                observed & (semantic >= 2) & (semantic <= 5)
            ).astype(np.float32)

            # Semantic classification remains foreground-only. Empty voxels
            # (including unobserved voxels) do not receive a semantic class.
            semantic[(~observed) | (semantic < 2) | (semantic > 5)] = IGNORE_INDEX
            semantic[semantic >= 2] -= 2

            # Occupancy is supervised over the ENTIRE fixed target grid.
            occupancy = foreground_occupancy
            # Occupancy has NO ignore mask. Every voxel is a real target.
            # 1 = foreground target; 0 = fully empty.
            supervision_mask = np.ones_like(occupancy, dtype=np.float32)

        if not sample["prior_path"].exists():
            raise FileNotFoundError(
                f"Missing V5 prior: {sample['prior_path']}\n"
                "Run generate_v5_priors_letterboxed.py first."
            )

        with np.load(sample["prior_path"], allow_pickle=True) as prior_data:
            prior = prior_data["prior"].astype(np.float32)

        if prior.ndim != 3 or prior.shape[0] < CACHED_RFDETR_PRIOR_CHANNELS:
            raise RuntimeError(
                f"Invalid prior shape {prior.shape}; expected at least "
                f"{CACHED_RFDETR_PRIOR_CHANNELS} RF-DETR channels"
            )

        prior = torch.from_numpy(prior)
        # Keep the cached prior at a lower resolution. It is later lifted
        # into the 48x48 RGB FPN feature map, so full 384x384 prior tensors
        # would waste a lot of GPU memory at batch size 64.
        prior = F.interpolate(
            prior.unsqueeze(0),
            size=(PRIOR_SIZE, PRIOR_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if tuple(semantic.shape) != VOXEL_SHAPE:
            raise RuntimeError(
                f"Wrong semantic shape {semantic.shape}; "
                f"expected {VOXEL_SHAPE}"
            )

        if tuple(occupancy.shape) != VOXEL_SHAPE:
            raise RuntimeError(
                f"Wrong occupancy shape {occupancy.shape}; "
                f"expected {VOXEL_SHAPE}"
            )

        if tuple(supervision_mask.shape) != VOXEL_SHAPE:
            raise RuntimeError(
                f"Wrong supervision mask shape {supervision_mask.shape}; "
                f"expected {VOXEL_SHAPE}"
            )

        do_flip = self.train and random.random() < 0.5

        if do_flip:
            # Image and 2D priors must undergo the same horizontal flip.
            image = torch.flip(image, dims=[2])
            prior = torch.flip(prior, dims=[2])

            # X axis is the first voxel axis.
            semantic = np.flip(semantic, axis=0).copy()
            occupancy = np.flip(occupancy, axis=0).copy()
            supervision_mask = np.flip(supervision_mask, axis=0).copy()

        camera_id = 0 if sample["camera"] == "CAM2" else 1

        # Use ONLY the four foreground RF-DETR class probabilities:
        # loose_trash, mud_spill, public_greenway, sidewalk.
        # Background, fallen_fence, and all non-RF channels are removed.
        cached_rf = prior[:CACHED_RFDETR_PRIOR_CHANNELS]
        rf_prior = cached_rf[2:6].contiguous()

        return {
            "image": image,
            "rf_prior": rf_prior,
            "semantic": torch.from_numpy(semantic),
            "occupancy": torch.from_numpy(occupancy),
            "supervision_mask": torch.from_numpy(supervision_mask),
            "camera_id": torch.tensor(camera_id, dtype=torch.long),
            "flip": torch.tensor(do_flip, dtype=torch.bool),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "pad_x": torch.tensor(pad_x, dtype=torch.float32),
            "pad_y": torch.tensor(pad_y, dtype=torch.float32),
            "orig_w": torch.tensor(ow, dtype=torch.float32),
            "orig_h": torch.tensor(oh, dtype=torch.float32),
            "timestamp": sample["timestamp"],
            "camera": sample["camera"],
            "npz_path": str(sample["npz_path"]),
        }


# ============================================================
# IMAGE ENCODER
# ============================================================

class ResNet50MultiScale(nn.Module):
    """
    ResNet-50 C2/C3/C4 multi-scale feature pyramid.

    C2 = stride 8
    C3 = stride 16
    C4 = stride 32
    """

    def __init__(self, pretrained=True, out_channels=LATENT_CHANNELS):
        super().__init__()

        try:
            weights = (
                models.ResNet50_Weights.DEFAULT
                if pretrained
                else None
            )
            backbone = models.resnet50(weights=weights)
            print(
                "[BACKBONE]",
                "ImageNet pretrained ResNet-50"
                if pretrained
                else "random ResNet-50",
            )
        except Exception as e:
            print(f"[WARNING] pretrained load failed: {e}")
            backbone = models.resnet50(weights=None)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.p2 = nn.Conv2d(512, out_channels, 1)
        self.p3 = nn.Conv2d(1024, out_channels, 1)
        self.p4 = nn.Conv2d(2048, out_channels, 1)

        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        p2 = self.p2(c2)
        p3 = self.p3(c3)
        p4 = self.p4(c4)

        target_size = p2.shape[-2:]
        p3 = F.interpolate(
            p3, size=target_size,
            mode="bilinear", align_corners=False
        )
        p4 = F.interpolate(
            p4, size=target_size,
            mode="bilinear", align_corners=False
        )

        fused = self.fuse(torch.cat([p2, p3, p4], dim=1))

        return {
            "p2": p2,
            "p3": p3,
            "p4": p4,
            "fused": fused,
        }


# ============================================================
# 2D PRIOR ENCODER / FUSION
# ============================================================

class PriorEncoder(nn.Module):
    """Encode one prior modality into the RGB FPN feature space."""

    def __init__(self, in_channels, out_channels=LATENT_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.GroupNorm(12, 96),
            nn.GELU(),
            nn.Conv2d(96, out_channels, 1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class RGBPriorFusion(nn.Module):
    """Learn how much to trust the imperfect prior at each pixel."""

    def __init__(self, channels=LATENT_CHANNELS):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(16, channels),
            nn.GELU(),
        )

    def forward(self, rgb, prior):
        gate = self.gate(torch.cat([rgb, prior], dim=1))
        return self.out(torch.cat([rgb, gate * prior], dim=1))


# ============================================================
# METRIC VOXEL GRID
# ============================================================

_GRID_CACHE = {}


def make_metric_voxel_grid(device, dtype):
    key = (
        str(device),
        str(dtype),
        LATENT_SHAPE,
        BOUNDS,
    )

    if key in _GRID_CACHE:
        return _GRID_CACHE[key]

    nx, ny, nz = LATENT_SHAPE
    xmin, xmax, ymin, ymax, zmin, zmax = BOUNDS

    # Sample the full physical ROI at the centres of the latent cells.
    # The latent grid has its own resolution, so its spacing is derived
    # from LATENT_SHAPE rather than the final output VOXEL_SHAPE.
    latent_size_x = (xmax - xmin) / nx
    latent_size_y = (ymax - ymin) / ny
    latent_size_z = (zmax - zmin) / nz

    x = xmin + (torch.arange(nx, device=device, dtype=dtype) + 0.5) * latent_size_x
    y = ymin + (torch.arange(ny, device=device, dtype=dtype) + 0.5) * latent_size_y
    z = zmin + (torch.arange(nz, device=device, dtype=dtype) + 0.5) * latent_size_z

    xx, yy, zz = torch.meshgrid(
        x, y, z, indexing="ij"
    )

    coords = torch.stack([xx, yy, zz], dim=-1)
    coords = coords.reshape(1, -1, 3)

    norm = coords.clone()
    norm[..., 0] = (
        2.0 * (norm[..., 0] - xmin) / (xmax - xmin) - 1.0
    )
    norm[..., 1] = (
        2.0 * (norm[..., 1] - ymin) / (ymax - ymin) - 1.0
    )
    norm[..., 2] = (
        2.0 * (norm[..., 2] - zmin) / (zmax - zmin) - 1.0
    )

    _GRID_CACHE[key] = (coords, norm)
    return coords, norm


# ============================================================
# FOURIER POSITION ENCODING
# ============================================================

class FourierPositionEncoding(nn.Module):

    def __init__(self, out_channels, num_bands=8):
        super().__init__()

        self.num_bands = num_bands

        freqs = 2.0 ** torch.arange(
            num_bands, dtype=torch.float32
        )
        self.register_buffer("freqs", freqs)

        in_dim = 3 * (2 * num_bands) + 3

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, coords_norm):
        # coords_norm: [1,N,3]
        parts = [coords_norm]

        for f in self.freqs:
            parts.append(torch.sin(math.pi * f * coords_norm))
            parts.append(torch.cos(math.pi * f * coords_norm))

        encoded = torch.cat(parts, dim=-1)
        return self.mlp(encoded)


# ============================================================
# FIXED CAMERA CALIBRATION
# ============================================================

# MapAnything's pts3d_cam is already expressed in the camera coordinate
# frame. Therefore no LiDAR-to-camera extrinsic is applied here.
#
# The model consumes images_rect, so the correct projection is the
# rectified P matrix from each camera calibration file. Distortion D is
# deliberately NOT applied: the image has already been rectified.
#
# camera_id 0 = CAM2
# camera_id 1 = CAM6

CALIBRATION_DIR = DATASET_ROOT / "extracted" / "calibration"
CALIBRATION_FILES = {
    0: CALIBRATION_DIR / "CAM2_calib.txt",
    1: CALIBRATION_DIR / "CAM6_calib.txt",
}


def _read_matrix(lines, key, rows, cols):
    """Read a whitespace-separated matrix following a calibration key."""
    for i, line in enumerate(lines):
        if line.strip() == f"{key}:":
            values = []
            for row in lines[i + 1:i + 1 + rows]:
                values.extend(float(v) for v in row.split())
            if len(values) != rows * cols:
                raise RuntimeError(
                    f"Calibration key {key} has {len(values)} values; "
                    f"expected {rows * cols}."
                )
            return np.asarray(values, dtype=np.float32).reshape(rows, cols)
    raise RuntimeError(f"Missing calibration matrix '{key}:'")


def load_camera_calibration(path):
    """
    Load the supplied calibration format:

        width: ...
        height: ...
        K:
        ...
        D:
        ...
        R:
        ...
        P:
        ...

    P is the rectified 3x4 projection matrix. V5 uses P[:, :3]
    because the input is images_rect.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Camera calibration file not found: {path}"
        )

    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    width = None
    height = None

    for line in lines:
        if line.startswith("width:"):
            width = int(line.split(":", 1)[1].strip())
        elif line.startswith("height:"):
            height = int(line.split(":", 1)[1].strip())

    if width is None or height is None:
        raise RuntimeError(
            f"Missing width/height in calibration file: {path}"
        )

    # Read all matrices so the file is validated, but only the rectified
    # P matrix is used for projection.
    K = _read_matrix(lines, "K", 3, 3)
    D = _read_matrix(lines, "D", 1, 4).reshape(4)
    R = _read_matrix(lines, "R", 3, 3)
    P = _read_matrix(lines, "P", 3, 4)

    return {
        "width": width,
        "height": height,
        "K": K,
        "D": D,
        "R": R,
        "P": P,
    }


CAMERA_CALIBRATIONS = {
    camera_id: load_camera_calibration(path)
    for camera_id, path in CALIBRATION_FILES.items()
}


print("[CALIBRATION] Loaded fixed rectified camera calibration:")

for camera_id, calib in CAMERA_CALIBRATIONS.items():
    camera_name = "CAM2" if camera_id == 0 else "CAM6"
    P = calib["P"]
    K_rect = P[:, :3]

    print(
        f"  {camera_name}: {CALIBRATION_FILES[camera_id]} "
        f"({calib['width']}x{calib['height']})"
    )
    print(
        f"    rectified fx={K_rect[0, 0]:.9f}, "
        f"fy={K_rect[1, 1]:.9f}, "
        f"cx={K_rect[0, 2]:.9f}, "
        f"cy={K_rect[1, 2]:.9f}"
    )
    print("    source: P[:, :3]")
    print("    distortion D: ignored because input is images_rect")


# ============================================================
# CAMERA PROJECTION
# ============================================================

class CameraProjection(nn.Module):
    """
    Fixed calibrated projection from MapAnything camera coordinates to
    the letterboxed RF-DETR image.

    Coordinate convention:
        X = right
        Y = down
        Z = forward

    MapAnything gives us pts3d_cam, so the coordinates are already in the
    camera frame. No LiDAR extrinsic is applied here.

    For rectified images, use the first 3x3 part of P:

        u = fx * X/Z + cx
        v = fy * Y/Z + cy

    Then apply the exact same horizontal flip and letterbox transforms
    used by the Dataset.

    All calibration values are fixed buffers, never learned.
    """

    def __init__(self):
        super().__init__()

        K_rect = torch.stack(
            [
                torch.from_numpy(
                    CAMERA_CALIBRATIONS[0]["P"][:, :3]
                ),
                torch.from_numpy(
                    CAMERA_CALIBRATIONS[1]["P"][:, :3]
                ),
            ],
            dim=0,
        ).float()

        self.register_buffer(
            "K_rect",
            K_rect,
            persistent=True,
        )

        image_size = torch.tensor(
            [
                [
                    CAMERA_CALIBRATIONS[0]["width"],
                    CAMERA_CALIBRATIONS[0]["height"],
                ],
                [
                    CAMERA_CALIBRATIONS[1]["width"],
                    CAMERA_CALIBRATIONS[1]["height"],
                ],
            ],
            dtype=torch.float32,
        )

        self.register_buffer(
            "calibration_image_size",
            image_size,
            persistent=True,
        )

    def forward(
        self,
        coords,
        camera_id,
        flip,
        scale,
        pad_x,
        pad_y,
        orig_w,
        orig_h,
    ):
        """
        coords: [B,N,3]
        returns: [B,N,2] normalized grid_sample coordinates.
        """

        if torch.any((camera_id < 0) | (camera_id >= 2)):
            raise RuntimeError(
                "Invalid camera_id values: "
                f"{camera_id.detach().cpu().tolist()}"
            )

        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2].clamp_min(0.5)

        K = self.K_rect[camera_id]

        fx = K[:, 0, 0].view(-1, 1)
        fy = K[:, 1, 1].view(-1, 1)
        cx = K[:, 0, 2].view(-1, 1)
        cy = K[:, 1, 2].view(-1, 1)

        # --------------------------------------------------------
        # 1. Camera coordinates -> original rectified image pixels
        # --------------------------------------------------------

        u_px = fx * x / z + cx
        v_px = fy * y / z + cy

        # --------------------------------------------------------
        # 2. Horizontal flip augmentation
        # --------------------------------------------------------

        flip_f = flip.float().view(-1, 1)

        u_px = torch.where(
            flip_f > 0.5,
            orig_w.view(-1, 1) - 1.0 - u_px,
            u_px,
        )

        # --------------------------------------------------------
        # 3. Letterbox transform
        # --------------------------------------------------------

        u_lb = (
            u_px * scale.view(-1, 1)
            + pad_x.view(-1, 1)
        )

        v_lb = (
            v_px * scale.view(-1, 1)
            + pad_y.view(-1, 1)
        )

        # --------------------------------------------------------
        # 4. Letterboxed pixels -> grid_sample [-1,1]
        # --------------------------------------------------------

        u_grid = (
            2.0 * u_lb / (IMAGE_SIZE - 1)
            - 1.0
        )

        v_grid = (
            2.0 * v_lb / (IMAGE_SIZE - 1)
            - 1.0
        )

        return torch.stack(
            [u_grid, v_grid],
            dim=-1,
        )


# ============================================================
# GEOMETRY-AWARE LIFT
# ============================================================

class GeometryAwareLift(nn.Module):

    def __init__(self, channels=LATENT_CHANNELS):
        super().__init__()

        self.camera_projection = CameraProjection()

        self.position = FourierPositionEncoding(
            channels,
            num_bands=8,
        )

        self.camera_embed = nn.Embedding(2, channels)

        self.p2_proj = nn.Conv2d(channels, channels, 1)
        self.p3_proj = nn.Conv2d(channels, channels, 1)
        self.p4_proj = nn.Conv2d(channels, channels, 1)

        self.sample_fuse = nn.Sequential(
            nn.Conv1d(channels * 3, channels, 1),
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )

        self.gate = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 1),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
            nn.Sigmoid(),
        )

        self.out = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(16, channels),
            nn.GELU(),
        )

    @staticmethod
    def sample(features, grid):
        # features: [B,C,H,W]
        # grid: [B,N,2]
        grid = grid.unsqueeze(1)
        out = F.grid_sample(
            features,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return out.squeeze(2)

    def forward(
        self,
        pyramid,
        coords_metric,
        coords_norm,
        camera_id,
        flip,
        scale,
        pad_x,
        pad_y,
        orig_w,
        orig_h,
    ):
        B = pyramid["p2"].shape[0]

        coords = coords_metric.expand(B, -1, -1)

        grid = self.camera_projection(
            coords,
            camera_id,
            flip,
            scale,
            pad_x,
            pad_y,
            orig_w,
            orig_h,
        )

        p2 = self.p2_proj(pyramid["p2"])
        p3 = self.p3_proj(pyramid["p3"])
        p4 = self.p4_proj(pyramid["p4"])

        s2 = self.sample(p2, grid)
        s3 = self.sample(p3, grid)
        s4 = self.sample(p4, grid)

        image_feat = self.sample_fuse(
            torch.cat([s2, s3, s4], dim=1)
        )

        pos = self.position(coords_norm.expand(B, -1, -1))
        pos = pos.transpose(1, 2)

        cam = self.camera_embed(camera_id).unsqueeze(-1)

        gate = self.gate(
            torch.cat([image_feat, pos], dim=1)
        )

        fused = gate * image_feat + (1.0 - gate) * pos
        fused = fused + cam

        nx, ny, nz = LATENT_SHAPE
        volume = fused.reshape(
            B, -1, nx, ny, nz
        )

        volume = self.out(volume)

        return volume


# ============================================================
# 3D RESIDUAL BLOCKS
# ============================================================

class ResBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, 3, padding=1
        )
        self.norm1 = nn.GroupNorm(
            min(16, out_channels),
            out_channels,
        )

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, 3, padding=1
        )
        self.norm2 = nn.GroupNorm(
            min(16, out_channels),
            out_channels,
        )

        self.skip = (
            nn.Conv3d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.skip(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)

        x = self.conv2(x)
        x = self.norm2(x)

        return F.gelu(x + residual)


class Semantic3DDecoder(nn.Module):

    def __init__(
        self,
        in_channels=LATENT_CHANNELS,
        base=DECODER_BASE,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()

        self.in_block = ResBlock3D(
            in_channels, base
        )

        self.mid1 = ResBlock3D(base, base)
        self.mid2 = ResBlock3D(base, base)

        self.up1 = nn.Conv3d(
            base, base, 3, padding=1
        )
        self.block1 = ResBlock3D(base, 96)

        self.up2 = nn.Conv3d(
            96, 80, 3, padding=1
        )
        self.block2 = ResBlock3D(80, 64)

        self.refine = ResBlock3D(64, 48)

        self.occupancy_head = nn.Conv3d(
            48, 1, 1
        )
        self.semantic_head = nn.Conv3d(
            48, num_classes, 1
        )

        # Coarse auxiliary head for deep supervision.
        self.coarse_head = nn.Conv3d(
            base, num_classes, 1
        )

    def forward(self, x):
        x = self.in_block(x)
        x = self.mid1(x)
        x = self.mid2(x)

        coarse = self.coarse_head(x)

        # V10.1 latent grid is 40 x 8 x 20. Decode progressively so the
        # model does not first collapse the higher-resolution latent volume
        # back down to the old 20 x 4 x 10-derived intermediate grid.
        x = F.interpolate(
            x,
            size=(80, 16, 40),
            mode="trilinear",
            align_corners=False,
        )
        x = self.up1(x)
        x = self.block1(x)

        x = F.interpolate(
            x,
            size=VOXEL_SHAPE,
            mode="trilinear",
            align_corners=False,
        )
        x = self.up2(x)
        x = self.block2(x)

        x = self.refine(x)

        occupancy = self.occupancy_head(x)
        semantic = self.semantic_head(x)

        return occupancy, semantic, coarse


# ============================================================
# EXPLICIT RF-DETR 2D -> 3D SEMANTIC LIFT
# ============================================================

class RFDETRSemanticLift(nn.Module):
    """
    Project RF-DETR's 2D semantic probabilities directly into the latent
    metric voxel grid.

    This is deliberately NOT a generic learned feature fusion block.
    RF-DETR has already solved the 2D semantic recognition problem, so this
    branch preserves that semantic identity explicitly and lets the learned
    network concentrate on the 2D -> 3D correspondence problem.
    """

    def __init__(self, camera_projection):
        super().__init__()
        self.camera_projection = camera_projection

    def forward(
        self,
        rf_prior,
        coords_metric,
        camera_id,
        flip,
        scale,
        pad_x,
        pad_y,
        orig_w,
        orig_h,
    ):
        B = rf_prior.shape[0]
        coords = coords_metric.expand(B, -1, -1)

        # CameraProjection returns coordinates in the 384x384 image frame,
        # but grid_sample uses normalized coordinates, so the same grid can
        # sample the RF prior at any spatial resolution.
        grid = self.camera_projection(
            coords,
            camera_id,
            flip,
            scale,
            pad_x,
            pad_y,
            orig_w,
            orig_h,
        )

        # RF prior: [B,4,H,W] -> [B,4,N]
        sampled = F.grid_sample(
            rf_prior,
            grid.unsqueeze(1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(2)

        # IMPORTANT V10.1 FIX:
        # Do NOT renormalize the four foreground channels and throw away their
        # absolute magnitude. If RF-DETR says [0.001, 0.001, 0.001, 0.001],
        # the old code turned that into [0.25, 0.25, 0.25, 0.25], which can
        # create a fake semantic anchor in pixels where RF-DETR saw nothing.
        #
        # Keep the total foreground mass and only normalize the class mixture.
        sampled = sampled.clamp_min(1e-6)
        foreground_mass = sampled.sum(dim=1, keepdim=True)

        class_distribution = (
            sampled
            / foreground_mass.clamp_min(1e-6)
        )

        class_distribution = F.softmax(
            torch.log(class_distribution) / SEMANTIC_PRIOR_TEMPERATURE,
            dim=1,
        )

        # Re-attach the absolute RF-DETR foreground confidence.
        # Low-confidence/background pixels therefore produce a weak prior
        # instead of an equally strong four-way semantic prediction.
        sampled = class_distribution * foreground_mass.clamp(0.0, 1.0)

        nx, ny, nz = LATENT_SHAPE
        return sampled.reshape(B, RFDETR_PRIOR_CHANNELS, nx, ny, nz)


# ============================================================
# MODEL
# ============================================================

class RGBPriorToSemanticVoxelModelV10(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = ResNet50MultiScale(
            pretrained=USE_PRETRAINED,
            out_channels=LATENT_CHANNELS,
        )

        # RF-DETR is part of the deployment input and is always available.
        self.rf_prior_encoder = PriorEncoder(
            in_channels=RFDETR_PRIOR_CHANNELS,
            out_channels=LATENT_CHANNELS,
        )
        # V7: inject the same calibrated RF-DETR prior at multiple RGB FPN
        # scales. This is especially useful for small hazards whose evidence
        # can disappear after the first downsampling stage.
        self.rf_prior_encoder_p3 = PriorEncoder(
            in_channels=RFDETR_PRIOR_CHANNELS,
            out_channels=LATENT_CHANNELS,
        )
        self.rf_prior_encoder_p4 = PriorEncoder(
            in_channels=RFDETR_PRIOR_CHANNELS,
            out_channels=LATENT_CHANNELS,
        )

        self.rf_prior_fusion = RGBPriorFusion(
            channels=LATENT_CHANNELS,
        )

        self.lift = GeometryAwareLift(
            channels=LATENT_CHANNELS
        )

        # Explicit semantic path: RF-DETR labels are projected into 3D using
        # the same fixed calibrated camera model as the RGB lift.
        self.rf_semantic_lift = RFDETRSemanticLift(
            self.lift.camera_projection
        )

        self.decoder = Semantic3DDecoder(
            in_channels=LATENT_CHANNELS,
            base=DECODER_BASE,
            num_classes=NUM_CLASSES,
        )

    def forward(
        self,
        image,
        rf_prior,
        camera_id,
        flip,
        scale,
        pad_x,
        pad_y,
        orig_w,
        orig_h,
    ):
        pyramid = self.encoder(image)

        # --------------------------------------------------------
        # RF-DETR branch: always available at train/val/test/deploy.
        # --------------------------------------------------------
        rf_prior_p2 = F.interpolate(
            rf_prior,
            size=pyramid["p2"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        rf_prior_p3 = F.interpolate(
            rf_prior,
            size=pyramid["p3"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        rf_prior_p4 = F.interpolate(
            rf_prior,
            size=pyramid["p4"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        rf_feat_p2 = self.rf_prior_encoder(rf_prior_p2)
        rf_feat_p3 = self.rf_prior_encoder_p3(rf_prior_p3)
        rf_feat_p4 = self.rf_prior_encoder_p4(rf_prior_p4)

        pyramid["p2"] = self.rf_prior_fusion(
            pyramid["p2"],
            rf_feat_p2,
        )
        pyramid["p3"] = self.rf_prior_fusion(
            pyramid["p3"],
            rf_feat_p3,
        )
        pyramid["p4"] = self.rf_prior_fusion(
            pyramid["p4"],
            rf_feat_p4,
        )

        coords_metric, coords_norm = make_metric_voxel_grid(
            image.device,
            pyramid["p2"].dtype,
        )

        volume = self.lift(
            pyramid,
            coords_metric,
            coords_norm,
            camera_id,
            flip,
            scale,
            pad_x,
            pad_y,
            orig_w,
            orig_h,
        )

        occ_logits, semantic_logits, coarse_logits = self.decoder(volume)

        # --------------------------------------------------------
        # EXPLICIT RF-DETR SEMANTIC ANCHOR
        # --------------------------------------------------------
        # RF-DETR has already identified the 2D class. Convert its projected
        # class probabilities into a 3D log-prior and add it directly to the
        # learned semantic logits. The decoder therefore learns a residual
        # correction around the RF-DETR semantic prediction rather than
        # having to rediscover the class identity from RGB.
        rf_semantic_volume = self.rf_semantic_lift(
            rf_prior,
            coords_metric,
            camera_id,
            flip,
            scale,
            pad_x,
            pad_y,
            orig_w,
            orig_h,
        )

        rf_semantic_full = F.interpolate(
            rf_semantic_volume,
            size=VOXEL_SHAPE,
            mode="trilinear",
            align_corners=False,
        )

        rf_log_prior = torch.log(
            rf_semantic_full.clamp_min(1e-5)
        )

        semantic_logits = (
            semantic_logits
            + SEMANTIC_PRIOR_WEIGHT * rf_log_prior
        )

        # --------------------------------------------------------
        # V10 FOREGROUND-ONLY OUTPUT
        # --------------------------------------------------------
        # There is no background semantic logit. Occupancy is a separate binary
        # decision that determines whether the four-class semantic prediction
        # should be instantiated at a voxel. At inference, semantic predictions
        # are masked by the predicted occupancy.

        return occ_logits, semantic_logits, coarse_logits


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights(samples):
    counts = np.zeros(
        NUM_CLASSES,
        dtype=np.float64,
    )

    for sample in samples:
        with np.load(
            sample["npz_path"],
            allow_pickle=True,
        ) as data:
            semantic = data["semantic"].astype(np.int64)
            semantic = crop_to_target_grid(semantic)
            # Background and fallen fence are not semantic classes in V10.
            semantic[(semantic < 2) | (semantic > 5)] = IGNORE_INDEX
            semantic[semantic >= 2] -= 2

        valid = semantic != IGNORE_INDEX
        labels = semantic[valid].astype(np.int64)

        counts += np.bincount(
            labels,
            minlength=NUM_CLASSES,
        )

    print("\n[LOSS] Training voxel counts:")
    for class_id in range(NUM_CLASSES):
        print(
            f"  {class_id}: "
            f"{CLASS_NAMES[class_id]:20s} "
            f"{int(counts[class_id]):,}"
        )

    # Square-root inverse frequency is deliberately less aggressive than
    # median/count, which can become unstable for extremely rare classes.
    freq = counts / max(counts.sum(), 1.0)
    nonzero = freq[freq > 0]

    median_freq = np.median(nonzero)
    weights = np.sqrt(
        median_freq / np.maximum(freq, 1e-12)
    )

    weights = np.clip(weights, 0.35, 5.0)

    for class_id, boost in RARE_CLASS_BOOST.items():
        weights[class_id] *= boost

    # All four classes are foreground semantic targets; there is no background
    # class to protect from being downweighted.

    weights /= weights.mean()

    print("\n[LOSS] Final class weights:")
    for class_id, weight in enumerate(weights):
        print(
            f"  {class_id}: "
            f"{CLASS_NAMES[class_id]:20s}"
            f"{weight:.4f}"
        )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================

def compute_occupancy_pos_weight(samples):
    """
    Compute full-grid foreground occupancy statistics from TRAINING data.

    V10:
      1 = one of the four supported foreground semantic classes
      0 = everything else, including unobserved/no-MapAnything voxels.

    The returned positive weight is retained for configuration compatibility;
    the active objective uses asymmetric false-positive/false-negative weights.
    """
    occupied = 0
    unoccupied = 0
    total_voxels = 0
    explicit_empty = 0
    unobserved = 0

    for sample in samples:
        with np.load(sample["npz_path"], allow_pickle=True) as data:
            semantic = crop_to_target_grid(data["semantic"].astype(np.int64))
            observed = crop_to_target_grid(
                data["occupancy"].astype(np.uint8) > 0
            )

            foreground = observed & (semantic >= 2) & (semantic <= 5)

            n_total = int(foreground.size)
            n_occ = int(foreground.sum())
            n_observed = int(observed.sum())

            occupied += n_occ
            unoccupied += n_total - n_occ
            total_voxels += n_total
            explicit_empty += n_observed - n_occ
            unobserved += n_total - n_observed

    if occupied == 0:
        raise RuntimeError("Training set contains zero occupied voxels.")

    raw_weight = unoccupied / float(occupied)
    pos_weight = float(np.clip(
        raw_weight, OCC_POS_WEIGHT_MIN, OCC_POS_WEIGHT_MAX
    ))

    print()
    print("[LOSS] FULL-GRID OCCUPANCY STATISTICS:")
    print(f"  Total target voxels:      {total_voxels:,}")
    print(f"  Occupied foreground:      {occupied:,}")
    print(f"  Explicit observed empty:  {explicit_empty:,}")
    print(f"  Unobserved -> empty:      {unobserved:,}")
    print(f"  Total unoccupied:         {unoccupied:,}")
    print(f"  Occupied fraction:        {occupied / max(total_voxels, 1):.6f}")
    print(f"  Explicit empty fraction:  {explicit_empty / max(total_voxels, 1):.6f}")
    print(f"  Unobserved fraction:      {unobserved / max(total_voxels, 1):.6f}")
    print(f"  Raw neg/pos ratio:        {raw_weight:.2f}")
    print(f"  Legacy neg/pos ratio (diagnostic only): {pos_weight:.2f}")
    print("  NOTE: actual occupancy loss uses FULL-GRID asymmetric FP/FN weighting.")
    print(f"  FP weight: {OCC_FALSE_POSITIVE_WEIGHT:.2f}")
    print(f"  FN weight: {OCC_FALSE_NEGATIVE_WEIGHT:.2f}")

    return torch.tensor(pos_weight, dtype=torch.float32)


# LOSS FUNCTIONS
# ============================================================

def soft_dice_loss(
    logits,
    target,
    class_weights,
):
    """
    Multi-class soft Dice over valid voxels.
    Background is included but naturally downweighted by class_weights.
    """
    valid = target != IGNORE_INDEX

    if valid.sum() == 0:
        return logits.sum() * 0.0

    probs = F.softmax(logits, dim=1)

    safe_target = target.clamp(
        min=0,
        max=NUM_CLASSES - 1,
    )

    one_hot = F.one_hot(
        safe_target,
        num_classes=NUM_CLASSES,
    ).permute(0, 4, 1, 2, 3).float()

    valid_f = valid.unsqueeze(1).float()

    probs = probs * valid_f
    one_hot = one_hot * valid_f

    dims = (0, 2, 3, 4)

    intersection = (probs * one_hot).sum(dims)
    denominator = (
        probs.sum(dims) +
        one_hot.sum(dims)
    )

    dice = (
        2.0 * intersection + 1.0
    ) / (
        denominator + 1.0
    )

    # Classes with no target voxels are ignored.
    present = one_hot.sum(dims) > 0
    dice = dice[present]
    weights = class_weights[present]

    if dice.numel() == 0:
        return logits.sum() * 0.0

    weights = weights / weights.mean()

    return (
        (1.0 - dice) * weights
    ).mean()


def asymmetric_occupancy_loss(occupancy_logits, occupancy_target, supervision_mask=None):
    """
    Hard-negative asymmetric focal occupancy loss over the FULL fixed target grid.

    V10.1 has an explicit occupancy target for every voxel:
        foreground semantic voxel = 1
        everything else            = 0

    There is deliberately NO unknown/unobserved occupancy class.
    Missing GP/MapAnything coverage is trained as fully empty.
    """
    occ_target = (occupancy_target > 0).float()
    logits = occupancy_logits.squeeze(1)
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(
        logits, occ_target, reduction="none"
    )

    # V10 deliberately ignores any legacy supervision mask here.
    positive = occ_target > 0.5
    negative = occ_target <= 0.5

    pos_factor = (1.0 - prob).pow(OCC_FOCAL_GAMMA)
    pos_loss = bce * pos_factor

    neg_factor = prob.pow(OCC_FOCAL_GAMMA)
    neg_loss = bce * neg_factor

    hard_neg = (prob >= OCC_HARD_NEGATIVE_START) & negative
    easy_neg = negative & ~hard_neg

    hard_loss = neg_loss[hard_neg].mean() if hard_neg.any() else logits.sum() * 0.0
    easy_loss = bce[easy_neg].mean() if easy_neg.any() else logits.sum() * 0.0
    fn_loss = pos_loss[positive].mean() if positive.any() else logits.sum() * 0.0

    fp_loss = (
        OCC_HARD_NEGATIVE_MIX * hard_loss
        + OCC_EASY_NEGATIVE_MIX * easy_loss
    )

    return (
        OCC_FALSE_POSITIVE_WEIGHT * fp_loss
        + OCC_FALSE_NEGATIVE_WEIGHT * fn_loss
    ) / (OCC_FALSE_POSITIVE_WEIGHT + OCC_FALSE_NEGATIVE_WEIGHT)


def focal_ce_loss(
    logits,
    target,
    class_weights,
    gamma=2.0,
):
    valid = target != IGNORE_INDEX

    if valid.sum() == 0:
        return logits.sum() * 0.0

    logits_v = logits.permute(
        0, 2, 3, 4, 1
    )[valid]

    target_v = target[valid]

    # Compute the true target log-probability WITHOUT class weighting first.
    # The previous implementation used weighted CE to derive pt, which makes
    # the focal factor depend on the class weight itself.
    log_probs = F.log_softmax(logits_v, dim=1)
    target_log_prob = log_probs.gather(
        1, target_v.unsqueeze(1)
    ).squeeze(1)
    pt = target_log_prob.exp()

    unweighted_ce = -target_log_prob
    alpha = class_weights[target_v]

    focal = (
        alpha * (1.0 - pt).pow(gamma) * unweighted_ce
    )

    return focal.mean()


class MultiTaskVoxelLoss(nn.Module):

    def __init__(
        self,
        class_weights,
        occupancy_pos_weight,
    ):
        super().__init__()

        self.register_buffer(
            "class_weights",
            class_weights,
        )

        self.register_buffer(
            "occupancy_pos_weight",
            occupancy_pos_weight,
        )

    def forward(
        self,
        occupancy_logits,
        semantic_logits,
        coarse_logits,
        occupancy_target,
        semantic_target,
        supervision_mask=None,
    ):

        # ====================================================
        # OCCUPANCY LOSS
        # ====================================================
        #
        # V10 occupancy is supervised over the ENTIRE fixed target grid.
        # Unobserved/no-MapAnything voxels are explicit EMPTY negatives.
        occ_target = (occupancy_target > 0).float()

        # Semantic supervision remains foreground-only.
        semantic_target_fg = semantic_target.clone()

        occ_logits = (
            occupancy_logits.squeeze(1)
        )

        # --------------------------------------------------------
        # ASYMMETRIC OCCUPANCY LOSS
        # --------------------------------------------------------
        # The previous BCE used pos_weight ~= 26.87 because occupancy is
        # sparse. That strongly rewarded recall and produced far too many
        # false occupied voxels (~27% predicted vs ~3-4% GT).
        #
        # Instead, measure the positive and negative populations separately
        # and explicitly make a false-positive occupied prediction more
        # expensive than a missed occupied voxel. This makes the objective
        # directly match the desired conservative geometry behaviour.
        bce = F.binary_cross_entropy_with_logits(
            occ_logits, occ_target, reduction="none"
        )
        positive_mask = occ_target > 0.5
        negative_mask = occ_target <= 0.5
        prob = torch.sigmoid(occ_logits)

        # Diagnostic components remain plain BCE so the CSV continues to
        # show interpretable FP/FN error, while the actual objective uses
        # hard-negative focal weighting.
        fn_loss = bce[positive_mask].mean() if positive_mask.any() else occ_logits.sum() * 0.0
        fp_loss = bce[negative_mask].mean() if negative_mask.any() else occ_logits.sum() * 0.0
        occ_loss = asymmetric_occupancy_loss(
            occupancy_logits,
            occupancy_target,
            supervision_mask=None,
        )

        # ====================================================
        # SEMANTIC LOSSES
        # ====================================================

        semantic_target_fg[occ_target <= 0.5] = IGNORE_INDEX

        semantic_valid = semantic_target_fg != IGNORE_INDEX
        if semantic_valid.any():
            ce_logits = semantic_logits.permute(0, 2, 3, 4, 1)[semantic_valid]
            ce_target = semantic_target_fg[semantic_valid]
            ce_loss = F.cross_entropy(
                ce_logits,
                ce_target,
                weight=self.class_weights,
            )
        else:
            ce_loss = semantic_logits.sum() * 0.0

        focal = focal_ce_loss(
            semantic_logits,
            semantic_target_fg,
            self.class_weights,
            FOCAL_GAMMA,
        )

        dice = soft_dice_loss(
            semantic_logits,
            semantic_target_fg,
            self.class_weights,
        )

        # ====================================================
        # COARSE SEMANTIC DEEP SUPERVISION
        # ====================================================

        coarse_target = F.interpolate(
            semantic_target_fg.unsqueeze(1).float(),
            size=coarse_logits.shape[-3:],
            mode="nearest",
        ).squeeze(1).long()

        # V9: nearest-neighbour downsampling can produce a coarse volume
        # containing only IGNORE_INDEX voxels. F.cross_entropy returns NaN when
        # every target voxel is ignored, which previously poisoned the logged
        # total loss even though the other losses remained finite.
        coarse_valid = coarse_target != IGNORE_INDEX
        if coarse_valid.any():
            coarse_logits_v = coarse_logits.permute(0, 2, 3, 4, 1)[coarse_valid]
            coarse_target_v = coarse_target[coarse_valid]
            coarse_loss = F.cross_entropy(
                coarse_logits_v,
                coarse_target_v,
                weight=self.class_weights,
            )
        else:
            coarse_loss = coarse_logits.sum() * 0.0

        # ====================================================
        # TOTAL
        # ====================================================

        total = (
            OCC_WEIGHT * occ_loss
            + SEM_CE_WEIGHT * ce_loss
            + SEM_FOCAL_WEIGHT * focal
            + SEM_DICE_WEIGHT * dice
            + COARSE_WEIGHT * coarse_loss
        )

        return {
            "total": total,
            "occ": occ_loss.detach(),
            "occ_fp": fp_loss.detach(),
            "occ_fn": fn_loss.detach(),
            "ce": ce_loss.detach(),
            "focal": focal.detach(),
            "dice": dice.detach(),
            "coarse": coarse_loss.detach(),
        }


# METRICS
# ============================================================

@torch.no_grad()
def update_confusion_matrix(cm, logits, target):
    pred = logits.argmax(dim=1)

    valid = target != IGNORE_INDEX

    pred = pred[valid]
    target = target[valid]

    indices = (
        target * NUM_CLASSES + pred
    )

    cm += torch.bincount(
        indices,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(
        NUM_CLASSES,
        NUM_CLASSES,
    )

    return cm


def iou_from_cm(cm):
    cm = cm.float()

    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    denom = tp + fp + fn

    iou = tp / (denom + 1e-8)

    # Classes with no target and no prediction should not influence mIoU.
    support = cm.sum(dim=1)
    iou[support == 0] = float("nan")

    return iou.cpu().numpy()


def summarize_ious(ious):
    valid = ious[~np.isnan(ious)]

    miou = (
        float(valid.mean())
        if len(valid)
        else 0.0
    )

    hazard = ious
    hazard_valid = hazard[~np.isnan(hazard)]

    hazard_miou = (
        float(hazard_valid.mean())
        if len(hazard_valid)
        else 0.0
    )

    small = ious[[0, 1]]
    small_valid = small[~np.isnan(small)]

    small_miou = (
        float(small_valid.mean())
        if len(small_valid)
        else 0.0
    )

    return miou, hazard_miou, small_miou


# ============================================================
# EPOCH
# ============================================================

def move_batch(batch, device):
    return {
        "image": batch["image"].to(
            device, non_blocking=True
        ),
        "rf_prior": batch["rf_prior"].to(
            device, non_blocking=True
        ),
        "semantic": batch["semantic"].to(
            device, non_blocking=True
        ),
        "occupancy": batch["occupancy"].to(
            device, non_blocking=True
        ),
        "supervision_mask": batch["supervision_mask"].to(
            device, non_blocking=True
        ),
        "camera_id": batch["camera_id"].to(
            device, non_blocking=True
        ),
        "flip": batch["flip"].to(
            device, non_blocking=True
        ),
        "scale": batch["scale"].to(
            device, non_blocking=True
        ),
        "pad_x": batch["pad_x"].to(
            device, non_blocking=True
        ),
        "pad_y": batch["pad_y"].to(
            device, non_blocking=True
        ),
        "orig_w": batch["orig_w"].to(
            device, non_blocking=True
        ),
        "orig_h": batch["orig_h"].to(
            device, non_blocking=True
        ),
    }


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train,
    epoch,
    scaler=None,
):
    model.train(train)

    totals = {
        "total": 0.0,
        "occ": 0.0,
        "occ_fp": 0.0,
        "occ_fn": 0.0,
        "ce": 0.0,
        "focal": 0.0,
        "dice": 0.0,
        "coarse": 0.0,
    }

    cm = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.long,
        device=device,
    )

    for batch_idx, batch in enumerate(loader):
        batch = move_batch(batch, device)

        if train and (batch_idx % GRAD_ACCUM_STEPS == 0):
            optimizer.zero_grad(set_to_none=True)

        # RGB + RF-DETR only. This run is the no-LiDAR ablation.
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(
                USE_AMP and device.type == "cuda"
            ),
        ):
            occ_logits, sem_logits, coarse_logits = model(
                batch["image"], batch["rf_prior"], batch["camera_id"],
                batch["flip"], batch["scale"], batch["pad_x"],
                batch["pad_y"], batch["orig_w"], batch["orig_h"],
            )

            losses = criterion(
                occ_logits, sem_logits, coarse_logits,
                batch["occupancy"], batch["semantic"],
                supervision_mask=batch["supervision_mask"],
            )

            loss = losses["total"]

        if train:
            scaled_loss = loss / float(GRAD_ACCUM_STEPS)
            scaled_loss.backward()

            should_step = (
                ((batch_idx + 1) % GRAD_ACCUM_STEPS == 0)
                or (batch_idx + 1 == len(loader))
            )

            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        for key in totals:
            totals[key] += losses[key].item()

        cm = update_confusion_matrix(
            cm,
            sem_logits,
            batch["semantic"],
        )

        if batch_idx % 20 == 0:
            phase = "TRAIN" if train else "VAL"

            print(
                f"[{phase}] "
                f"batch {batch_idx:04d}/{len(loader):04d} "
                f"loss={loss.item():.4f} "
                f"occ={losses['occ'].item():.4f} "
                f"fp={losses['occ_fp'].item():.4f} "
                f"fn={losses['occ_fn'].item():.4f} "
                f"ce={losses['ce'].item():.4f} "
                f"focal={losses['focal'].item():.4f} "
                f"dice={losses['dice'].item():.4f}"
            )

    n = max(len(loader), 1)

    ious = iou_from_cm(cm)
    miou, hazard_miou, small_miou = summarize_ious(
        ious
    )

    result = {
        key: value / n
        for key, value in totals.items()
    }

    result["ious"] = ious
    result["miou"] = miou
    result["hazard_miou"] = hazard_miou
    result["small_hazard_miou"] = small_miou

    return result



# ============================================================
# LOGGING / PERIODIC TEST INFERENCE
# ============================================================

TRAINING_LOG_COLUMNS = [
    "epoch",
    "lr",
    "train_total",
    "train_occ",
    "train_occ_fp",
    "train_occ_fn",
    "train_ce",
    "train_focal",
    "train_dice",
    "train_coarse",
    "train_miou",
    "train_hazard_miou",
    "train_small_hazard_miou",
    "val_total",
    "val_occ",
    "val_occ_fp",
    "val_occ_fn",
    "val_ce",
    "val_focal",
    "val_dice",
    "val_coarse",
    "val_miou",
    "val_hazard_miou",
    "val_small_hazard_miou",
    "val_loose_trash_iou",
    "val_mud_spill_iou",
    "val_public_greenway_iou",
    "val_sidewalk_iou",
    "test_ran",
    "test_hazard_miou",
    "test_small_hazard_miou",
    "test_occupancy_precision",
    "test_occupancy_recall",
    "test_occupancy_f1",
    "test_gt_occupied_fraction",
    "test_pred_occupied_fraction",
    "test_pred_occupancy_probability_mean",
    "test_gt_occupied_fraction_observed",
    "test_pred_occupied_fraction_observed",
    "test_pred_occupancy_probability_mean_observed",
]


def append_training_log(path, row):
    """
    Append one complete epoch to CSV.

    This is deliberately separate from history.json so the file can
    be opened directly in Excel/pandas while training is running.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    exists = path.exists()

    with open(
        path,
        "a",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=TRAINING_LOG_COLUMNS,
            extrasaction="ignore",
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            {
                key: (
                    "" if row.get(key) is None
                    else row.get(key)
                )
                for key in TRAINING_LOG_COLUMNS
            }
        )


def select_test_samples(
    test_samples,
    samples_per_camera=TEST_SAMPLES_PER_CAMERA,
):
    """
    Deterministically select a small, spread-out subset of the test set.

    We choose samples across the temporal extent rather than simply
    taking the first few test frames.
    """

    selected = []

    for camera in ("CAM2", "CAM6"):

        camera_samples = [
            s
            for s in test_samples
            if s["camera"] == camera
        ]

        camera_samples.sort(
            key=lambda x: x["timestamp"]
        )

        if not camera_samples:
            continue

        n = min(
            samples_per_camera,
            len(camera_samples),
        )

        indices = np.linspace(
            0,
            len(camera_samples) - 1,
            n,
            dtype=int,
        )

        selected.extend(
            camera_samples[i]
            for i in sorted(set(indices))
        )

    return selected


def run_test_inference(
    model,
    test_samples,
    device,
    epoch,
    output_root,
):
    """
    Run RGB-only inference on a small held-out test subset.

    This does NOT affect:
      - checkpoint selection
      - early stopping
      - optimizer/scheduler

    It produces:
      test_inference/
        epoch_005/
          CAM2_<timestamp>.npz
          CAM6_<timestamp>.npz
        ...
        test_inference_log.csv

    Each prediction NPZ contains:
      semantic           [X,Y,Z] uint8
      occupancy          [X,Y,Z] float32 probability
      semantic_confidence[X,Y,Z] float32
      voxel_size
      origin
      bounds
      camera
      timestamp
      image_path
    """

    selected = select_test_samples(
        test_samples
    )

    if not selected:
        return {
            "ran": False,
            "n_samples": 0,
        }

    test_dataset = SemanticVoxelDataset(
        selected,
        train=False,
    )

    epoch_dir = (
        output_root
        / "test_inference"
        / f"epoch_{epoch:03d}"
    )
    epoch_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_sample_log_path = (
        output_root
        / "test_inference_log.csv"
    )

    log_columns = [
        "epoch",
        "camera",
        "timestamp",
        "npz_path",
        "occupancy_precision",
        "occupancy_recall",
        "occupancy_f1",
        "gt_occupied_fraction",
        "pred_occupied_fraction",
        "miou",
        "hazard_miou",
        "small_hazard_miou",
        "pred_npz",
    ]

    file_exists = per_sample_log_path.exists()

    with open(
        per_sample_log_path,
        "a",
        newline="",
    ) as log_file:

        writer = csv.DictWriter(
            log_file,
            fieldnames=log_columns,
        )

        if not file_exists:
            writer.writeheader()

        model.eval()

        all_ious = []
        all_occ_tp = 0
        all_occ_fp = 0
        all_occ_fn = 0
        all_gt_occ = 0
        all_pred_occ = 0
        all_pred_occ_probability = 0.0
        total_voxels = 0
        total_observed_voxels = 0
        all_gt_occ_observed = 0
        all_pred_occ_observed = 0
        all_pred_occ_probability_observed = 0.0

        for i, sample in enumerate(selected):

            item = test_dataset[i]

            image = item["image"].unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            # Ablation/deployment path: only the 4-channel foreground RF-DETR prior is used.
            rf_prior = item["rf_prior"].unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            camera_id = item["camera_id"].unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            flip = item["flip"].unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            scale = item["scale"].unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            pad_x = item["pad_x"].unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            pad_y = item["pad_y"].unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            orig_w = item["orig_w"].unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            orig_h = item["orig_h"].unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(
                    USE_AMP and device.type == "cuda"
                ),
            ):

                occ_logits, sem_logits, _ = model(
                    image,
                    rf_prior,
                    camera_id,
                    flip,
                    scale,
                    pad_x,
                    pad_y,
                    orig_w,
                    orig_h,
                )

            occupancy_prob = (
                torch.sigmoid(
                    occ_logits
                )
                .squeeze(0)
                .squeeze(0)
                .float()
                .cpu()
                .numpy()
            )

            semantic_prob = (
                torch.softmax(
                    sem_logits,
                    dim=1,
                )
                .squeeze(0)
                .float()
                .cpu()
                .numpy()
            )

            pred_occupancy = (
                occupancy_prob
                >= TEST_OCCUPANCY_THRESHOLD
            )

            semantic_argmax = (
                semantic_prob.argmax(axis=0)
                .astype(np.int64)
            )

            # No background semantic class exists. A voxel predicted as not
            # occupied is represented as 255 in the saved semantic volume.
            semantic_pred = semantic_argmax.astype(np.uint8)
            semantic_pred[~pred_occupancy] = 255

            semantic_confidence = (
                semantic_prob.max(axis=0)
                .astype(np.float32)
            )

            gt_semantic = (
                item["semantic"]
                .numpy()
            )

            # V8 occupancy means "one of the four target semantic classes",
            # not generic physical geometry occupancy.
            gt_occupancy = (
                item["occupancy"]
                .numpy()
                > 0
            )

            # ----------------------------
            # Occupancy metrics
            # ----------------------------

            tp = int(
                np.logical_and(
                    pred_occupancy,
                    gt_occupancy,
                ).sum()
            )

            fp = int(
                np.logical_and(
                    pred_occupancy,
                    ~gt_occupancy,
                ).sum()
            )

            fn = int(
                np.logical_and(
                    ~pred_occupancy,
                    gt_occupancy,
                ).sum()
            )

            precision = (
                tp / (tp + fp)
                if (tp + fp)
                else 0.0
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn)
                else 0.0
            )

            f1 = (
                2.0 * precision * recall
                / (precision + recall)
                if (precision + recall)
                else 0.0
            )

            gt_occ_fraction = float(
                gt_occupancy.mean()
            )

            pred_occ_fraction = float(
                pred_occupancy.mean()
            )

            pred_occ_probability_mean = float(
                occupancy_prob.mean()
            )

            # Secondary diagnostic on the original MapAnything-observed subset.
            # V10 does NOT use this subset for training occupancy supervision;
            # unobserved voxels are also explicit negative targets.
            observed_mask_np = item["supervision_mask"].numpy() > 0.5
            if observed_mask_np.any():
                gt_occ_fraction_observed = float(
                    gt_occupancy[observed_mask_np].mean()
                )
                pred_occ_fraction_observed = float(
                    pred_occupancy[observed_mask_np].mean()
                )
                pred_occ_probability_mean_observed = float(
                    occupancy_prob[observed_mask_np].mean()
                )
            else:
                gt_occ_fraction_observed = 0.0
                pred_occ_fraction_observed = 0.0
                pred_occ_probability_mean_observed = 0.0

            # ----------------------------
            # Semantic metrics
            # ----------------------------

            cm = np.zeros(
                (NUM_CLASSES, NUM_CLASSES),
                dtype=np.int64,
            )

            valid = (
                gt_semantic
                != IGNORE_INDEX
            )

            gt_valid = gt_semantic[valid]
            pred_valid = semantic_argmax[valid]

            bincount = np.bincount(
                gt_valid * NUM_CLASSES
                + pred_valid,
                minlength=NUM_CLASSES * NUM_CLASSES,
            )

            cm += bincount.reshape(
                NUM_CLASSES,
                NUM_CLASSES,
            )

            ious = iou_from_cm(
                torch.from_numpy(cm)
            )

            miou, hazard_miou, small_miou = (
                summarize_ious(ious)
            )

            # ----------------------------
            # Save prediction
            # ----------------------------

            voxel_size = (
                BOUNDS[1] - BOUNDS[0]
            ) / VOXEL_SHAPE[0]

            pred_path = (
                epoch_dir
                / f"{sample['camera']}_"
                f"{sample['timestamp']}.npz"
            )

            np.savez_compressed(
                pred_path,
                semantic=semantic_pred,
                occupancy=occupancy_prob.astype(
                    np.float32
                ),
                semantic_confidence=semantic_confidence,
                voxel_size=np.float32(voxel_size),
                origin=np.array(
                    [
                        BOUNDS[0],
                        BOUNDS[2],
                        BOUNDS[4],
                    ],
                    dtype=np.float32,
                ),
                bounds=np.array(
                    BOUNDS,
                    dtype=np.float32,
                ),
                camera=np.array(
                    sample["camera"]
                ),
                timestamp=np.int64(
                    sample["timestamp"]
                ),
                image_path=np.array(
                    str(sample["image_path"])
                ),
                class_names=np.array(
                    [
                        CLASS_NAMES[i]
                        for i in range(NUM_CLASSES)
                    ],
                    dtype="<U32",
                ),
            )

            writer.writerow(
                {
                    "epoch": epoch,
                    "camera": sample["camera"],
                    "timestamp": sample["timestamp"],
                    "npz_path": str(sample["npz_path"]),
                    "occupancy_precision": precision,
                    "occupancy_recall": recall,
                    "occupancy_f1": f1,
                    "gt_occupied_fraction": gt_occ_fraction,
                    "pred_occupied_fraction": pred_occ_fraction,
                    "miou": miou,
                    "hazard_miou": hazard_miou,
                    "small_hazard_miou": small_miou,
                    "pred_npz": str(pred_path),
                }
            )

            log_file.flush()

            all_occ_tp += tp
            all_occ_fp += fp
            all_occ_fn += fn
            all_gt_occ += int(
                gt_occupancy.sum()
            )
            all_pred_occ += int(
                pred_occupancy.sum()
            )
            all_pred_occ_probability += float(
                occupancy_prob.sum()
            )
            total_voxels += int(
                gt_occupancy.size
            )

            observed_count = int(observed_mask_np.sum())
            total_observed_voxels += observed_count
            if observed_count > 0:
                all_gt_occ_observed += int(gt_occupancy[observed_mask_np].sum())
                all_pred_occ_observed += int(pred_occupancy[observed_mask_np].sum())
                all_pred_occ_probability_observed += float(
                    occupancy_prob[observed_mask_np].sum()
                )

            all_ious.append(ious)

            print(
                f"[TEST] epoch={epoch:03d} "
                f"{sample['camera']} "
                f"{sample['timestamp']} | "
                f"occ P={precision:.3f} "
                f"R={recall:.3f} "
                f"F1={f1:.3f} | "
                f"pred_occ={pred_occ_fraction:.4f} "
                f"gt_occ={gt_occ_fraction:.4f} | "
                f"pred_obs={pred_occ_fraction_observed:.4f} "
                f"gt_obs={gt_occ_fraction_observed:.4f} | "
                f"hazard_mIoU={hazard_miou:.3f}"
            )

        # Aggregate occupancy metrics.
        aggregate_precision = (
            all_occ_tp / (all_occ_tp + all_occ_fp)
            if (all_occ_tp + all_occ_fp)
            else 0.0
        )

        aggregate_recall = (
            all_occ_tp / (all_occ_tp + all_occ_fn)
            if (all_occ_tp + all_occ_fn)
            else 0.0
        )

        aggregate_f1 = (
            2.0
            * aggregate_precision
            * aggregate_recall
            / (aggregate_precision + aggregate_recall)
            if (aggregate_precision + aggregate_recall)
            else 0.0
        )

        mean_ious = np.nanmean(
            np.stack(all_ious),
            axis=0,
        )

        _, aggregate_hazard_miou, aggregate_small_miou = (
            summarize_ious(mean_ious)
        )

        result = {
            "ran": True,
            "n_samples": len(selected),
            "hazard_miou": aggregate_hazard_miou,
            "small_hazard_miou": aggregate_small_miou,
            "occupancy_precision": aggregate_precision,
            "occupancy_recall": aggregate_recall,
            "occupancy_f1": aggregate_f1,
            "gt_occupied_fraction": (
                all_gt_occ / max(total_voxels, 1)
            ),
            "pred_occupied_fraction": (
                all_pred_occ / max(total_voxels, 1)
            ),
            "pred_occupancy_probability_mean": (
                all_pred_occ_probability / max(total_voxels, 1)
            ),
            "gt_occupied_fraction_observed": (
                all_gt_occ_observed / max(total_observed_voxels, 1)
            ),
            "pred_occupied_fraction_observed": (
                all_pred_occ_observed / max(total_observed_voxels, 1)
            ),
            "pred_occupancy_probability_mean_observed": (
                all_pred_occ_probability_observed
                / max(total_observed_voxels, 1)
            ),
        }

        print(
            f"[TEST] epoch={epoch:03d} SUMMARY | "
            f"occ P={aggregate_precision:.3f} "
            f"R={aggregate_recall:.3f} "
            f"F1={aggregate_f1:.3f} | "
            f"pred_occ={result['pred_occupied_fraction']:.4f} "
            f"pred_prob={result['pred_occupancy_probability_mean']:.4f} "
            f"gt_occ={result['gt_occupied_fraction']:.4f} | "
            f"pred_obs={result['pred_occupied_fraction_observed']:.4f} "
            f"gt_obs={result['gt_occupied_fraction_observed']:.4f} | "
            f"hazard_mIoU={aggregate_hazard_miou:.3f}"
        )

        return result


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    metrics,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "voxel_shape": VOXEL_SHAPE,
            "voxel_size": VOXEL_SIZE,
            "bounds": BOUNDS,
            "source_voxel_shape": SOURCE_VOXEL_SHAPE,
            "source_bounds": SOURCE_BOUNDS,
            "in_memory_y_crop": True,
            "class_names": CLASS_NAMES,
            "prior_channels": PRIOR_CHANNELS,
            "rf_detr_prior_channels": RFDETR_PRIOR_CHANNELS,
            "rf_detr_foreground_only": True,
            "cached_rf_detr_prior_channels": CACHED_RFDETR_PRIOR_CHANNELS,
            "foreground_only_semantic_task": True,
            "background_is_negative_occupancy": True,
            "fallen_fence_is_negative_occupancy": True,
            "unobserved_is_negative_occupancy": True,
            "occupancy_supervision_full_grid": True,
            "occupancy_focal_gamma": OCC_FOCAL_GAMMA,
            "occupancy_hard_negative_start": OCC_HARD_NEGATIVE_START,
            "rf_multiscale_fusion": True,
            "explicit_rfdetr_semantic_lift": True,
            "semantic_prior_weight": SEMANTIC_PRIOR_WEIGHT,
            "semantic_prior_temperature": SEMANTIC_PRIOR_TEMPERATURE,
            "semantic_occupancy_gate_weight": 0.0,
            "foreground_only_semantic_output": True,
            "occupancy_loss_type": "asymmetric_hard_negative_focal",
            "occupancy_false_positive_weight": OCC_FALSE_POSITIVE_WEIGHT,
            "occupancy_false_negative_weight": OCC_FALSE_NEGATIVE_WEIGHT,
            "occupancy_loss_weight": OCC_WEIGHT,
            "occupancy_hard_negative_mix": OCC_HARD_NEGATIVE_MIX,
            "occupancy_easy_negative_mix": OCC_EASY_NEGATIVE_MIX,
            "lidar_ablation": True,
            "lidar_used": False,
        },
        path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
    )

    parser.add_argument(
        "--overfit_test",
        action="store_true",
    )

    parser.add_argument(
        "--no_sampler",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help=(
            "Resume training from a checkpoint. "
            "Use --resume to load checkpoints/latest.pth, "
            "or --resume /path/to/checkpoint.pth."
        ),
    )

    args = parser.parse_args()

    seed_everything(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    device = torch.device("cuda")

    print("=" * 80)
    print("RGB + RF-DETR -> 3D FOREGROUND SEMANTIC VOXEL MODEL V10 (5 CM) — LIDAR ABLATION")
    print("=" * 80)
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )
    print(
        "VRAM:",
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
    )
    print("Occupancy supervision: FULL TARGET GRID (UNOBSERVED = EMPTY)")
    print("Training: RGB + RF-DETR only")
    print("Validation/test/deployment: RGB + RF-DETR only")
    print("RF-DETR prior source: v5_priors_letterboxed (LiDAR channels ignored)")
    print("Experiment output:", OUTPUT_ROOT)
    print("RF-DETR fusion: MULTI-SCALE P2/P3/P4")
    print("Occupancy loss: HARD-NEGATIVE ASYMMETRIC FOCAL")
    print("V10 source grid (stored NPZ):", SOURCE_VOXEL_SHAPE, SOURCE_BOUNDS)
    print("V10 target grid (in-memory crop):", VOXEL_SHAPE, BOUNDS)
    print("V10 latent grid:", LATENT_SHAPE)
    print("5 cm target voxels:", int(np.prod(VOXEL_SHAPE)))
    print("Target physical size (m):", (BOUNDS[1]-BOUNDS[0], BOUNDS[3]-BOUNDS[2], BOUNDS[5]-BOUNDS[4]))

    samples = find_samples()

    if not samples:
        raise RuntimeError(
            f"No NPZs found under {NPZ_ROOT}"
        )

    train_samples, val_samples, test_samples = (
        temporal_split(samples)
    )

    if args.overfit_test:
        selected = []

        for camera in ("CAM2", "CAM6"):
            camera_samples = [
                s for s in samples
                if s["camera"] == camera
            ]

            if camera_samples:
                step = max(
                    1,
                    len(camera_samples) // 25,
                )

                selected.extend(
                    camera_samples[::step][:25]
                )

        train_samples = selected
        val_samples = selected
        test_samples = []

        print(
            "******** OVERFIT TEST ********"
        )

    print(f"Train: {len(train_samples):,}")
    print(f"Val:   {len(val_samples):,}")
    print(f"Test:  {len(test_samples):,}")

    train_dataset = SemanticVoxelDataset(
        train_samples,
        train=True,
    )
    val_dataset = SemanticVoxelDataset(
        val_samples,
        train=False,
    )

    if args.no_sampler:
        train_sampler = None
        shuffle = True
    else:
        sample_weights = compute_sample_weights(
            train_samples
        )

        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_samples),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = RGBPriorToSemanticVoxelModelV10().to(device)

    class_weights = compute_class_weights(
        train_samples
    ).to(device)

    # Compute the old ratio for diagnostics/checkpoint metadata only.
    # The actual loss uses the asymmetric FP/FN formulation above.
    occupancy_pos_weight = compute_occupancy_pos_weight(
        train_samples
    ).to(device)

    criterion = MultiTaskVoxelLoss(
        class_weights,
        occupancy_pos_weight,
    ).to(device)

    backbone_params = []
    other_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name.startswith("encoder."):
            backbone_params.append(parameter)
        else:
            other_params.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": args.lr * BACKBONE_LR_MULT,
            },
            {
                "params": other_params,
                "lr": args.lr,
            },
        ],
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(
                max(WARMUP_EPOCHS, 1)
            )

        progress = (
            epoch - WARMUP_EPOCHS
        ) / max(
            args.epochs - WARMUP_EPOCHS,
            1,
        )

        cosine = 0.5 * (
            1.0 + math.cos(
                math.pi * progress
            )
        )

        return (
            MIN_LR / args.lr
            + (
                1.0 - MIN_LR / args.lr
            ) * cosine
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir = (
        OUTPUT_ROOT / "checkpoints"
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_ROOT / "dataset_split.json",
        "w",
    ) as f:
        json.dump(
            {
                "train": [
                    str(x["npz_path"])
                    for x in train_samples
                ],
                "val": [
                    str(x["npz_path"])
                    for x in val_samples
                ],
                "test": [
                    str(x["npz_path"])
                    for x in test_samples
                ],
            },
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # RESUME SUPPORT
    # ------------------------------------------------------------
    # Checkpoints are written at the end of every completed epoch.
    # Therefore, resuming from latest.pth starts at the next epoch.
    #
    # The checkpoint restores:
    #   - model weights
    #   - optimizer state
    #   - LR scheduler state
    #
    # history.json is also restored so best-metric tracking and
    # early-stopping patience continue from the previous run.
    # ------------------------------------------------------------
    start_epoch = 1

    best_hazard_miou = -1.0
    best_full_miou = -1.0
    patience = 0
    max_patience = 8

    history = []

    if args.resume is not None:
        if args.resume == "latest":
            resume_path = checkpoint_dir / "latest.pth"
        else:
            resume_path = Path(args.resume).expanduser().resolve()

        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )

        print()
        print("=" * 80)
        print("RESUMING TRAINING")
        print("=" * 80)
        print("Checkpoint:", resume_path)

        checkpoint = torch.load(
            resume_path,
            map_location=device,
        )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                f"{resume_path} does not contain "
                "'model_state_dict'."
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )
        else:
            raise KeyError(
                f"{resume_path} does not contain "
                "'optimizer_state_dict'."
            )

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )
        else:
            raise KeyError(
                f"{resume_path} does not contain "
                "'scheduler_state_dict'."
            )

        last_epoch = int(
            checkpoint.get("epoch", 0)
        )
        start_epoch = last_epoch + 1

        # Restore the previous epoch history if available.
        history_path = OUTPUT_ROOT / "history.json"

        if history_path.exists():
            try:
                with open(history_path, "r") as f:
                    loaded_history = json.load(f)

                if isinstance(loaded_history, list):
                    history = loaded_history

            except Exception as exc:
                print(
                    "[RESUME] Warning: could not load "
                    f"{history_path}: {exc}"
                )

        # Reconstruct best metrics from the restored history.
        for item in history:
            try:
                val = item.get("val", {})

                hazard = val.get("hazard_miou")
                full = val.get("miou")

                if hazard is not None:
                    best_hazard_miou = max(
                        best_hazard_miou,
                        float(hazard),
                    )

                if full is not None:
                    best_full_miou = max(
                        best_full_miou,
                        float(full),
                    )

            except (TypeError, ValueError):
                continue

        # Reconstruct early-stopping patience as the number of
        # completed epochs since the most recent hazard-mIoU best.
        patience = 0

        for item in reversed(history):
            try:
                hazard = float(
                    item["val"]["hazard_miou"]
                )
            except (KeyError, TypeError, ValueError):
                continue

            if (
                best_hazard_miou >= 0.0
                and abs(
                    hazard - best_hazard_miou
                ) < 1e-12
            ):
                break

            patience += 1

        print(
            f"[RESUME] Last completed epoch: {last_epoch}"
        )
        print(
            f"[RESUME] Starting epoch: {start_epoch}"
        )
        print(
            f"[RESUME] Target epoch: {args.epochs}"
        )
        print(
            f"[RESUME] Best hazard mIoU: "
            f"{best_hazard_miou:.6f}"
        )
        print(
            f"[RESUME] Best full mIoU: "
            f"{best_full_miou:.6f}"
        )
        print(
            f"[RESUME] Early-stop patience: "
            f"{patience}/{max_patience}"
        )

        if start_epoch > args.epochs:
            print()
            print(
                "[RESUME] Checkpoint is already at or beyond "
                "the requested number of epochs."
            )
            print(
                "[RESUME] Nothing to train."
            )
            return

        print("=" * 80)
        print()

    # Separate, append-only epoch log for easy plotting/inspection.
    training_log_path = OUTPUT_ROOT / "training_log.csv"

    # Fixed held-out subset used only for periodic qualitative/metric
    # monitoring. It never affects checkpoint selection.
    test_inference_samples = select_test_samples(
        test_samples
    )

    with open(
        OUTPUT_ROOT / "test_inference_samples.json",
        "w",
    ) as f:
        json.dump(
            [
                {
                    "camera": s["camera"],
                    "timestamp": s["timestamp"],
                    "npz_path": str(s["npz_path"]),
                    "image_path": str(s["image_path"]),
                }
                for s in test_inference_samples
            ],
            f,
            indent=2,
        )

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        print()
        print("=" * 80)
        print(
            f"EPOCH {epoch}/{args.epochs}"
        )
        print("=" * 80)

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            epoch=epoch,
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                optimizer,
                device,
                train=False,
                epoch=epoch,
            )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # Periodic TEST inference.
        #
        # IMPORTANT: test metrics are monitoring only. They do not
        # influence checkpoint selection or early stopping.
        test_metrics = {
            "ran": False,
            "n_samples": 0,
        }

        if (
            test_inference_samples
            and (
                epoch % TEST_INFERENCE_EVERY == 0
                or epoch == args.epochs
            )
        ):
            with torch.no_grad():
                test_metrics = run_test_inference(
                    model,
                    test_inference_samples,
                    device,
                    epoch,
                    OUTPUT_ROOT,
                )

        print()
        print(
            f"TRAIN loss={train_metrics['total']:.5f} "
            f"mIoU={train_metrics['miou']:.4f} "
            f"hazard={train_metrics['hazard_miou']:.4f} "
            f"small={train_metrics['small_hazard_miou']:.4f}"
        )

        print(
            f"VAL   loss={val_metrics['total']:.5f} "
            f"mIoU={val_metrics['miou']:.4f} "
            f"hazard={val_metrics['hazard_miou']:.4f} "
            f"small={val_metrics['small_hazard_miou']:.4f}"
        )

        print(
            f"LR={current_lr:.7f}"
        )

        print()
        print("Validation IoU:")

        for class_id, iou in enumerate(
            val_metrics["ious"]
        ):
            print(
                f"  {class_id}: "
                f"{CLASS_NAMES[class_id]:20s} "
                f"{iou:.4f}"
            )

        metrics = {
            "train": {
                "loss": train_metrics["total"],
                "occ": train_metrics["occ"],
                "ce": train_metrics["ce"],
                "focal": train_metrics["focal"],
                "dice": train_metrics["dice"],
                "coarse": train_metrics["coarse"],
                "miou": train_metrics["miou"],
                "hazard_miou": train_metrics[
                    "hazard_miou"
                ],
                "small_hazard_miou": train_metrics[
                    "small_hazard_miou"
                ],
                "ious": train_metrics["ious"].tolist(),
            },
            "val": {
                "loss": val_metrics["total"],
                "occ": val_metrics["occ"],
                "ce": val_metrics["ce"],
                "focal": val_metrics["focal"],
                "dice": val_metrics["dice"],
                "coarse": val_metrics["coarse"],
                "miou": val_metrics["miou"],
                "hazard_miou": val_metrics[
                    "hazard_miou"
                ],
                "small_hazard_miou": val_metrics[
                    "small_hazard_miou"
                ],
                "ious": val_metrics["ious"].tolist(),
            },
            "test_inference": test_metrics,
            "lr": current_lr,
        }

        history.append(metrics)

        val_ious = val_metrics["ious"]

        training_log_row = {
            "epoch": epoch,
            "lr": current_lr,

            "train_total": train_metrics["total"],
            "train_occ": train_metrics["occ"],
            "train_occ_fp": train_metrics["occ_fp"],
            "train_occ_fn": train_metrics["occ_fn"],
            "train_ce": train_metrics["ce"],
            "train_focal": train_metrics["focal"],
            "train_dice": train_metrics["dice"],
            "train_coarse": train_metrics["coarse"],
            "train_miou": train_metrics["miou"],
            "train_hazard_miou": train_metrics["hazard_miou"],
            "train_small_hazard_miou": train_metrics[
                "small_hazard_miou"
            ],

            "val_total": val_metrics["total"],
            "val_occ": val_metrics["occ"],
            "val_occ_fp": val_metrics["occ_fp"],
            "val_occ_fn": val_metrics["occ_fn"],
            "val_ce": val_metrics["ce"],
            "val_focal": val_metrics["focal"],
            "val_dice": val_metrics["dice"],
            "val_coarse": val_metrics["coarse"],
            "val_miou": val_metrics["miou"],
            "val_hazard_miou": val_metrics["hazard_miou"],
            "val_small_hazard_miou": val_metrics[
                "small_hazard_miou"
            ],

            "val_loose_trash_iou": val_ious[0],
            "val_mud_spill_iou": val_ious[1],
            "val_public_greenway_iou": val_ious[2],
            "val_sidewalk_iou": val_ious[3],

            "test_ran": int(
                test_metrics.get("ran", False)
            ),
            "test_hazard_miou": test_metrics.get(
                "hazard_miou"
            ),
            "test_small_hazard_miou": test_metrics.get(
                "small_hazard_miou"
            ),
            "test_occupancy_precision": test_metrics.get(
                "occupancy_precision"
            ),
            "test_occupancy_recall": test_metrics.get(
                "occupancy_recall"
            ),
            "test_occupancy_f1": test_metrics.get(
                "occupancy_f1"
            ),
            "test_gt_occupied_fraction": test_metrics.get(
                "gt_occupied_fraction"
            ),
            "test_pred_occupied_fraction": test_metrics.get(
                "pred_occupied_fraction"
            ),
            "test_pred_occupancy_probability_mean": test_metrics.get(
                "pred_occupancy_probability_mean"
            ),
            "test_gt_occupied_fraction_observed": test_metrics.get(
                "gt_occupied_fraction_observed"
            ),
            "test_pred_occupied_fraction_observed": test_metrics.get(
                "pred_occupied_fraction_observed"
            ),
            "test_pred_occupancy_probability_mean_observed": test_metrics.get(
                "pred_occupancy_probability_mean_observed"
            ),
        }

        append_training_log(
            training_log_path,
            training_log_row,
        )

        save_checkpoint(
            checkpoint_dir / "latest.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            metrics,
        )

        # Primary checkpoint criterion: hazard mIoU.
        if (
            val_metrics["hazard_miou"]
            > best_hazard_miou
        ):
            best_hazard_miou = (
                val_metrics["hazard_miou"]
            )
            patience = 0

            save_checkpoint(
                checkpoint_dir / "best_hazard.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                metrics,
            )

            print(
                "[CHECKPOINT] New best HAZARD mIoU."
            )
        else:
            patience += 1

        if (
            val_metrics["miou"]
            > best_full_miou
        ):
            best_full_miou = val_metrics["miou"]

            save_checkpoint(
                checkpoint_dir / "best_full.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                metrics,
            )

        with open(
            OUTPUT_ROOT / "history.json",
            "w",
        ) as f:
            json.dump(
                history,
                f,
                indent=2,
            )

        if patience >= max_patience:
            print(
                f"[EARLY STOP] No hazard mIoU "
                f"improvement for {max_patience} epochs."
            )
            break

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(
        "Best hazard checkpoint:",
        checkpoint_dir / "best_hazard.pth",
    )
    print(
        "Best full mIoU checkpoint:",
        checkpoint_dir / "best_full.pth",
    )
    print(
        "Epoch metrics log:",
        training_log_path,
    )
    print(
        "Test inference log:",
        OUTPUT_ROOT / "test_inference_log.csv",
    )


if __name__ == "__main__":
    main()
