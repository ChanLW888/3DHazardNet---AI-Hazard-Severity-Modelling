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

Training-only teacher:
    RGB + four foreground RF-DETR prior channels + existing 2-channel LiDAR prior

The cached prior files are unchanged: they contain 6 RF-DETR channels followed
by 2 LiDAR channels. The model uses only the four foreground RF-DETR channels.
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
- LiDAR is privileged training information only.
- RGB + RF-DETR is the deployment path.

Training
--------
- Separate binary occupancy and four-class semantic heads.
- Occupancy uses asymmetric hard-negative focal loss.
- Semantic supervision is applied only on target-foreground voxels.
- LiDAR teacher provides occupancy and semantic distillation.
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

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from rfdetr import RFDETRSegSmall


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/hazard_test_101"
)
NPZ_ROOT = DATASET_ROOT / "semantic_voxels_new_roi_5cm_filtered"
OUTPUT_ROOT = DATASET_ROOT / "rgb_voxel_training_v10_1_5cm_xm8_8_y0_15_z025_10_dense_latent_hazard_empty"
PRIOR_ROOT = DATASET_ROOT / "v5_priors_letterboxed"
# v5_priors is the MASTER SAMPLE INDEX: only samples with valid priors are used.

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

# Cached priors still contain the original 6 RF-DETR channels on disk.
# V10.1 still uses only the four foreground RF-DETR class channels for the
# learned branch. The RF-DETR lift preserves their absolute foreground mass,
# so low-confidence/background pixels do not become fake class evidence.
CACHED_RFDETR_PRIOR_CHANNELS = 6
RFDETR_PRIOR_CHANNELS = 4
LIDAR_PRIOR_CHANNELS = 2
PRIOR_CHANNELS = 8
PRIOR_SIZE = 192

# Deployment inference generates the RF-DETR prior directly from the RGB
# image instead of reading the cached V5 prior NPZ.
RFDETR_CONFIDENCE_THRESHOLD = 0.30
RFDETR_MODEL_PATH = Path("/home/lcha0115/bt60_scratch/lcha_data/map-anything/personal_data/2D-Seg-Road-Nissan.v13i.coco-segmentation/outputs/checkpoint_best_total.pth")


# RF-DETR semantic prior is now an EXPLICIT 2D -> 3D semantic anchor.
# The projected RF-DETR class probabilities are converted to log-priors
# and added directly to the final 3D semantic logits. This prevents the
# network from simply learning to ignore the already-available 2D labels.
SEMANTIC_PRIOR_WEIGHT = 1.00
SEMANTIC_PRIOR_TEMPERATURE = 0.8



# LiDAR teacher/student training. The cached LiDAR prior remains exactly 2 channels.
# The teacher receives LiDAR; the deployment student never does.
# V9: LiDAR remains a privileged teacher, but its influence is deliberately weak.
# The deployment student is still RGB + RF-DETR only.
DISTILL_OCC_WEIGHT = 0.15
DISTILL_SEM_WEIGHT = 0.10
TEACHER_OCC_GT_WEIGHT = 0.15
TEACHER_SEM_GT_WEIGHT = 0.10
DISTILL_WARMUP_EPOCHS = 12
DISTILL_TEMPERATURE = 1.0


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


class LiDARPrivilegedFusion(nn.Module):
    """
    Training-time LiDAR residual branch.

    LiDAR is deliberately kept separate from RF-DETR so the network can
    learn which geometric information is useful without making LiDAR part
    of the deployment representation. When lidar_present is false, this
    branch contributes exactly zero.
    """

    def __init__(self, channels=LATENT_CHANNELS):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid(),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(16, channels),
            nn.GELU(),
        )

    def forward(self, rgb_rf, lidar_feat, lidar_present):
        gate = self.gate(torch.cat([rgb_rf, lidar_feat], dim=1))
        residual = self.residual(
            torch.cat([rgb_rf, gate * lidar_feat], dim=1)
        )
        present = lidar_present.float().view(-1, 1, 1, 1)
        return rgb_rf + present * residual


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
# frame. Therefore V5 must NOT apply the LiDAR-to-camera extrinsic here.
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
    camera frame. No LiDAR-to-camera extrinsic is applied here.

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

class LiDARVoxelGeometryLift(nn.Module):
    """
    Turn the EXISTING 2-channel LiDAR prior into a voxel-aligned 3D feature.

    No new LiDAR prior channels are required. Each latent voxel is projected
    into the same rectified/letterboxed image coordinate system and the two
    cached LiDAR channels are sampled at that location. A small 3D network
    then converts those two sampled values into the latent feature space.

    This is deliberately stronger than the old 2D residual-only LiDAR fusion:
    the LiDAR evidence is explicitly attached to the corresponding 3D voxel.
    """

    def __init__(self, camera_projection, out_channels=LATENT_CHANNELS):
        super().__init__()
        self.camera_projection = camera_projection
        self.net = nn.Sequential(
            nn.Conv3d(2, 64, 1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv3d(64, 96, 3, padding=1),
            nn.GroupNorm(12, 96),
            nn.GELU(),
            nn.Conv3d(96, out_channels, 1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv3d(out_channels, out_channels // 2, 1),
            nn.GELU(),
            nn.Conv3d(out_channels // 2, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        lidar_prior,
        coords_metric,
        camera_id,
        flip,
        scale,
        pad_x,
        pad_y,
        orig_w,
        orig_h,
    ):
        B = lidar_prior.shape[0]
        coords = coords_metric.expand(B, -1, -1)

        grid = self.camera_projection(
            coords, camera_id, flip, scale, pad_x, pad_y, orig_w, orig_h
        )

        sampled = F.grid_sample(
            lidar_prior,
            grid.unsqueeze(1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(2)

        nx, ny, nz = LATENT_SHAPE
        sampled = sampled.reshape(B, LIDAR_PRIOR_CHANNELS, nx, ny, nz)

        feat = self.net(sampled)
        return feat * self.gate(feat)


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

        # LiDAR is a separate privileged branch used only during training.
        self.lidar_prior_encoder = PriorEncoder(
            in_channels=LIDAR_PRIOR_CHANNELS,
            out_channels=LATENT_CHANNELS,
        )

        self.rf_prior_fusion = RGBPriorFusion(
            channels=LATENT_CHANNELS,
        )

        self.lidar_privileged_fusion = LiDARPrivilegedFusion(
            channels=LATENT_CHANNELS,
        )

        self.lift = GeometryAwareLift(
            channels=LATENT_CHANNELS
        )

        # Stronger voxel-aligned LiDAR geometry path. Still training-only.
        self.lidar_voxel_lift = LiDARVoxelGeometryLift(
            self.lift.camera_projection,
            out_channels=LATENT_CHANNELS,
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
        lidar_prior=None,
        lidar_present=None,
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

        # --------------------------------------------------------
        # LiDAR branch: privileged training signal only.
        # If lidar_prior is None, or lidar_present is all false, this
        # branch has no effect on the RGB + RF-DETR representation.
        # --------------------------------------------------------
        if lidar_prior is not None:
            lidar_prior = F.interpolate(
                lidar_prior,
                size=pyramid["p2"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            lidar_feat = self.lidar_prior_encoder(lidar_prior)

            if lidar_present is None:
                lidar_present = torch.ones(
                    image.shape[0],
                    device=image.device,
                    dtype=torch.bool,
                )

            pyramid["p2"] = self.lidar_privileged_fusion(
                pyramid["p2"],
                lidar_feat,
                lidar_present,
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

        # --------------------------------------------------------
        # VOXEL-ALIGNED LiDAR GEOMETRY (teacher path only)
        # --------------------------------------------------------
        # The same cached 2-channel prior is projected directly into the
        # latent voxel grid. This gives the LiDAR teacher explicit 3D
        # correspondence instead of relying only on 2D feature fusion.
        if lidar_prior is not None:
            lidar_volume = self.lidar_voxel_lift(
                lidar_prior,
                coords_metric,
                camera_id,
                flip,
                scale,
                pad_x,
                pad_y,
                orig_w,
                orig_h,
            )
            present = (
                torch.ones(image.shape[0], device=image.device, dtype=volume.dtype)
                if lidar_present is None
                else lidar_present.to(volume.dtype)
            ).view(-1, 1, 1, 1, 1)
            volume = volume + present * lidar_volume

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


def norm_mask(mask, h, w):
    """Normalize an RF-DETR mask to a boolean HxW NumPy array."""
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    mask = np.squeeze(np.asarray(mask))

    if mask.dtype != np.bool_:
        mask = mask > 0.5

    if mask.shape != (h, w):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    return mask


def rf_prior_from_detections(det, h, w):
    """
    Reproduce the exact RF-DETR prior construction used by the V5 generator.

    Output:
        [6, H, W]
        channel 0 = RF-DETR background
        channel 1..5 = RF-DETR class IDs 0..4

    V10.1 subsequently keeps channels 2..5:
        loose_trash, mud_spill, public_greenway, sidewalk
    """
    out = np.zeros((6, h, w), np.float32)

    ids = det.class_id
    conf = det.confidence
    masks = det.mask

    assigned = np.zeros((h, w), bool)

    order = sorted(
        range(len(ids)),
        key=lambda i: float(conf[i]),
        reverse=True,
    )

    for i in order:
        c = float(conf[i])
        cid = int(ids[i])

        if c < RFDETR_CONFIDENCE_THRESHOLD or cid not in range(5):
            continue

        m = norm_mask(masks[i], h, w)

        # Match the generator: highest-confidence detection gets ownership
        # of overlapping pixels first.
        a = m & (~assigned)
        out[cid + 1, a] = c
        assigned[a] = True

    out[0] = 1.0 - out[1:].max(axis=0)
    return out


def letterbox_prior(prior, target_size):
    """
    Exact aspect-ratio-preserving prior resize/padding used by the V5
    RF-DETR prior generator.

    Input:
        [C, H, W] in original camera-image coordinates.

    Output:
        [C, target_size, target_size]
    """
    if prior.ndim != 3:
        raise ValueError(
            f"Expected prior with shape [C,H,W], got {prior.shape}"
        )

    c, oh, ow = prior.shape

    scale = min(
        target_size / ow,
        target_size / oh,
    )

    nw = max(1, int(round(ow * scale)))
    nh = max(1, int(round(oh * scale)))

    prior_tensor = torch.from_numpy(
        prior.astype(np.float32)
    ).unsqueeze(0)

    prior_tensor = F.interpolate(
        prior_tensor,
        size=(nh, nw),
        mode="bilinear",
        align_corners=False,
    )

    pad_x = (target_size - nw) // 2
    pad_y = (target_size - nh) // 2

    # Zero padding = no prior information.
    padded = torch.zeros(
        (1, c, target_size, target_size),
        dtype=prior_tensor.dtype,
    )

    padded[
        :,
        :,
        pad_y:pad_y + nh,
        pad_x:pad_x + nw,
    ] = prior_tensor

    return padded.squeeze(0).numpy()


def generate_rfdetr_prior(image_path, rfdetr_model):
    """
    Run RF-DETR directly on the original RGB image and produce the exact
    4-channel V10.1 model-facing prior at PRIOR_SIZE resolution.

    Returns:
        rf_prior_4: [4, PRIOR_SIZE, PRIOR_SIZE] float32
        full_prior_6: [6, PRIOR_SIZE, PRIOR_SIZE] float32
    """
    image_rgb = np.asarray(
        Image.open(image_path).convert("RGB")
    )

    h, w = image_rgb.shape[:2]

    # The original generator explicitly converts RGB -> BGR before
    # model.predict([bgr]).
    bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    with torch.inference_mode():
        det = rfdetr_model.predict([bgr])[0]

    raw_prior = rf_prior_from_detections(
        det,
        h,
        w,
    )

    full_prior = letterbox_prior(
        raw_prior,
        PRIOR_SIZE,
    ).astype(np.float32)

    # Exact V10.1 foreground channel selection from the cached 8-channel
    # training prior: cached RF channels [2:6].
    rf_prior = full_prior[2:6].copy()

    return rf_prior, full_prior




def main():
    """Standalone V10.1 single-image inference.

    This script does NOT use:
      - the semantic voxel dataset
      - dataset discovery
      - sample indexing
      - cached prior NPZ files
      - timestamp matching
      - LiDAR

    It takes exactly one RGB image and a camera selection (CAM2/CAM6),
    generates the RF-DETR prior on-the-fly, runs the trained V10.1 model,
    and saves the generated prior, prediction volume, original image, and
    metadata.
    """
    parser = argparse.ArgumentParser(
        description="Standalone V10.1 single-image RGB + RF-DETR inference."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to one RGB image.",
    )
    parser.add_argument(
        "--cam",
        type=str,
        required=True,
        choices=["CAM2", "CAM6", "cam2", "cam6"],
        help="Camera that captured the image: CAM2 or CAM6.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "/home/lcha0115/bt60_scratch/lcha_data/map-anything/"
            "personal_data/hazard_test_101/"
            "rgb_voxel_training_v10_1_5cm_xm8_8_y0_15_z025_10_"
            "dense_latent_hazard_empty/checkpoints/best_full.pth"
        ),
        help="Path to the trained V10.1 checkpoint.",
    )
    parser.add_argument(
        "--rfdetr_checkpoint",
        type=str,
        default=str(RFDETR_MODEL_PATH),
        help="Path to the RF-DETR segmentation checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(
            DATASET_ROOT / "v10_1_single_image_inference"
        ),
        help="Output directory root.",
    )
    parser.add_argument(
        "--occupancy_threshold",
        type=float,
        default=TEST_OCCUPANCY_THRESHOLD,
        help="Threshold used to convert occupancy probability to binary occupancy.",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable CUDA BF16 autocast for V10.1 inference.",
    )
    args = parser.parse_args()

    seed_everything(SEED)

    image_path = Path(args.image).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    rfdetr_checkpoint_path = Path(args.rfdetr_checkpoint).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    camera = args.cam.upper()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found:\n{image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"V10.1 checkpoint not found:\n{checkpoint_path}")
    if not rfdetr_checkpoint_path.exists():
        raise FileNotFoundError(
            f"RF-DETR checkpoint not found:\n{rfdetr_checkpoint_path}"
        )
    if not (0.0 <= args.occupancy_threshold <= 1.0):
        raise ValueError("--occupancy_threshold must be in [0,1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    camera_id_value = 0 if camera == "CAM2" else 1

    print("=" * 80)
    print("V10.1 STANDALONE SINGLE-IMAGE INFERENCE")
    print("=" * 80)
    print("Image:", image_path)
    print("Camera:", camera)
    print("GPU:", torch.cuda.get_device_name(0))
    print("V10.1 checkpoint:", checkpoint_path)
    print("RF-DETR checkpoint:", rfdetr_checkpoint_path)
    print("Output root:", output_root)
    print("Deployment input: RGB + generated RF-DETR")
    print("LiDAR: DISABLED")
    print("Dataset/indexing: DISABLED")
    print("Cached prior NPZ: DISABLED")
    print("Target grid:", VOXEL_SHAPE, BOUNDS)
    print("Latent grid:", LATENT_SHAPE)
    print("Voxel size:", VOXEL_SIZE)

    # ------------------------------------------------------------------
    # 1. LOAD + LETTERBOX THE SINGLE IMAGE
    # ------------------------------------------------------------------
    original_image = Image.open(image_path).convert("RGB")
    image_tensor, scale_value, pad_x_value, pad_y_value, ow_value, oh_value = (
        letterbox_image(original_image, IMAGE_SIZE)
    )

    print("\n[IMAGE]")
    print("  Original size:", int(ow_value), "x", int(oh_value))
    print("  Letterbox size:", IMAGE_SIZE, "x", IMAGE_SIZE)
    print("  Scale:", float(scale_value))
    print("  Padding x/y:", int(pad_x_value), int(pad_y_value))

    # ------------------------------------------------------------------
    # 2. GENERATE RF-DETR PRIOR DIRECTLY FROM THIS IMAGE
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("GENERATING RF-DETR PRIOR")
    print("-" * 80)
    print("Confidence threshold:", RFDETR_CONFIDENCE_THRESHOLD)

    rfdetr_model = RFDETRSegSmall.from_checkpoint(
        str(rfdetr_checkpoint_path)
    )
    rfdetr_model.inference(
        dtype=torch.float16,
        batch_size=1,
    )

    rf_prior_np, full_rfdetr_prior_np = generate_rfdetr_prior(
        image_path,
        rfdetr_model,
    )

    del rfdetr_model

    print("RF-DETR prior shape:", rf_prior_np.shape)
    print(
        "RF-DETR foreground prior min/max:",
        float(rf_prior_np.min()),
        float(rf_prior_np.max()),
    )

    # ------------------------------------------------------------------
    # 3. BUILD EXACT V10.1 MODEL INPUTS
    # ------------------------------------------------------------------
    device = torch.device("cuda")
    image = image_tensor.unsqueeze(0).to(device, non_blocking=True)
    rf_prior = torch.from_numpy(
        rf_prior_np.astype(np.float32)
    ).unsqueeze(0).to(device, non_blocking=True)

    camera_id = torch.tensor(
        [camera_id_value], dtype=torch.long, device=device
    )
    flip = torch.tensor([0], dtype=torch.long, device=device)
    scale = torch.tensor([scale_value], dtype=torch.float32, device=device)
    pad_x = torch.tensor([pad_x_value], dtype=torch.float32, device=device)
    pad_y = torch.tensor([pad_y_value], dtype=torch.float32, device=device)
    orig_w = torch.tensor([ow_value], dtype=torch.long, device=device)
    orig_h = torch.tensor([oh_value], dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # 4. LOAD EXACT V10.1 MODEL
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("LOADING V10.1 MODEL")
    print("-" * 80)

    model = RGBPriorToSemanticVoxelModelV10().to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"{checkpoint_path} does not contain 'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.eval()

    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Loaded strictly: True")

    # ------------------------------------------------------------------
    # 5. DEPLOYMENT FORWARD PASS
    # ------------------------------------------------------------------
    amp_enabled = USE_AMP and not args.no_amp and device.type == "cuda"

    print("\n" + "-" * 80)
    print("RUNNING V10.1 INFERENCE")
    print("-" * 80)

    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=amp_enabled,
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
                lidar_prior=None,
                lidar_present=torch.zeros(
                    1, device=device, dtype=torch.bool
                ),
            )

    # ------------------------------------------------------------------
    # 6. CONVERT OUTPUTS TO VOXEL PREDICTIONS
    # ------------------------------------------------------------------
    occupancy_prob = (
        torch.sigmoid(occ_logits)
        .squeeze(0)
        .squeeze(0)
        .float()
        .cpu()
        .numpy()
    )

    semantic_prob = (
        torch.softmax(sem_logits, dim=1)
        .squeeze(0)
        .float()
        .cpu()
        .numpy()
    )

    pred_occupancy = occupancy_prob >= args.occupancy_threshold
    semantic_argmax = semantic_prob.argmax(axis=0).astype(np.uint8)

    # No semantic background class. 255 means predicted empty/unoccupied.
    semantic_pred = semantic_argmax.copy()
    semantic_pred[~pred_occupancy] = 255

    semantic_confidence = semantic_prob.max(axis=0).astype(np.float32)

    if tuple(occupancy_prob.shape) != tuple(VOXEL_SHAPE):
        raise RuntimeError(
            f"Occupancy output shape {occupancy_prob.shape}; expected {VOXEL_SHAPE}"
        )
    if tuple(semantic_pred.shape) != tuple(VOXEL_SHAPE):
        raise RuntimeError(
            f"Semantic output shape {semantic_pred.shape}; expected {VOXEL_SHAPE}"
        )

    print("Prediction shape:", semantic_pred.shape)
    print("Predicted occupied voxels:", int(pred_occupancy.sum()))
    print("Predicted occupied fraction:", float(pred_occupancy.mean()))
    print("Mean occupancy probability:", float(occupancy_prob.mean()))

    # ------------------------------------------------------------------
    # 7. SAVE OUTPUTS
    # ------------------------------------------------------------------
    output_dir = output_root / f"{camera}_{image_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_image_path = output_dir / (
        f"image_original{image_path.suffix or '.jpg'}"
    )
    import shutil
    shutil.copy2(image_path, saved_image_path)

    prior_path = output_dir / "rfdetr_prior.npz"
    np.savez_compressed(
        prior_path,
        prior=rf_prior_np.astype(np.float32),
        full_rfdetr_prior=full_rfdetr_prior_np.astype(np.float32),
        channel_names=np.array(
            ["loose_trash", "mud_spill", "public_greenway", "sidewalk"],
            dtype="<U32",
        ),
        model_input_channels=np.array([2, 3, 4, 5], dtype=np.int32),
        full_prior_channel_names=np.array(
            [
                "background", "fallen_fence", "loose_trash",
                "mud_spill", "public_greenway", "sidewalk",
            ],
            dtype="<U32",
        ),
        camera=np.array(camera),
        image_path=np.array(str(image_path)),
        image_size=np.int32(IMAGE_SIZE),
        prior_size=np.int32(PRIOR_SIZE),
        rfdetr_checkpoint=np.array(str(rfdetr_checkpoint_path)),
        rfdetr_confidence_threshold=np.float32(
            RFDETR_CONFIDENCE_THRESHOLD
        ),
    )

    prediction_path = output_dir / "prediction.npz"
    np.savez_compressed(
        prediction_path,
        semantic=semantic_pred,
        occupancy=occupancy_prob.astype(np.float32),
        occupancy_binary=pred_occupancy.astype(np.uint8),
        semantic_confidence=semantic_confidence,
        semantic_probabilities=semantic_prob.astype(np.float32),
        voxel_size=np.float32(VOXEL_SIZE),
        origin=np.array(
            [BOUNDS[0], BOUNDS[2], BOUNDS[4]], dtype=np.float32
        ),
        bounds=np.array(BOUNDS, dtype=np.float32),
        camera=np.array(camera),
        image_path=np.array(str(image_path)),
        checkpoint=np.array(str(checkpoint_path)),
        class_names=np.array(
            [CLASS_NAMES[i] for i in range(NUM_CLASSES)], dtype="<U32"
        ),
        occupancy_threshold=np.float32(args.occupancy_threshold),
    )

    metadata = {
        "mode": "standalone_single_image",
        "image": str(image_path),
        "camera": camera,
        "camera_id": camera_id_value,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "rfdetr_checkpoint": str(rfdetr_checkpoint_path),
        "rfdetr_confidence_threshold": float(RFDETR_CONFIDENCE_THRESHOLD),
        "cached_prior_used": False,
        "dataset_used": False,
        "index_used": False,
        "lidar_used": False,
        "voxel_size": float(VOXEL_SIZE),
        "voxel_shape": list(VOXEL_SHAPE),
        "bounds": list(BOUNDS),
        "latent_shape": list(LATENT_SHAPE),
        "image_size": int(IMAGE_SIZE),
        "rf_prior_channels": [2, 3, 4, 5],
        "rf_prior_channel_names": [
            "loose_trash", "mud_spill", "public_greenway", "sidewalk"
        ],
        "letterbox": {
            "scale": float(scale_value),
            "pad_x": int(pad_x_value),
            "pad_y": int(pad_y_value),
            "orig_w": int(ow_value),
            "orig_h": int(oh_value),
            "flip": False,
        },
        "occupancy_threshold": float(args.occupancy_threshold),
        "amp_enabled": bool(amp_enabled),
        "files": {
            "image_original": str(saved_image_path),
            "rfdetr_prior": str(prior_path),
            "prediction": str(prediction_path),
        },
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    print("Output directory:", output_dir)
    print("  Image:       ", saved_image_path)
    print("  RF-DETR:     ", prior_path)
    print("  Prediction:  ", prediction_path)
    print("  Metadata:    ", metadata_path)


if __name__ == "__main__":
    main()
