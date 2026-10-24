#!/usr/bin/env python3
"""
RGB-to-semantic-voxel comparison training for the road-hazard dataset.

This script trains ONE of two RGB-only image-to-voxel models per run:
  1) image2voxel_unet: ResNet-50 image encoder + calibrated voxel feature
     sampling + compact 3D U-Net decoder.
  2) image2voxel_resunet: ResNet-50 multi-scale image encoder + calibrated
     voxel feature sampling + residual 3D decoder.

Both models:
  * take only the rectified RGB image as input (no RF-DETR or LiDAR priors);
  * predict four foreground semantic classes on the target voxel grid;
  * predict a separate foreground/empty occupancy logit;
  * use the same data split, target mapping, losses, and validation metrics.

Dataset conventions inherited from the supplied V10.1 training code:
  * semantic NPZ root: DATASET_ROOT/semantic_voxels_new_roi_5cm_filtered
  * cameras: CAM2 and CAM6
  * RGB path: NPZ field `image_path` (with timestamp-based fallback)
  * stored voxel grid: (320, 40, 195), bounds (-8,8, 0,2, 0.25,10), voxel size 5 cm
  * crop Y=0..1.5 m -> output grid (320,30,195)
  * original labels 2..5 map to model labels 0..3:
      0 loose_trash, 1 mud_spill, 2 public_greenway, 3 sidewalk
  * other labels and unobserved voxels are empty for occupancy and ignored
    for semantic classification, as in the supplied V10.1 code.

Geometry:
  * The grid coordinates use X=right, Y=down, Z=forward in the camera frame.
  * Rectified projection matrices P[:, :3] from CAM2/CAM6 calibration files
    are used to sample image features at each latent voxel centre.
  * Images are letterboxed to IMAGE_SIZE; projection accounts for this resize.
  * No occupancy/semantic target is used as a model input.

Example:
  python train_rgb_to_voxels_comparison.py --model image2voxel_unet --epochs 50
  python train_rgb_to_voxels_comparison.py --model image2voxel_resunet --epochs 50

The script writes checkpoints, epoch metrics CSV, JSON history, and the exact
train/validation/test split under a separate output folder for each model.

Important research note:
  Temporal splitting is performed independently per camera to reduce adjacent-
  frame leakage. The test set is not used for model selection. The best
  checkpoint is selected by validation foreground semantic mIoU.
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
from torch.utils.data import Dataset, DataLoader
from torchvision import models


# ---------------------------------------------------------------------------
# Configuration: paths and target geometry copied from the supplied V10.1 code
# ---------------------------------------------------------------------------
DATASET_ROOT = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/hazard_test_101"
)
NPZ_ROOT = DATASET_ROOT / "semantic_voxels_new_roi_5cm_filtered"
IMAGE_ROOT = DATASET_ROOT / "extracted"
CALIBRATION_DIR = IMAGE_ROOT / "calibration"

CALIBRATION_FILES = {
    "CAM2": CALIBRATION_DIR / "CAM2_calib.txt",
    "CAM6": CALIBRATION_DIR / "CAM6_calib.txt",
}

NUM_CLASSES = 4
IGNORE_INDEX = -1
CLASS_NAMES = ["loose_trash", "mud_spill", "public_greenway", "sidewalk"]

SOURCE_SHAPE = (320, 40, 195)
SOURCE_BOUNDS = (-8.0, 8.0, 0.0, 2.0, 0.25, 10.0)
VOXEL_SHAPE = (320, 30, 195)
BOUNDS = (-8.0, 8.0, 0.0, 1.5, 0.25, 10.0)
VOXEL_SIZE = 0.05

IMAGE_SIZE = 384
LATENT_SHAPE = (40, 8, 20)  # X,Y,Z
LATENT_CHANNELS = 96
BATCH_SIZE = 8
NUM_WORKERS = 4
EPOCHS = 50
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
SEED = 42
OCCUPANCY_THRESHOLD = 0.5
USE_PRETRAINED = True
USE_AMP = True


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unpack_scalar(value):
    """Convert scalar or 0-D NumPy metadata into a Python value."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
    return value


def letterbox_image(image, size=IMAGE_SIZE):
    """Resize RGB without changing aspect ratio, pad, then ImageNet-normalize."""
    image = image.convert("RGB")
    original_w, original_h = image.size
    scale = min(size / original_w, size / original_h)
    new_w = max(1, int(round(original_w * scale)))
    new_h = max(1, int(round(original_h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], np.float32))
    arr = arr / np.array([0.229, 0.224, 0.225], np.float32)
    arr = np.transpose(arr, (2, 0, 1)).copy()
    metadata = {
        "scale": scale, "pad_x": pad_x, "pad_y": pad_y,
        "orig_w": original_w, "orig_h": original_h,
    }
    return torch.from_numpy(arr), metadata


def augment_rgb(image):
    """Photometric-only augmentation; no image/voxel geometry is changed."""
    if random.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.5:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.20))
    if random.random() < 0.35:
        image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
    return image


def resolve_image_path(npz_path: Path, data):
    """Prefer stored image_path; fall back to camera/timestamp search."""
    raw = unpack_scalar(data["image_path"]) if "image_path" in data else ""
    raw = str(raw).strip()
    if raw and raw.lower() not in {"none", "nan"}:
        p = Path(raw).expanduser()
        if p.is_file():
            return p
        # If stored paths came from a different mount, match their tail under dataset root.
        candidates = [
            DATASET_ROOT / p,
            IMAGE_ROOT / npz_path.parent.name / "images_rect" / p.name,
            IMAGE_ROOT / npz_path.parent.name / "images_rect" / f"{npz_path.stem}.jpg",
            IMAGE_ROOT / npz_path.parent.name / "images_rect" / f"{npz_path.stem}.png",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate

    camera = npz_path.parent.name
    image_dir = IMAGE_ROOT / camera / "images_rect"
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        candidate = image_dir / f"{npz_path.stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve RGB image for {npz_path}. "
        f"Checked image_path metadata and {image_dir}/<timestamp>.[jpg/png/...]."
    )


def discover_samples():
    """Pair each semantic target NPZ with its own RGB image."""
    if not NPZ_ROOT.is_dir():
        raise FileNotFoundError(
            f"Semantic target directory not found: {NPZ_ROOT}\n"
            "Edit DATASET_ROOT at the top of this script to match your cluster."
        )

    samples = []
    for camera in ("CAM2", "CAM6"):
        camera_dir = NPZ_ROOT / camera
        if not camera_dir.is_dir():
            print(f"[WARN] Missing camera directory: {camera_dir}")
            continue
        for npz_path in sorted(camera_dir.glob("*.npz")):
            try:
                with np.load(npz_path, allow_pickle=True) as data:
                    if "semantic" not in data:
                        print(f"[WARN] No semantic array: {npz_path}")
                        continue
                    image_path = resolve_image_path(npz_path, data)
                    semantic_shape = tuple(data["semantic"].shape)
                    if semantic_shape != SOURCE_SHAPE:
                        raise ValueError(
                            f"Expected source grid {SOURCE_SHAPE}, got {semantic_shape}"
                        )
                    if "bounds" in data:
                        bounds = tuple(float(v) for v in np.asarray(data["bounds"]).reshape(-1))
                        if len(bounds) == 6 and not np.allclose(bounds, SOURCE_BOUNDS, atol=1e-4):
                            raise ValueError(
                                f"NPZ bounds {bounds} do not match configured {SOURCE_BOUNDS}"
                            )
                samples.append({
                    "npz_path": npz_path,
                    "image_path": image_path,
                    "camera": camera,
                    "timestamp": int(npz_path.stem) if npz_path.stem.isdigit() else npz_path.stem,
                })
            except Exception as exc:
                print(f"[WARN] Skipping {npz_path}: {exc}")

    if not samples:
        raise RuntimeError(f"No usable RGB/semantic-NPZ pairs found under {NPZ_ROOT}")

    print(f"[DATA] Matched samples: {len(samples):,}")
    for camera in ("CAM2", "CAM6"):
        print(f"[DATA] {camera}: {sum(s['camera'] == camera for s in samples):,}")
    return samples


def temporal_split(samples, train_fraction=0.8, val_fraction=0.1):
    """Chronological split per camera to reduce near-duplicate frame leakage."""
    train, val, test = [], [], []
    for camera in ("CAM2", "CAM6"):
        items = sorted(
            [s for s in samples if s["camera"] == camera],
            key=lambda s: str(s["timestamp"]).zfill(24),
        )
        n = len(items)
        n_train = int(n * train_fraction)
        n_val = int(n * val_fraction)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    return train, val, test


def crop_target(array):
    """Validate source grid, then crop the first 30 Y slices (0 to 1.5 m)."""
    if tuple(array.shape) != SOURCE_SHAPE:
        raise ValueError(f"Bad voxel shape {array.shape}; expected {SOURCE_SHAPE}")
    result = array[:, :VOXEL_SHAPE[1], :]
    if tuple(result.shape) != VOXEL_SHAPE:
        raise RuntimeError(f"Internal crop error: {result.shape}")
    return result


class RGBToVoxelDataset(Dataset):
    def __init__(self, samples, train=False):
        self.samples = samples
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample["image_path"]) as im:
            image_pil = im.convert("RGB")
        if self.train:
            image_pil = augment_rgb(image_pil)
        image, image_meta = letterbox_image(image_pil)

        with np.load(sample["npz_path"], allow_pickle=True) as data:
            semantic = crop_target(data["semantic"].astype(np.int64))

        # Source convention: IDs 2..5 are the four classes of interest.
        # All other labels and unobserved voxels are ignored for semantics.
        target = np.full(VOXEL_SHAPE, IGNORE_INDEX, dtype=np.int64)
        foreground = (semantic >= 2) & (semantic <= 5)
        target[foreground] = semantic[foreground] - 2

        return {
            "image": image,
            "semantic": torch.from_numpy(target.copy()),
            "camera": sample["camera"],
            "timestamp": str(sample["timestamp"]),
            "npz_path": str(sample["npz_path"]),
            "image_path": str(sample["image_path"]),
            **{k: torch.tensor(v, dtype=torch.float32) for k, v in image_meta.items()},
        }


def read_matrix(lines, key, rows, cols):
    for i, line in enumerate(lines):
        if line.strip() == f"{key}:":
            values = []
            for row in lines[i + 1:i + 1 + rows]:
                values.extend(float(v) for v in row.split())
            if len(values) != rows * cols:
                raise ValueError(f"{key}: expected {rows * cols} values, got {len(values)}")
            return np.asarray(values, dtype=np.float32).reshape(rows, cols)
    raise ValueError(f"Missing {key}: in calibration file")


def load_calibration(path):
    """Read the rectified projection matrix P from the source calibration format."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Calibration file missing: {path}. Check CALIBRATION_DIR and camera filenames."
        )
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return read_matrix(lines, "P", 3, 4)[:, :3]


def make_voxel_centres():
    nx, ny, nz = LATENT_SHAPE
    xmin, xmax, ymin, ymax, zmin, zmax = BOUNDS
    xs = xmin + (torch.arange(nx, dtype=torch.float32) + 0.5) * ((xmax - xmin) / nx)
    ys = ymin + (torch.arange(ny, dtype=torch.float32) + 0.5) * ((ymax - ymin) / ny)
    zs = zmin + (torch.arange(nz, dtype=torch.float32) + 0.5) * ((zmax - zmin) / nz)
    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
    coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    normalized = coords.clone()
    normalized[:, 0] = 2 * (coords[:, 0] - xmin) / (xmax - xmin) - 1
    normalized[:, 1] = 2 * (coords[:, 1] - ymin) / (ymax - ymin) - 1
    normalized[:, 2] = 2 * (coords[:, 2] - zmin) / (zmax - zmin) - 1
    return coords, normalized


class CalibratedVoxelSampler(nn.Module):
    """Project latent voxel centres into the matching rectified RGB image."""
    def __init__(self):
        super().__init__()
        projections = torch.stack([
            torch.from_numpy(load_calibration(CALIBRATION_FILES["CAM2"])),
            torch.from_numpy(load_calibration(CALIBRATION_FILES["CAM6"])),
        ])
        self.register_buffer("K_rect", projections)
        coords, normalized = make_voxel_centres()
        self.register_buffer("coords", coords)
        self.register_buffer("coords_normalized", normalized)

    def forward(self, feature, camera_id, scale, pad_x, pad_y, orig_w, orig_h, flip=None):
        # feature: [B,C,Hf,Wf]; sample in normalized letterboxed-image coordinates.
        batch = feature.shape[0]
        coords = self.coords.unsqueeze(0).expand(batch, -1, -1)
        k = self.K_rect[camera_id]
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2].clamp_min(0.5)
        u = k[:, 0, 0:1] * x / z + k[:, 0, 2:3]
        v = k[:, 1, 1:2] * y / z + k[:, 1, 2:3]
        if flip is not None:
            u = torch.where(flip[:, None], orig_w[:, None] - 1.0 - u, u)
        u = u * scale[:, None] + pad_x[:, None]
        v = v * scale[:, None] + pad_y[:, None]
        # grid_sample coordinates are resolution-independent, so the same
        # normalized coordinates work for each feature-map resolution.
        u_grid = 2.0 * u / (IMAGE_SIZE - 1) - 1.0
        v_grid = 2.0 * v / (IMAGE_SIZE - 1) - 1.0
        grid = torch.stack([u_grid, v_grid], dim=-1).unsqueeze(1)
        sampled = F.grid_sample(
            feature, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).squeeze(2)
        nx, ny, nz = LATENT_SHAPE
        return sampled.reshape(batch, feature.shape[1], nx, ny, nz)


class ResNet50Pyramid(nn.Module):
    """ImageNet ResNet-50 feature encoder returning stride-8/16/32 maps."""
    def __init__(self, pretrained=True, channels=LATENT_CHANNELS):
        super().__init__()
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
        except Exception as exc:
            print(f"[WARN] Could not load ImageNet weights ({exc}); using random initialization.")
            backbone = models.resnet50(weights=None)
        self.conv1, self.bn1 = backbone.conv1, backbone.bn1
        self.relu, self.maxpool = backbone.relu, backbone.maxpool
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4
        self.proj2 = nn.Conv2d(512, channels, 1)
        self.proj3 = nn.Conv2d(1024, channels, 1)
        self.proj4 = nn.Conv2d(2048, channels, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.GroupNorm(8, channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels), nn.GELU(),
        )

    def forward(self, image):
        x = self.maxpool(self.relu(self.bn1(self.conv1(image))))
        x = self.layer1(x)
        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        p2 = self.proj2(c2)
        p3 = F.interpolate(self.proj3(c3), size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.proj4(c4), size=p2.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([p2, p3, p4], dim=1))
        return p2, p3, p4, fused


def conv3d_block(in_ch, out_ch, groups=8):
    g = min(groups, out_ch)
    while out_ch % g:
        g -= 1
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(g, out_ch), nn.GELU(),
        nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(g, out_ch), nn.GELU(),
    )


class ResidualBlock3D(nn.Module):
    """Two-convolution residual block used by the residual decoder."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class Image2VoxelUNet(nn.Module):
    """
    Baseline A: ResNet-50 multi-scale encoder, calibrated voxel feature lifting,
    and a compact 3D U-Net-style decoder with skip connections.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = ResNet50Pyramid(pretrained=pretrained, channels=LATENT_CHANNELS)
        self.sampler = CalibratedVoxelSampler()
        self.position = nn.Sequential(
            nn.Conv3d(3, 32, 1), nn.GELU(),
            nn.Conv3d(32, 32, 1), nn.GELU(),
        )
        self.lift_fuse = nn.Sequential(
            nn.Conv3d(LATENT_CHANNELS + 32, 128, 1, bias=False),
            nn.GroupNorm(8, 128), nn.GELU(),
        )
        self.enc0 = conv3d_block(128, 96)
        self.down1 = nn.Conv3d(96, 128, 3, stride=2, padding=1)
        self.enc1 = conv3d_block(128, 128)
        self.down2 = nn.Conv3d(128, 160, 3, stride=2, padding=1)
        self.bottleneck = conv3d_block(160, 160)
        self.up1 = conv3d_block(160 + 128, 128)
        self.up2 = conv3d_block(128 + 96, 96)
        self.semantic_head = nn.Conv3d(96, NUM_CLASSES, 1)
        self.occupancy_head = nn.Conv3d(96, 1, 1)

    def forward(self, image, camera_id, scale, pad_x, pad_y, orig_w, orig_h, flip=None):
        p2, p3, p4, _ = self.encoder(image)
        # Lift three scales separately, then fuse voxel-aligned features.
        v2 = self.sampler(p2, camera_id, scale, pad_x, pad_y, orig_w, orig_h, flip)
        v3 = self.sampler(p3, camera_id, scale, pad_x, pad_y, orig_w, orig_h, flip)
        v4 = self.sampler(p4, camera_id, scale, pad_x, pad_y, orig_w, orig_h, flip)
        image_voxels = (v2 + v3 + v4) / 3.0
        coords = self.sampler.coords_normalized.T.reshape(1, 3, *LATENT_SHAPE)
        coords = coords.expand(image.shape[0], -1, -1, -1, -1)
        x = self.lift_fuse(torch.cat([image_voxels, self.position(coords)], dim=1))
        e0 = self.enc0(x)
        e1 = self.enc1(self.down1(e0))
        b = self.bottleneck(self.down2(e1))
        u1 = F.interpolate(b, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, e1], dim=1))
        u2 = F.interpolate(u1, size=e0.shape[-3:], mode="trilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, e0], dim=1))
        # Predict at latent resolution, then upsample logits rather than applying
        # expensive 3D convolutions over all ~1.87M output voxels.
        sem = F.interpolate(self.semantic_head(u2), size=VOXEL_SHAPE, mode="trilinear", align_corners=False)
        occ = F.interpolate(self.occupancy_head(u2), size=VOXEL_SHAPE, mode="trilinear", align_corners=False)
        return occ, sem


class Image2VoxelResUNet(nn.Module):
    """
    Baseline B: multi-scale ResNet-50 voxel lifting with residual 3D processing
    and a progressive residual decoder. It differs from A in the decoder's
    residual blocks and multi-scale latent fusion.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = ResNet50Pyramid(pretrained=pretrained, channels=LATENT_CHANNELS)
        self.sampler = CalibratedVoxelSampler()
        self.scale_fuse = nn.Sequential(
            nn.Conv3d(LATENT_CHANNELS * 3, 160, 1, bias=False),
            nn.GroupNorm(8, 160), nn.GELU(),
            ResidualBlock3D(160, 160),
        )
        self.position = nn.Sequential(nn.Conv3d(3, 32, 1), nn.GELU(), nn.Conv3d(32, 32, 1))
        self.input_fuse = nn.Sequential(
            nn.Conv3d(192, 128, 1, bias=False), nn.GroupNorm(8, 128), nn.GELU()
        )
        self.core = nn.Sequential(ResidualBlock3D(128, 128), ResidualBlock3D(128, 128))
        self.up_mid = nn.Sequential(ResidualBlock3D(128, 64), ResidualBlock3D(64, 64))
        self.up_fine = nn.Sequential(ResidualBlock3D(64, 32), ResidualBlock3D(32, 32))
        self.semantic_head = nn.Conv3d(32, NUM_CLASSES, 1)
        self.occupancy_head = nn.Conv3d(32, 1, 1)

    def forward(self, image, camera_id, scale, pad_x, pad_y, orig_w, orig_h, flip=None):
        p2, p3, p4, _ = self.encoder(image)
        v2 = self.sampler(p2, camera_id, scale, pad_x, pad_y, orig_w, orig_h, flip)
        v3 = self.sampler(p3, camera_id, scale, pad_x, pad_y, orig_w, orig_h, flip)
        v4 = self.sampler(p4, camera_id, scale, pad_x, pad_y, orig_w, orig_h, flip)
        x = self.scale_fuse(torch.cat([v2, v3, v4], dim=1))
        coords = self.sampler.coords_normalized.T.reshape(1, 3, *LATENT_SHAPE)
        coords = coords.expand(image.shape[0], -1, -1, -1, -1)
        x = self.input_fuse(torch.cat([x, self.position(coords)], dim=1))
        x = self.core(x)
        # Progressive upsampling with residual blocks limits memory use.
        x = F.interpolate(x, size=(80, 16, 40), mode="trilinear", align_corners=False)
        x = self.up_mid(x)
        x = F.interpolate(x, size=(160, 30, 98), mode="trilinear", align_corners=False)
        x = self.up_fine(x)
        sem = F.interpolate(self.semantic_head(x), size=VOXEL_SHAPE, mode="trilinear", align_corners=False)
        occ = F.interpolate(self.occupancy_head(x), size=VOXEL_SHAPE, mode="trilinear", align_corners=False)
        return occ, sem


def compute_class_weights(train_samples):
    """Square-root inverse-frequency weights estimated from training labels only."""
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for sample in train_samples:
        with np.load(sample["npz_path"], allow_pickle=True) as data:
            labels = crop_target(data["semantic"].astype(np.int64))
        valid = (labels >= 2) & (labels <= 5)
        if valid.any():
            counts += np.bincount(labels[valid] - 2, minlength=NUM_CLASSES)
    freq = counts / max(counts.sum(), 1.0)
    nonzero = freq[freq > 0]
    if not len(nonzero):
        raise RuntimeError("Training split has no target-class voxels.")
    weights = np.sqrt(np.median(nonzero) / np.maximum(freq, 1e-12))
    weights = np.clip(weights, 0.5, 5.0)
    weights /= weights.mean()
    print("[LOSS] class voxel counts:", counts.astype(np.int64).tolist())
    print("[LOSS] class weights:", np.round(weights, 3).tolist())
    return torch.tensor(weights, dtype=torch.float32)


def semantic_loss(logits, target, class_weights):
    """Weighted CE + soft Dice over foreground-labelled voxels only."""
    valid = target != IGNORE_INDEX
    if not valid.any():
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits, target, weight=class_weights, ignore_index=IGNORE_INDEX)
    probs = F.softmax(logits, dim=1)
    safe_target = target.clamp(0, NUM_CLASSES - 1)
    onehot = F.one_hot(safe_target, NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
    mask = valid.unsqueeze(1).float()
    probs, onehot = probs * mask, onehot * mask
    dims = (0, 2, 3, 4)
    intersection = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * intersection + 1.0) / (denom + 1.0)
    present = onehot.sum(dims) > 0
    dice_loss = (1.0 - dice[present]).mean() if present.any() else logits.sum() * 0.0
    return ce + dice_loss


def occupancy_target_from_semantic(target):
    """Foreground=1, all other/unknown voxels=0, matching supplied V10.1 policy."""
    return (target != IGNORE_INDEX).float()


def occupancy_loss(occ_logits, target):
    """Balanced binary loss for sparse foreground occupancy."""
    occ_target = occupancy_target_from_semantic(target)
    logits = occ_logits.squeeze(1)
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, occ_target, reduction="none")
    pos = occ_target > 0.5
    neg = ~pos
    pos_loss = bce[pos].mean() if pos.any() else logits.sum() * 0.0
    neg_loss = (bce[neg] * prob[neg].detach().clamp_min(0.05)).mean() if neg.any() else logits.sum() * 0.0
    return 2.0 * pos_loss + neg_loss


def update_confusion(cm, sem_logits, occ_logits, target):
    """Accumulate a 4-class confusion matrix; occupancy-empty predictions count as FN."""
    sem_pred = sem_logits.argmax(dim=1)
    occ_pred = torch.sigmoid(occ_logits.squeeze(1)) >= OCCUPANCY_THRESHOLD
    valid = target != IGNORE_INDEX
    # A predicted empty voxel cannot be counted as any of the semantic classes.
    # Build a 4x5 matrix: final prediction column represents empty.
    pred = sem_pred.clone()
    pred[~occ_pred] = NUM_CLASSES
    t = target[valid]
    p = pred[valid]
    cm += torch.bincount(
        t * (NUM_CLASSES + 1) + p,
        minlength=NUM_CLASSES * (NUM_CLASSES + 1)
    ).reshape(NUM_CLASSES, NUM_CLASSES + 1)
    return cm


def metrics_from_cm(cm):
    cm = cm.float()
    tp = torch.diag(cm[:, :NUM_CLASSES])
    fp = cm[:, :NUM_CLASSES].sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    iou = tp / (tp + fp + fn).clamp_min(1.0)
    support = cm.sum(dim=1)
    iou[support == 0] = float("nan")
    miou = float(torch.nanmean(iou).item()) if torch.isfinite(iou).any() else 0.0
    return miou, iou.detach().cpu().numpy()


def move_batch(batch, device):
    moved = {}
    for key in ("image", "semantic", "scale", "pad_x", "pad_y", "orig_w", "orig_h"):
        moved[key] = batch[key].to(device, non_blocking=True)
    moved["camera_id"] = torch.tensor(
        [0 if c == "CAM2" else 1 for c in batch["camera"]],
        device=device, dtype=torch.long
    )
    return moved


def run_epoch(model, loader, optimizer, class_weights, device, train, epoch, scaler):
    model.train(train)
    running_loss, running_sem, running_occ, batches = 0.0, 0.0, 0.0, 0
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES + 1, dtype=torch.long, device=device)
    for batch_idx, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)

        amp_enabled = USE_AMP and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            occ_logits, sem_logits = model(
                batch["image"], batch["camera_id"], batch["scale"], batch["pad_x"],
                batch["pad_y"], batch["orig_w"], batch["orig_h"]
            )
            sem = semantic_loss(sem_logits.float(), batch["semantic"], class_weights)
            occ = occupancy_loss(occ_logits.float(), batch["semantic"])
            loss = sem + 0.5 * occ

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        running_loss += float(loss.detach())
        running_sem += float(sem.detach())
        running_occ += float(occ.detach())
        batches += 1
        update_confusion(cm, sem_logits.detach().float(), occ_logits.detach().float(), batch["semantic"])

        if batch_idx % 20 == 0:
            phase = "TRAIN" if train else "VAL"
            print(f"[{phase}] epoch={epoch:03d} batch={batch_idx:04d}/{len(loader):04d} "
                  f"loss={float(loss.detach()):.4f} sem={float(sem.detach()):.4f} "
                  f"occ={float(occ.detach()):.4f}", flush=True)

    miou, ious = metrics_from_cm(cm)
    denom = max(batches, 1)
    return {
        "loss": running_loss / denom,
        "semantic_loss": running_sem / denom,
        "occupancy_loss": running_occ / denom,
        "miou": miou,
        "ious": ious.tolist(),
    }


def save_checkpoint(path, model, optimizer, epoch, metrics, model_name):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "model_name": model_name,
        "class_names": CLASS_NAMES,
        "voxel_shape": VOXEL_SHAPE,
        "voxel_size": VOXEL_SIZE,
        "bounds": BOUNDS,
        "source_voxel_shape": SOURCE_SHAPE,
        "source_bounds": SOURCE_BOUNDS,
        "image_size": IMAGE_SIZE,
        "rgb_only": True,
    }, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["image2voxel_unet", "image2voxel_resunet"], required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0, help="Optional small smoke-test limit; 0 uses all data.")
    args = parser.parse_args()

    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this script on a CUDA-enabled node.")
    device = torch.device("cuda")
    print("=" * 88)
    print(f"RGB -> SEMANTIC VOXELS | model={args.model}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Voxel shape: {VOXEL_SHAPE}; bounds={BOUNDS}; voxel={VOXEL_SIZE} m")
    print("Input modalities: RGB only (no RF-DETR prior, no LiDAR)")
    print("=" * 88)

    samples = discover_samples()
    train_samples, val_samples, test_samples = temporal_split(samples)
    if args.limit_train > 0:
        train_samples = train_samples[:args.limit_train]
        val_samples = val_samples[:max(1, min(len(val_samples), args.limit_train // 4 or 1))]
        test_samples = []
        print("[SMOKE TEST] Limited samples; no held-out test evaluation.")
    print(f"[SPLIT] train={len(train_samples):,} val={len(val_samples):,} test={len(test_samples):,}")
    if not train_samples or not val_samples:
        raise RuntimeError("Train/validation split is empty.")

    output_root = DATASET_ROOT / f"comparison_{args.model}_rgb_only"
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "dataset_split.json", "w") as f:
        json.dump({
            "train": [str(s["npz_path"]) for s in train_samples],
            "val": [str(s["npz_path"]) for s in val_samples],
            "test": [str(s["npz_path"]) for s in test_samples],
        }, f, indent=2)

    train_ds = RGBToVoxelDataset(train_samples, train=True)
    val_ds = RGBToVoxelDataset(val_samples, train=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )

    pretrained = not args.no_pretrained
    if args.model == "image2voxel_unet":
        model = Image2VoxelUNet(pretrained=pretrained)
    else:
        model = Image2VoxelResUNet(pretrained=pretrained)
    model = model.to(device)
    class_weights = compute_class_weights(train_samples).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01
    )
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "training_log.csv"
    columns = [
        "epoch", "lr", "train_loss", "train_semantic_loss", "train_occupancy_loss",
        "train_miou", "val_loss", "val_semantic_loss", "val_occupancy_loss", "val_miou",
        *[f"val_iou_{name}" for name in CLASS_NAMES],
    ]
    best_miou = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 28} EPOCH {epoch}/{args.epochs} {'=' * 28}")
        train_metrics = run_epoch(model, train_loader, optimizer, class_weights, device, True, epoch, None)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, class_weights, device, False, epoch, None)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"[SUMMARY] train loss={train_metrics['loss']:.5f} mIoU={train_metrics['miou']:.4f}")
        print(f"[SUMMARY] val   loss={val_metrics['loss']:.5f} mIoU={val_metrics['miou']:.4f}")
        for name, iou in zip(CLASS_NAMES, val_metrics["ious"]):
            print(f"  val IoU {name:20s}: {iou:.4f}")

        metrics = {"epoch": epoch, "lr": lr, "train": train_metrics, "val": val_metrics}
        history.append(metrics)
        row = {
            "epoch": epoch, "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_semantic_loss": train_metrics["semantic_loss"],
            "train_occupancy_loss": train_metrics["occupancy_loss"],
            "train_miou": train_metrics["miou"],
            "val_loss": val_metrics["loss"],
            "val_semantic_loss": val_metrics["semantic_loss"],
            "val_occupancy_loss": val_metrics["occupancy_loss"],
            "val_miou": val_metrics["miou"],
        }
        row.update({f"val_iou_{name}": iou for name, iou in zip(CLASS_NAMES, val_metrics["ious"])})
        write_header = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        save_checkpoint(checkpoint_dir / "latest.pth", model, optimizer, epoch, metrics, args.model)
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            save_checkpoint(checkpoint_dir / "best_val_miou.pth", model, optimizer, epoch, metrics, args.model)
            print(f"[CHECKPOINT] New best validation mIoU={best_miou:.4f}")

        with open(output_root / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print("\nTraining finished.")
    print(f"Best checkpoint: {checkpoint_dir / 'best_val_miou.pth'}")
    print(f"Epoch metrics:   {log_path}")
    print(f"Split manifest:  {output_root / 'dataset_split.json'}")
    print("The test set was held out and was not used for checkpoint selection.")


if __name__ == "__main__":
    main()
