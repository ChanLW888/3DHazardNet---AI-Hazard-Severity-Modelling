
#!/usr/bin/env python3
"""
3DHazardNet V10.1: full-dataset inference with safe resume.

- Uses EXACT V10.1 find_samples() indexing.
- Defaults to 0..len(samples)-1, inclusive.
- Skips existing, verified predictions.
- Keeps original NPZ, PNG and JSON outputs.
- Merges old and new frames into one CSV manifest and video.
- Camera-only inference: RGB + RF-DETR, no LiDAR.
"""

import argparse
import csv
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/hazard_test_101"
)

CHECKPOINT_PATH = (
    DATASET_ROOT
    / "rgb_voxel_training_v10_1_5cm_xm8_8_y0_15_z025_10_dense_latent_hazard_empty"
    / "checkpoints/best_full.pth"
)

RFDETR_CHECKPOINT = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/"
    "2D-Seg-Road-Nissan.v13i.coco-segmentation/"
    "outputs/checkpoint_best_total.pth"
)

DEFAULT_OUTPUT = DATASET_ROOT / "v10_1_sequence_inference"

START_INDEX = 0
END_INDEX = None

RFDETR_CONFIDENCE_THRESHOLD = 0.30
DEFAULT_OCCUPANCY_THRESHOLD = 0.50

CLASS_NAMES = [
    "loose_trash",
    "mud_spill",
    "public_greenway",
    "sidewalk",
]

# OpenCV BGR colours.
CLASS_COLORS = [
    (0, 0, 255),
    (0, 165, 255),
    (0, 180, 0),
    (230, 170, 80),
]

EMPTY_LABEL = 255
IGNORE_INDEX = -1
GT_BACKGROUND_VIS_LABEL = 4
GT_BACKGROUND_COLOR = (140, 140, 140)

PANEL_W = 480
PANEL_H = 360
VIDEO_SIZE = (PANEL_W * 3, PANEL_H)


# ============================================================
# ORIGINAL V10.1 MODEL IMPORT
# ============================================================

def load_model_source(script_path):
    script_path = Path(script_path).expanduser().resolve()

    if not script_path.is_file():
        raise FileNotFoundError(
            f"V10.1 source not found: {script_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "v10_1_training_source",
        str(script_path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import V10.1 source: {script_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = (
        "find_samples",
        "crop_to_target_grid",
        "letterbox_image",
        "RGBPriorToSemanticVoxelModelV10",
        "VOXEL_SHAPE",
        "SOURCE_VOXEL_SHAPE",
        "BOUNDS",
        "VOXEL_SIZE",
        "IMAGE_SIZE",
        "PRIOR_SIZE",
    )

    missing = [
        name for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise RuntimeError(
            f"Missing V10.1 definitions: {missing}"
        )

    return module


# ============================================================
# FILE UTILITIES
# ============================================================

def sample_folder(output_dir, index, camera):
    return (
        output_dir
        / f"sample_{index:05d}_{camera}"
    )


def atomic_npz(path, **arrays):
    tmp = path.with_name(path.name + ".partial")

    try:
        with tmp.open("wb") as f:
            np.savez_compressed(f, **arrays)

        os.replace(tmp, path)

    finally:
        tmp.unlink(missing_ok=True)


def atomic_json(path, payload):
    tmp = path.with_name(path.name + ".partial")

    try:
        tmp.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        os.replace(tmp, path)

    finally:
        tmp.unlink(missing_ok=True)


def atomic_copy(src, dst):
    tmp = dst.with_name(dst.name + ".partial")

    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)

    finally:
        tmp.unlink(missing_ok=True)


# ============================================================
# FULL DATASET DISCOVERY
# ============================================================

def get_selected(v10, args):
    samples = v10.find_samples()

    if not samples:
        raise RuntimeError(
            "V10.1 find_samples() returned no samples."
        )

    end = (
        len(samples) - 1
        if args.end is None
        else args.end
    )

    if (
        args.start < 0
        or args.start >= len(samples)
        or end < args.start
        or end >= len(samples)
    ):
        raise ValueError(
            f"Invalid indices: start={args.start}, "
            f"end={end}, total={len(samples)}"
        )

    print("=" * 70)
    print("3DHAZARDNET FULL DATASET INFERENCE")
    print("=" * 70)

    print(f"Total dataset samples: {len(samples)}")
    print(f"Start index: {args.start}")
    print(f"End index: {end}")
    print(f"Selected samples: {end - args.start + 1}")

    selected = [
        (i, samples[i])
        for i in range(args.start, end + 1)
    ]

    return selected, len(samples), end


# ============================================================
# CHECK ALREADY INFERRED SAMPLES
# ============================================================

def check_existing(folder, sample, index, v10, args):
    """
    A prediction is considered complete only when:

    1. prediction.npz exists.
    2. metadata.json exists.
    3. Sample index/camera/timestamp match.
    4. Source image and GT paths match.
    5. Checkpoint and occupancy threshold match.
    6. Voxel dimensions and labels are valid.

    This prevents accidentally skipping incorrect results.
    """

    pred_path = folder / "prediction.npz"
    meta_path = folder / "metadata.json"

    if not pred_path.is_file() or not meta_path.is_file():
        return False, "prediction or metadata missing"

    try:
        meta = json.loads(
            meta_path.read_text(encoding="utf-8")
        )

        if int(meta["sample_index"]) != index:
            return False, "sample index mismatch"

        if str(meta["camera"]) != str(sample["camera"]):
            return False, "camera mismatch"

        if str(meta["timestamp"]) != str(sample["timestamp"]):
            return False, "timestamp mismatch"

        old_image = Path(meta["original_image"]).resolve()
        new_image = Path(sample["image_path"]).resolve()

        if old_image != new_image:
            return False, "image path mismatch"

        old_gt = Path(meta["source_gt"]).resolve()
        new_gt = Path(sample["npz_path"]).resolve()

        if old_gt != new_gt:
            return False, "GT path mismatch"

        old_ckpt = Path(meta["checkpoint"]).resolve()
        new_ckpt = Path(args.checkpoint).resolve()

        if old_ckpt != new_ckpt:
            return False, "checkpoint mismatch"

        if not np.isclose(
            float(meta["occupancy_threshold"]),
            args.occupancy_threshold,
        ):
            return False, "occupancy threshold mismatch"

        if tuple(meta["voxel_shape"]) != tuple(v10.VOXEL_SHAPE):
            return False, "voxel shape mismatch"

        if not np.isclose(
            float(meta["voxel_size"]),
            float(v10.VOXEL_SIZE),
        ):
            return False, "voxel size mismatch"

        if not np.allclose(meta["bounds"], v10.BOUNDS):
            return False, "bounds mismatch"

        with np.load(
            pred_path,
            allow_pickle=False,
        ) as data:

            required = {
                "semantic",
                "occupancy",
                "occupancy_binary",
                "semantic_confidence",
                "semantic_probabilities",
            }

            if not required.issubset(data.files):
                return False, "incomplete prediction arrays"

            semantic = data["semantic"]
            binary = data["occupancy_binary"]

            if (
                semantic.shape != tuple(v10.VOXEL_SHAPE)
                or binary.shape != semantic.shape
            ):
                return False, "incorrect voxel dimensions"

            valid_labels = (
                (semantic <= 3)
                | (semantic == EMPTY_LABEL)
            )

            if not np.all(valid_labels):
                return False, "invalid semantic labels"

            if not np.array_equal(
                binary > 0,
                semantic != EMPTY_LABEL,
            ):
                return False, "occupancy/semantic disagree"

        return True, "verified"

    except Exception as exc:
        return False, f"invalid saved result: {exc}"


# ============================================================
# RF-DETR PRIORS
# ============================================================

def normalise_mask(mask, height, width):
    import torch

    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    mask = np.squeeze(np.asarray(mask))

    if mask.dtype != np.bool_:
        mask = mask > 0.5

    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    return mask


def detections_to_prior(detections, height, width):
    """
    Original RF-DETR channels:

    0 background
    1 fallen_fence
    2 loose_trash
    3 mud_spill
    4 public_greenway
    5 sidewalk
    """

    prior = np.zeros(
        (6, height, width),
        dtype=np.float32,
    )

    assigned = np.zeros(
        (height, width),
        dtype=bool,
    )

    ids = detections.class_id
    confidence = detections.confidence
    masks = detections.mask

    order = sorted(
        range(len(ids)),
        key=lambda i: float(confidence[i]),
        reverse=True,
    )

    for i in order:
        score = float(confidence[i])
        cls = int(ids[i])

        if score < RFDETR_CONFIDENCE_THRESHOLD:
            continue

        if cls not in range(5):
            continue

        available = (
            normalise_mask(
                masks[i],
                height,
                width,
            )
            & ~assigned
        )

        prior[cls + 1, available] = score
        assigned[available] = True

    prior[0] = 1.0 - prior[1:].max(axis=0)

    return prior


def letterbox_prior(prior, target_size):
    import torch
    import torch.nn.functional as F

    _, h, w = prior.shape

    scale = min(
        target_size / w,
        target_size / h,
    )

    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    x = torch.from_numpy(
        prior.astype(np.float32)
    ).unsqueeze(0)

    x = F.interpolate(
        x,
        size=(nh, nw),
        mode="bilinear",
        align_corners=False,
    )

    padded = torch.zeros(
        (1, 6, target_size, target_size),
        dtype=torch.float32,
    )

    px = (target_size - nw) // 2
    py = (target_size - nh) // 2

    padded[
        :,
        :,
        py:py + nh,
        px:px + nw,
    ] = x

    return padded.squeeze(0).numpy()


def generate_rfdetr_prior(rgb, detector, prior_size):
    import torch

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    with torch.inference_mode():
        detections = detector.predict([bgr])[0]

    full = letterbox_prior(
        detections_to_prior(
            detections,
            *rgb.shape[:2],
        ),
        prior_size,
    )

    prior = full[2:6].copy()

    expected = (
        4,
        prior_size,
        prior_size,
    )

    if prior.shape != expected:
        raise RuntimeError(
            f"Unexpected RF-DETR prior shape: {prior.shape}"
        )

    return prior, full


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth(sample, v10):
    with np.load(
        sample["npz_path"],
        allow_pickle=False,
    ) as data:

        raw_sem = np.asarray(data["semantic"])
        raw_occ = np.asarray(data["occupancy"])

    if tuple(raw_sem.shape) != tuple(
        v10.SOURCE_VOXEL_SHAPE
    ):
        raise ValueError(
            f"Bad source GT shape {raw_sem.shape} "
            f"for {sample['npz_path']}"
        )

    if raw_occ.shape != raw_sem.shape:
        raise ValueError(
            "Source GT semantic/occupancy dimensions disagree"
        )

    sem = v10.crop_to_target_grid(raw_sem)

    observed = (
        v10.crop_to_target_grid(raw_occ) > 0
    )

    if tuple(sem.shape) != tuple(v10.VOXEL_SHAPE):
        raise ValueError(
            f"Cropped GT shape mismatch: {sem.shape}"
        )

    fg = (
        observed
        & (sem >= 2)
        & (sem <= 5)
    )

    bg = (
        observed
        & (sem == 0)
    )

    labels = np.full(
        sem.shape,
        IGNORE_INDEX,
        dtype=np.int64,
    )

    labels[fg] = sem[fg] - 2

    replay = np.full(
        sem.shape,
        EMPTY_LABEL,
        dtype=np.uint8,
    )

    replay[fg] = (
        sem[fg] - 2
    ).astype(np.uint8)

    visual = replay.copy()
    visual[bg] = GT_BACKGROUND_VIS_LABEL

    return {
        "semantic": labels,
        "semantic_replay": replay,
        "semantic_visual": visual,
        "occupancy": fg.astype(np.uint8),
        "observed": observed.astype(np.uint8),
        "background_observed": bg.astype(np.uint8),
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    occupied,
    pred_semantic,
    gt,
    semantic_argmax,
):
    pred_occ = occupied > 0
    gt_occ = gt["occupancy"] > 0

    tp = int(np.count_nonzero(
        pred_occ & gt_occ
    ))

    fp = int(np.count_nonzero(
        pred_occ & ~gt_occ
    ))

    fn = int(np.count_nonzero(
        ~pred_occ & gt_occ
    ))

    precision = (
        tp / (tp + fp)
        if tp + fp else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    valid = (
        gt["semantic"] != IGNORE_INDEX
    )

    cm = np.bincount(
        gt["semantic"][valid] * 4
        + semantic_argmax[valid],
        minlength=16,
    ).reshape(4, 4)

    ious = []

    for cls in range(4):
        intersection = int(cm[cls, cls])

        union = int(
            cm[cls, :].sum()
            + cm[:, cls].sum()
            - intersection
        )

        if cm[cls, :].sum():
            ious.append(
                intersection / max(union, 1)
            )
        else:
            ious.append(None)

    gated = []

    for cls in range(4):
        p = pred_semantic == cls
        g = gt["semantic_replay"] == cls

        u = np.count_nonzero(p | g)

        gated.append(
            float(np.count_nonzero(p & g) / u)
            if u else None
        )

    def average(seq):
        values = [
            v for v in seq
            if v is not None
        ]

        return (
            float(np.mean(values))
            if values else None
        )

    return {
        "occupancy_tp": tp,
        "occupancy_fp": fp,
        "occupancy_fn": fn,
        "occupancy_precision": precision,
        "occupancy_recall": recall,
        "occupancy_f1": f1,
        "gt_occupied_voxels": int(gt_occ.sum()),
        "pred_occupied_voxels": int(pred_occ.sum()),
        "gt_observed_background_voxels": int(
            gt["background_observed"].sum()
        ),
        "semantic_miou_gt_foreground": average(ious),
        "small_hazard_miou_gt_foreground": average(
            ious[:2]
        ),
        "gated_full_grid_miou": average(gated),
        "semantic_ious": dict(zip(CLASS_NAMES, ious)),
        "gated_full_grid_ious": dict(
            zip(CLASS_NAMES, gated)
        ),
    }


# ============================================================
# VISUALISATION
# ============================================================

def make_bev(semantic, include_gt_background=False):
    if semantic.ndim != 3:
        raise ValueError(
            f"Expected [X,Y,Z], got {semantic.shape}"
        )

    nx, _, nz = semantic.shape

    bev = np.full(
        (nz, nx, 3),
        255,
        dtype=np.uint8,
    )

    if include_gt_background:
        mask = np.any(
            semantic == GT_BACKGROUND_VIS_LABEL,
            axis=1,
        ).T

        bev[mask] = GT_BACKGROUND_COLOR

    for cls in (2, 3, 1, 0):
        mask = np.any(
            semantic == cls,
            axis=1,
        ).T

        bev[mask] = CLASS_COLORS[cls]

    bev = cv2.flip(bev, 0)

    bev = cv2.resize(
        bev,
        (PANEL_W, PANEL_H),
        interpolation=cv2.INTER_NEAREST,
    )

    cv2.rectangle(
        bev,
        (0, 0),
        (PANEL_W - 1, PANEL_H - 1),
        (210, 210, 210),
        1,
    )

    return bev


def add_heading(im, label):
    im = im.copy()

    cv2.rectangle(
        im,
        (0, 0),
        (im.shape[1], 45),
        (247, 247, 247),
        -1,
    )

    cv2.putText(
        im,
        label,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.61,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    return im


def make_comparison(
    rgb,
    pred,
    gt_visual,
    index,
    camera,
):
    rgb_panel = cv2.resize(
        cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        ),
        (PANEL_W, PANEL_H),
        interpolation=cv2.INTER_AREA,
    )

    return np.hstack([
        add_heading(
            rgb_panel,
            f"RGB | {camera} | Index {index}",
        ),
        add_heading(
            make_bev(pred),
            "3DHazardNet Prediction",
        ),
        add_heading(
            make_bev(gt_visual, True),
            "Ground Truth + Background",
        ),
    ])


# ============================================================
# SAVE COMPLETED INFERENCE
# ============================================================

def save_sample(
    output_dir,
    index,
    sample,
    rgb,
    gt,
    result,
    prior,
    full_prior,
    metrics,
    inference_seconds,
    args,
    v10,
):
    folder = sample_folder(
        output_dir,
        index,
        sample["camera"],
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Invalidate previous completion marker before overwriting.
    (folder / "metadata.json").unlink(missing_ok=True)

    image_path = Path(sample["image_path"])

    original_dest = (
        folder
        / f"image_original{image_path.suffix}"
    )

    atomic_copy(
        image_path,
        original_dest,
    )

    atomic_copy(
        Path(sample["npz_path"]),
        folder / "gt_original.npz",
    )

    bounds = np.asarray(
        v10.BOUNDS,
        dtype=np.float32,
    )

    origin = bounds[[0, 2, 4]]

    # Original GT, cropped to the V10.1 target grid.
    atomic_npz(
        folder / "gt_target.npz",
        semantic=gt["semantic"],
        semantic_replay=gt["semantic_replay"],
        semantic_visual=gt["semantic_visual"],
        occupancy=gt["occupancy"],
        observed=gt["observed"],
        background_observed=gt["background_observed"],
        supervision_mask=np.ones(
            v10.VOXEL_SHAPE,
            dtype=np.uint8,
        ),
        voxel_size=np.float32(v10.VOXEL_SIZE),
        bounds=bounds,
        origin=origin,
        camera=np.array(sample["camera"]),
        timestamp=np.int64(sample["timestamp"]),
        class_names=np.asarray(CLASS_NAMES),
        visual_class_names=np.asarray(
            CLASS_NAMES + ["background"]
        ),
        ignore_index=np.int64(IGNORE_INDEX),
        empty_label=np.uint8(EMPTY_LABEL),
        gt_background_visual_label=np.uint8(
            GT_BACKGROUND_VIS_LABEL
        ),
    )

    # Exact prediction format for downstream processing.
    atomic_npz(
        folder / "prediction.npz",
        semantic=result["semantic"],
        occupancy=result["occupancy"],
        occupancy_binary=result["occupancy_binary"],
        semantic_confidence=result["semantic_confidence"],
        semantic_probabilities=result["semantic_probabilities"],
        voxel_size=np.float32(v10.VOXEL_SIZE),
        bounds=bounds,
        origin=origin,
        camera=np.array(sample["camera"]),
        timestamp=np.int64(sample["timestamp"]),
        class_names=np.asarray(CLASS_NAMES),
        occupancy_threshold=np.float32(
            args.occupancy_threshold
        ),
    )

    # RF-DETR priors.
    atomic_npz(
        folder / "rfdetr_prior.npz",
        prior=prior.astype(np.float32),
        full_rfdetr_prior=full_prior.astype(np.float32),
        channel_names=np.asarray(CLASS_NAMES),
        full_channel_names=np.asarray(
            ["background", "fallen_fence"] + CLASS_NAMES
        ),
        model_input_channels=np.array(
            [2, 3, 4, 5],
            dtype=np.int32,
        ),
        camera=np.array(sample["camera"]),
        timestamp=np.int64(sample["timestamp"]),
    )

    comparison_path = folder / "comparison.png"

    frame = make_comparison(
        rgb,
        result["semantic"],
        gt["semantic_visual"],
        index,
        sample["camera"],
    )

    if not cv2.imwrite(
        str(comparison_path),
        frame,
    ):
        raise OSError(
            f"Could not save {comparison_path}"
        )

    meta = {
        "sample_index": int(index),
        "camera": str(sample["camera"]),
        "timestamp": int(sample["timestamp"]),
        "original_image": str(
            Path(sample["image_path"]).resolve()
        ),
        "source_gt": str(
            Path(sample["npz_path"]).resolve()
        ),
        "checkpoint": str(
            Path(args.checkpoint).expanduser().resolve()
        ),
        "inference_seconds": float(inference_seconds),
        "voxel_shape": list(
            map(int, v10.VOXEL_SHAPE)
        ),
        "voxel_size": float(v10.VOXEL_SIZE),
        "bounds": list(
            map(float, v10.BOUNDS)
        ),
        "occupancy_threshold": float(
            args.occupancy_threshold
        ),
        "lidar_used": False,
        "cached_prior_used_for_inference": False,
        "gt_background_included": True,
        "gt_background_raw_class": 0,
        "gt_background_visual_class": GT_BACKGROUND_VIS_LABEL,
        "metrics": metrics,
        "files": {
            "rgb": str(original_dest),
            "gt_original": str(
                folder / "gt_original.npz"
            ),
            "gt_target": str(
                folder / "gt_target.npz"
            ),
            "prediction": str(
                folder / "prediction.npz"
            ),
            "rfdetr_prior": str(
                folder / "rfdetr_prior.npz"
            ),
            "comparison": str(comparison_path),
        },
    }

    # Written LAST: marks a successfully completed sample.
    atomic_json(
        folder / "metadata.json",
        meta,
    )

    return meta


# ============================================================
# INITIALISE 3DHAZARDNET AND RF-DETR
# ============================================================

def init_models(v10, args):
    import torch
    from rfdetr import RFDETRSegSmall

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable: use a GPU node."
        )

    for path in (
        args.checkpoint,
        args.rfdetr_checkpoint,
    ):
        if not Path(path).expanduser().is_file():
            raise FileNotFoundError(
                f"Missing checkpoint: {path}"
            )

    if (
        hasattr(v10, "seed_everything")
        and hasattr(v10, "SEED")
    ):
        v10.seed_everything(v10.SEED)

    v10.USE_PRETRAINED = False

    model = (
        v10.RGBPriorToSemanticVoxelModelV10()
        .to("cuda")
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "3DHazardNet checkpoint missing model_state_dict"
        )

    if "voxel_shape" in checkpoint:
        if tuple(checkpoint["voxel_shape"]) != tuple(
            v10.VOXEL_SHAPE
        ):
            raise RuntimeError(
                "Checkpoint voxel shape mismatch"
            )

    if "bounds" in checkpoint:
        if not np.allclose(
            checkpoint["bounds"],
            v10.BOUNDS,
        ):
            raise RuntimeError(
                "Checkpoint bounds mismatch"
            )

    if "voxel_size" in checkpoint:
        if not np.isclose(
            checkpoint["voxel_size"],
            v10.VOXEL_SIZE,
        ):
            raise RuntimeError(
                "Checkpoint voxel size mismatch"
            )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    epoch = checkpoint.get("epoch")
    del checkpoint

    model.eval()

    detector = RFDETRSegSmall.from_checkpoint(
        str(args.rfdetr_checkpoint)
    )

    detector.inference(
        dtype=torch.float16,
        batch_size=1,
    )

    amp_enabled = (
        bool(getattr(v10, "USE_AMP", True))
        and not args.no_amp
        and torch.cuda.is_bf16_supported()
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"3DHazardNet epoch: {epoch}")
    print(f"BF16 AMP: {amp_enabled}")

    return (
        torch,
        model,
        detector,
        amp_enabled,
        epoch,
    )


# ============================================================
# INFER ONE SAMPLE
# ============================================================

def infer_one(
    sample,
    index,
    v10,
    torch,
    model,
    detector,
    amp_enabled,
    args,
):
    from PIL import Image

    tic = time.perf_counter()

    with Image.open(sample["image_path"]) as img:
        image_pil = img.convert("RGB")

        rgb = np.asarray(
            image_pil
        ).copy()

        (
            image_tensor,
            scale_value,
            pad_x_value,
            pad_y_value,
            original_width,
            original_height,
        ) = v10.letterbox_image(
            image_pil,
            v10.IMAGE_SIZE,
        )

    gt = load_ground_truth(
        sample,
        v10,
    )

    prior, full_prior = generate_rfdetr_prior(
        rgb,
        detector,
        v10.PRIOR_SIZE,
    )

    device = "cuda"

    def f32(value):
        return torch.tensor(
            [value],
            device=device,
            dtype=torch.float32,
        )

    camera_id = torch.tensor(
        [
            0
            if sample["camera"] == "CAM2"
            else 1
        ],
        device=device,
        dtype=torch.long,
    )

    flip = torch.tensor(
        [False],
        device=device,
        dtype=torch.bool,
    )

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            (
                occ_logits,
                sem_logits,
                _,
            ) = model(
                image_tensor.unsqueeze(0).to(device),
                torch.from_numpy(prior)
                .unsqueeze(0)
                .to(device),
                camera_id,
                flip,
                f32(scale_value),
                f32(pad_x_value),
                f32(pad_y_value),
                f32(original_width),
                f32(original_height),
                lidar_prior=None,
                lidar_present=None,
            )

            occ = torch.sigmoid(
                occ_logits[0, 0]
            )

            sem_probs = torch.softmax(
                sem_logits[0],
                dim=0,
            )

            argmax = sem_probs.argmax(
                dim=0
            ).to(torch.uint8)

            occupied = (
                occ >= args.occupancy_threshold
            )

            semantic = argmax.masked_fill(
                ~occupied,
                EMPTY_LABEL,
            )

            result = {
                "semantic": (
                    semantic.cpu().numpy()
                ),
                "occupancy": (
                    occ.float().cpu().numpy()
                ),
                "occupancy_binary": (
                    occupied.to(torch.uint8)
                    .cpu().numpy()
                ),
                "semantic_confidence": (
                    sem_probs.max(dim=0)
                    .values.float().cpu().numpy()
                ),
                "semantic_probabilities": (
                    sem_probs.float().cpu().numpy()
                ),
            }

            argmax_np = argmax.cpu().numpy()

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter() - tic
    )

    if tuple(
        result["semantic"].shape
    ) != tuple(v10.VOXEL_SHAPE):
        raise RuntimeError(
            "Wrong inferred voxel shape: "
            f"{result['semantic'].shape}"
        )

    metrics = calculate_metrics(
        result["occupancy_binary"],
        result["semantic"],
        gt,
        argmax_np,
    )

    return (
        rgb,
        gt,
        result,
        prior,
        full_prior,
        metrics,
        elapsed,
    )


# ============================================================
# RESTORE MISSING COMPARISON IMAGES
# ============================================================

def regenerate_comparison(folder, sample, index, v10):
    """
    Rebuild comparison.png from already-saved predictions.
    Neither 3DHazardNet nor RF-DETR is run.
    """
    from PIL import Image

    rgb_candidates = sorted(
        folder.glob("image_original.*")
    )

    rgb_path = (
        rgb_candidates[0]
        if rgb_candidates
        else Path(sample["image_path"])
    )

    with Image.open(rgb_path) as img:
        rgb = np.asarray(
            img.convert("RGB")
        )

    with np.load(
        folder / "prediction.npz",
        allow_pickle=False,
    ) as data:
        pred = np.asarray(
            data["semantic"],
            dtype=np.uint8,
        )

    gt_path = folder / "gt_target.npz"

    if gt_path.is_file():
        with np.load(
            gt_path,
            allow_pickle=False,
        ) as data:
            gt_visual = (
                data["semantic_visual"]
                if "semantic_visual" in data
                else None
            )
    else:
        gt_visual = None

    if gt_visual is None:
        gt_visual = load_ground_truth(
            sample,
            v10,
        )["semantic_visual"]

    if pred.shape != gt_visual.shape:
        raise ValueError(
            f"GT/prediction mismatch at sample {index}"
        )

    out = make_comparison(
        rgb,
        pred,
        gt_visual,
        index,
        sample["camera"],
    )

    if not cv2.imwrite(
        str(folder / "comparison.png"),
        out,
    ):
        raise OSError(
            f"Could not restore comparison for sample {index}"
        )

    return out


# ============================================================
# COMBINED MANIFEST
# ============================================================

def save_manifest(output_dir, selected, states):
    rows = []

    for index, sample in selected:
        if index not in states:
            continue

        folder = sample_folder(
            output_dir,
            index,
            sample["camera"],
        )

        meta = json.loads(
            (folder / "metadata.json").read_text(
                encoding="utf-8"
            )
        )

        m = meta.get("metrics", {})

        rows.append({
            "index": index,
            "camera": sample["camera"],
            "timestamp": sample["timestamp"],
            "image_path": str(sample["image_path"]),
            "gt_source": str(sample["npz_path"]),
            "output_folder": str(folder),
            "inference_seconds": meta.get(
                "inference_seconds", ""
            ),
            "occupancy_precision": m.get(
                "occupancy_precision", ""
            ),
            "occupancy_recall": m.get(
                "occupancy_recall", ""
            ),
            "occupancy_f1": m.get(
                "occupancy_f1", ""
            ),
            "semantic_miou_gt_foreground": m.get(
                "semantic_miou_gt_foreground", ""
            ),
            "gated_full_grid_miou": m.get(
                "gated_full_grid_miou", ""
            ),
            "gt_observed_background_voxels": m.get(
                "gt_observed_background_voxels", ""
            ),
            "pred_occupied_voxels": m.get(
                "pred_occupied_voxels", ""
            ),
            "gt_occupied_voxels": m.get(
                "gt_occupied_voxels", ""
            ),
            "status": states[index],
            "video_frame_number": "",
        })

    return rows


def write_manifest(output_dir, rows):
    path = output_dir / "frame_manifest.csv"

    if not rows:
        return

    tmp = path.with_name(
        path.name + ".partial"
    )

    try:
        with tmp.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0]),
            )

            writer.writeheader()
            writer.writerows(rows)

        os.replace(tmp, path)

    finally:
        tmp.unlink(missing_ok=True)


# ============================================================
# REBUILD COMPLETE VIDEO
# ============================================================

def build_video(
    output_dir,
    selected,
    rows,
    v10,
    fps,
    preview,
):
    """
    Combine ALL completed frames.

    Previously inferred frames are included.
    Newly inferred frames are included.

    Order follows find_samples() master indices.
    """

    by_index = {
        int(row["index"]): row
        for row in rows
    }

    final_path = (
        output_dir / "comparison_video.mp4"
    )

    tmp_path = (
        output_dir / "comparison_video.partial.mp4"
    )

    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        VIDEO_SIZE,
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Cannot create video {tmp_path}"
        )

    count = 0

    try:
        for index, sample in selected:
            if index not in by_index:
                continue

            folder = sample_folder(
                output_dir,
                index,
                sample["camera"],
            )

            frame = cv2.imread(
                str(folder / "comparison.png"),
                cv2.IMREAD_COLOR,
            )

            expected_shape = (
                PANEL_H,
                VIDEO_SIZE[0],
            )

            if (
                frame is None
                or frame.shape[:2] != expected_shape
            ):
                frame = regenerate_comparison(
                    folder,
                    sample,
                    index,
                    v10,
                )

            if frame.shape[:2] != expected_shape:
                raise RuntimeError(
                    f"Invalid comparison frame at "
                    f"index {index}: {frame.shape}"
                )

            writer.write(frame)

            by_index[index]["video_frame_number"] = count
            count += 1

            if count % 100 == 0:
                print(
                    f"Video: {count} frames",
                    flush=True,
                )

            if preview:
                cv2.imshow(
                    "V10.1 full-dataset replay",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    print(
                        "Preview stopped; "
                        "continuing video generation."
                    )
                    preview = False
                    cv2.destroyAllWindows()

    finally:
        writer.release()

        if preview:
            cv2.destroyAllWindows()

    os.replace(
        tmp_path,
        final_path,
    )

    print(
        f"Video: {final_path} ({count} frames)"
    )

    return count


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "V10.1 full-dataset inference "
            "with automatic resume"
        )
    )

    parser.add_argument(
        "--model_script",
        help="Path to original V10.1 training script",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=START_INDEX,
    )

    parser.add_argument(
        "--end",
        type=int,
        default=END_INDEX,
        help="Inclusive last index; default = entire dataset",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_PATH),
    )

    parser.add_argument(
        "--rfdetr_checkpoint",
        type=str,
        default=str(RFDETR_CHECKPOINT),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
    )

    parser.add_argument(
        "--occupancy_threshold",
        type=float,
        default=DEFAULT_OCCUPANCY_THRESHOLD,
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--no_amp",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinfer even completed predictions",
    )

    parser.add_argument(
        "--render_only",
        action="store_true",
        help="Rebuild manifest/video from saved outputs only",
    )

    parser.add_argument(
        "--skip_video",
        action="store_true",
        help="Save predictions and manifest without encoding video",
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display replay; do not use on headless GPU nodes",
    )

    parser.add_argument(
        "--manifest_every",
        type=int,
        default=25,
        help="Update manifest every N newly inferred frames",
    )

    args = parser.parse_args()

    if not args.model_script:
        parser.error(
            "--model_script is required"
        )

    if (
        args.fps <= 0
        or not 0 <= args.occupancy_threshold <= 1
        or args.start < 0
    ):
        parser.error(
            "Invalid fps, occupancy threshold or start"
        )

    if (
        args.end is not None
        and args.end < args.start
    ):
        parser.error(
            "--end must be >= --start"
        )

    if args.manifest_every < 1:
        parser.error(
            "--manifest_every must be >= 1"
        )

    if args.force and args.render_only:
        parser.error(
            "--force and --render_only cannot be combined"
        )

    args.checkpoint = str(
        Path(args.checkpoint).expanduser().resolve()
    )

    args.rfdetr_checkpoint = str(
        Path(args.rfdetr_checkpoint).expanduser().resolve()
    )

    output_dir = (
        Path(args.output).expanduser().resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load exact sample discovery from original training script.
    v10 = load_model_source(
        args.model_script
    )

    selected, total, last_index = get_selected(
        v10,
        args,
    )

    print("Output:", output_dir)

    if args.force:
        print("Force inference: ON")
    else:
        print("Skip existing: ON")

    # --------------------------------------------------------
    # Identify saved and missing predictions
    # --------------------------------------------------------

    states = {}
    pending = []

    for index, sample in selected:
        folder = sample_folder(
            output_dir,
            index,
            sample["camera"],
        )

        valid, reason = check_existing(
            folder,
            sample,
            index,
            v10,
            args,
        )

        if valid and not args.force:
            states[index] = "skipped_existing"

        elif args.render_only:
            print(
                f"[MISSING] {index} "
                f"{sample['camera']}: {reason}"
            )

        else:
            pending.append((
                index,
                sample,
                "forced" if args.force else reason,
            ))

    print("=" * 70)
    print(f"Already complete: {len(states)}")
    print(f"Need inference: {len(pending)}")
    print("=" * 70)

    if args.render_only and not states:
        raise RuntimeError(
            "No verified saved results to render"
        )

    # Check input files only for samples requiring inference.
    if not args.render_only:
        for index, sample, reason in pending:
            for key in ("image_path", "npz_path"):
                if not Path(sample[key]).is_file():
                    raise FileNotFoundError(
                        f"Index {index} missing {key}: "
                        f"{sample[key]}"
                    )

    completed_new = 0
    failed = None
    model_epoch = None
    started = time.perf_counter()

    # --------------------------------------------------------
    # Run only missing inference
    # --------------------------------------------------------

    try:
        if pending:
            (
                torch,
                model,
                detector,
                amp_enabled,
                model_epoch,
            ) = init_models(
                v10,
                args,
            )

            for n, (
                index,
                sample,
                reason,
            ) in enumerate(
                pending,
                start=1,
            ):
                print(
                    f"[{n}/{len(pending)}] "
                    f"INFER {index} "
                    f"{sample['camera']} "
                    f"t={sample['timestamp']} "
                    f"({reason})",
                    flush=True,
                )

                (
                    rgb,
                    gt,
                    result,
                    prior,
                    full_prior,
                    metrics,
                    elapsed,
                ) = infer_one(
                    sample,
                    index,
                    v10,
                    torch,
                    model,
                    detector,
                    amp_enabled,
                    args,
                )

                save_sample(
                    output_dir,
                    index,
                    sample,
                    rgb,
                    gt,
                    result,
                    prior,
                    full_prior,
                    metrics,
                    elapsed,
                    args,
                    v10,
                )

                states[index] = "inferred_new"
                completed_new += 1

                folder = sample_folder(
                    output_dir,
                    index,
                    sample["camera"],
                )

                print(
                    f"Saved: {folder} | "
                    f"{elapsed:.2f}s | "
                    f"occupancy F1="
                    f"{metrics['occupancy_f1']:.4f}",
                    flush=True,
                )

                # Periodic checkpoint of the manifest.
                if (
                    completed_new % args.manifest_every == 0
                ):
                    write_manifest(
                        output_dir,
                        save_manifest(
                            output_dir,
                            selected,
                            states,
                        ),
                    )

    except (Exception, KeyboardInterrupt) as exc:
        failed = exc

        print(
            f"Inference interrupted after "
            f"{completed_new} new frames: {exc}",
            flush=True,
        )

    finally:
        # Always preserve the manifest, even on interruption.
        rows = save_manifest(
            output_dir,
            selected,
            states,
        )

        write_manifest(
            output_dir,
            rows,
        )

        video_count = 0

        if (
            failed is None
            and rows
            and not args.skip_video
        ):
            print(
                "Building combined replay from "
                "saved + new comparisons...",
                flush=True,
            )

            video_count = build_video(
                output_dir,
                selected,
                rows,
                v10,
                args.fps,
                args.preview,
            )

            write_manifest(
                output_dir,
                rows,
            )

        summary = {
            "model": "V10.1",
            "dataset_samples": total,
            "start_index": args.start,
            "end_index": last_index,
            "expected_frames": len(selected),
            "processed_frames": len(rows),
            "skipped_existing": sum(
                s == "skipped_existing"
                for s in states.values()
            ),
            "newly_inferred": completed_new,
            "completed": (
                failed is None
                and len(rows) == len(selected)
            ),
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": model_epoch,
            "rfdetr_checkpoint": args.rfdetr_checkpoint,
            "voxel_shape": list(v10.VOXEL_SHAPE),
            "bounds": list(v10.BOUNDS),
            "voxel_size": float(v10.VOXEL_SIZE),
            "occupancy_threshold": args.occupancy_threshold,
            "sample_index_source": "V10.1 find_samples()",
            "lidar_used": False,
            "fps_playback": args.fps,
            "video": (
                str(output_dir / "comparison_video.mp4")
                if video_count else None
            ),
            "video_frames": video_count,
            "elapsed_wall_seconds": (
                time.perf_counter() - started
            ),
            "error": (
                str(failed)
                if failed is not None
                else None
            ),
        }

        atomic_json(
            output_dir / "sequence_metadata.json",
            summary,
        )

        print("=" * 70)
        print("FULL DATASET RUN FINISHED")
        print("=" * 70)
        print(f"Selected: {len(selected)}")
        print(f"Saved: {len(rows)}")
        print(
            f"Skipped: {summary['skipped_existing']}"
        )
        print(f"Newly inferred: {completed_new}")
        print(f"Video frames: {video_count}")
        print(
            "Manifest:",
            output_dir / "frame_manifest.csv",
        )
        print("=" * 70)

    if failed is not None:
        raise failed


if __name__ == "__main__":
    main()
