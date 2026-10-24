#!/usr/bin/env python3

import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# ============================================================
# IMPORT YOUR V10.1 MODEL
# ============================================================

from lidar_ablation import (
    RGBPriorToSemanticVoxelModelV10,
    letterbox_image,
    IMAGE_SIZE,
    VOXEL_SHAPE,
    BOUNDS,
    VOXEL_SIZE,
    CLASS_NAMES,
    USE_AMP,
)

# ============================================================
# RF-DETR
# ============================================================

from rfdetr import RFDETRSegSmall


# ============================================================
# CONFIG
# ============================================================

OCCUPANCY_THRESHOLD = 0.50
RFDETR_THRESHOLD = 0.20
PRIOR_SIZE = 192

NUM_CLASSES = 4

CLASS_NAMES = {
    0: "loose_trash",
    1: "mud_spill",
    2: "public_greenway",
    3: "sidewalk",
}


# ============================================================
# RF-DETR
# ============================================================

def load_rfdetr(checkpoint):
    print("=" * 80)
    print("LOADING RF-DETR")
    print("=" * 80)

    model = RFDETRSegSmall(
        pretrain_weights=str(checkpoint),
        trust_checkpoint=True,
    )

    print(f"RF-DETR checkpoint: {checkpoint}")
    print()

    return model


def generate_rfdetr_prior(
    rfdetr,
    image_path,
):
    """
    Generate the four RF-DETR foreground probability channels.

    Output:
        [4, 192, 192]

    Channels:
        0 = loose_trash
        1 = mud_spill
        2 = public_greenway
        3 = sidewalk
    """

    print("=" * 80)
    print("GENERATING RF-DETR PRIOR")
    print("=" * 80)

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image:\n{image_path}"
        )

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    height, width = image_rgb.shape[:2]

    print(
        f"Image: {width} x {height}"
    )

    # --------------------------------------------------------
    # RF-DETR inference
    # --------------------------------------------------------

    detections = rfdetr.predict(
        image_rgb,
        threshold=RFDETR_THRESHOLD,
    )

    # --------------------------------------------------------
    # Empty prior
    #
    # [class, H, W]
    # --------------------------------------------------------

    prior = np.zeros(
        (
            NUM_CLASSES,
            height,
            width,
        ),
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # RF-DETR result handling
    # --------------------------------------------------------

    if detections is None:
        print(
            "[RF-DETR] No detections."
        )

        return torch.from_numpy(
            prior
        ).unsqueeze(0)

    class_ids = (
        np.asarray(
            detections.class_id
        )
        if hasattr(
            detections,
            "class_id",
        )
        else np.array([])
    )

    confidences = (
        np.asarray(
            detections.confidence
        )
        if hasattr(
            detections,
            "confidence",
        )
        else np.ones(
            len(class_ids),
            dtype=np.float32,
        )
    )

    masks = (
        np.asarray(
            detections.mask
        )
        if hasattr(
            detections,
            "mask",
        )
        and detections.mask is not None
        else None
    )

    print(
        f"[RF-DETR] detections: "
        f"{len(class_ids)}"
    )

    if masks is None:
        raise RuntimeError(
            "RF-DETR did not return segmentation masks."
        )

    # --------------------------------------------------------
    # Convert each segmentation mask into a semantic prior.
    #
    # Multiple instances of the same class are combined using
    # maximum confidence.
    # --------------------------------------------------------

    for i in range(len(class_ids)):

        class_id = int(
            class_ids[i]
        )

        confidence = float(
            confidences[i]
        )

        # Ignore classes outside the four V10 classes.
        if class_id < 0 or class_id >= NUM_CLASSES:
            continue

        mask = masks[i]

        mask = mask.astype(
            np.float32
        )

        if mask.shape != (
            height,
            width,
        ):
            mask = cv2.resize(
                mask,
                (
                    width,
                    height,
                ),
                interpolation=cv2.INTER_LINEAR,
            )

        # RF-DETR confidence × mask probability.
        score_map = (
            mask * confidence
        )

        prior[class_id] = np.maximum(
            prior[class_id],
            score_map,
        )

        print(
            f"  detection {i:03d}: "
            f"{CLASS_NAMES[class_id]:20s} "
            f"confidence={confidence:.3f}"
        )

    # --------------------------------------------------------
    # Resize to exactly the same resolution used during
    # V10.1 training.
    #
    # Training code:
    #
    # prior -> 192 x 192
    # --------------------------------------------------------

    prior_tensor = torch.from_numpy(
        prior
    ).unsqueeze(0)

    prior_tensor = F.interpolate(
        prior_tensor,
        size=(
            PRIOR_SIZE,
            PRIOR_SIZE,
        ),
        mode="bilinear",
        align_corners=False,
    )

    prior_tensor = prior_tensor.clamp(
        0.0,
        1.0,
    )

    print(
        "RF-DETR prior:",
        tuple(prior_tensor.shape),
    )

    for i in range(NUM_CLASSES):
        print(
            f"  {i}: "
            f"{CLASS_NAMES[i]:20s} "
            f"max={prior_tensor[0, i].max().item():.4f} "
            f"mean={prior_tensor[0, i].mean().item():.4f}"
        )

    print()

    return prior_tensor


# ============================================================
# LOAD V10.1
# ============================================================

def load_v10(
    checkpoint,
    device,
):
    print("=" * 80)
    print("LOADING V10.1")
    print("=" * 80)

    checkpoint_data = torch.load(
        checkpoint,
        map_location=device,
    )

    model = (
        RGBPriorToSemanticVoxelModelV10()
        .to(device)
    )

    model.load_state_dict(
        checkpoint_data[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()

    print(
        f"Checkpoint: {checkpoint}"
    )

    print(
        "Epoch:",
        checkpoint_data.get(
            "epoch",
            "unknown",
        ),
    )

    if "metrics" in checkpoint_data:

        try:
            val = checkpoint_data[
                "metrics"
            ]["val"]

            print(
                f"Val mIoU: "
                f"{val['miou']:.4f}"
            )

            print(
                f"Val hazard mIoU: "
                f"{val['hazard_miou']:.4f}"
            )

        except Exception:
            pass

    print()

    return model


# ============================================================
# RGB PREPROCESSING
# ============================================================

def prepare_image(
    image_path,
):

    image = Image.open(
        image_path
    ).convert("RGB")

    (
        tensor,
        scale,
        pad_x,
        pad_y,
        orig_w,
        orig_h,
    ) = letterbox_image(
        image,
        IMAGE_SIZE,
    )

    tensor = tensor.unsqueeze(0)

    return (
        tensor,
        scale,
        pad_x,
        pad_y,
        orig_w,
        orig_h,
    )


# ============================================================
# V10.1 INFERENCE
# ============================================================

@torch.no_grad()
def run_v10_inference(
    model,
    image,
    rf_prior,
    camera,
    scale,
    pad_x,
    pad_y,
    orig_w,
    orig_h,
    device,
):

    camera_id = (
        0
        if camera == "CAM2"
        else 1
    )

    image = image.to(
        device,
        non_blocking=True,
    )

    rf_prior = rf_prior.to(
        device,
        non_blocking=True,
    )

    camera_id = torch.tensor(
        [camera_id],
        dtype=torch.long,
        device=device,
    )

    # No augmentation during inference.
    flip = torch.tensor(
        [False],
        dtype=torch.bool,
        device=device,
    )

    scale = torch.tensor(
        [scale],
        dtype=torch.float32,
        device=device,
    )

    pad_x = torch.tensor(
        [pad_x],
        dtype=torch.float32,
        device=device,
    )

    pad_y = torch.tensor(
        [pad_y],
        dtype=torch.float32,
        device=device,
    )

    orig_w = torch.tensor(
        [orig_w],
        dtype=torch.float32,
        device=device,
    )

    orig_h = torch.tensor(
        [orig_h],
        dtype=torch.float32,
        device=device,
    )

    print("=" * 80)
    print("RUNNING V10.1")
    print("=" * 80)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=(
            USE_AMP
            and device.type == "cuda"
        ),
    ):

        (
            occupancy_logits,
            semantic_logits,
            coarse_logits,
        ) = model(
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

    # ========================================================
    # OCCUPANCY
    # ========================================================

    occupancy_prob = (
        torch.sigmoid(
            occupancy_logits
        )
        .squeeze(0)
        .squeeze(0)
        .float()
        .cpu()
        .numpy()
    )

    # ========================================================
    # SEMANTIC
    # ========================================================

    semantic_prob = (
        torch.softmax(
            semantic_logits,
            dim=1,
        )
        .squeeze(0)
        .float()
        .cpu()
        .numpy()
    )

    semantic_argmax = (
        semantic_prob.argmax(
            axis=0
        )
        .astype(np.uint8)
    )

    semantic_confidence = (
        semantic_prob.max(
            axis=0
        )
        .astype(np.float32)
    )

    # ========================================================
    # OCCUPANCY GATING
    # ========================================================

    pred_occupancy = (
        occupancy_prob
        >= OCCUPANCY_THRESHOLD
    )

    semantic = (
        semantic_argmax.copy()
    )

    # 255 = empty / not occupied
    semantic[
        ~pred_occupancy
    ] = 255

    return (
        semantic,
        occupancy_prob,
        semantic_confidence,
    )


# ============================================================
# SAVE
# ============================================================

def save_prediction(
    output_path,
    semantic,
    occupancy,
    confidence,
    camera,
    image_path,
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,

        semantic=semantic.astype(
            np.uint8
        ),

        occupancy=occupancy.astype(
            np.float32
        ),

        semantic_confidence=confidence.astype(
            np.float32
        ),

        voxel_size=np.float32(
            VOXEL_SIZE
        ),

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
            camera
        ),

        image_path=np.array(
            str(image_path)
        ),

        class_names=np.array(
            [
                CLASS_NAMES[0],
                CLASS_NAMES[1],
                CLASS_NAMES[2],
                CLASS_NAMES[3],
            ],
            dtype="<U32",
        ),
    )

    print(
        f"[SAVED] {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
    )

    parser.add_argument(
        "--camera",
        required=True,
        choices=[
            "CAM2",
            "CAM6",
        ],
    )

    parser.add_argument(
        "--v10_checkpoint",
        default = "/home/lcha0115/bt60_scratch/lcha_data/map-anything/personal_data/hazard_test_101/rgb_voxel_training_v10_1_5cm_xm8_8_y0_15_z025_10_dense_latent_hazard_empty_no_lidar_ablation/checkpoints/best_full.pth",
    )

    parser.add_argument(
        "--rfdetr_checkpoint",
        default = "personal_data/2D-Seg-Road-Nissan.v13i.coco-segmentation/outputs/checkpoint_best_total.pth",
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable."
        )

    device = torch.device(
        "cuda"
    )

    print("=" * 80)
    print("V10.1 RGB-ONLY DEPLOYMENT")
    print("=" * 80)

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print()

    # --------------------------------------------------------
    # RF-DETR
    # --------------------------------------------------------

    rfdetr = load_rfdetr(
        args.rfdetr_checkpoint
    )

    # --------------------------------------------------------
    # V10.1
    # --------------------------------------------------------

    model = load_v10(
        args.v10_checkpoint,
        device,
    )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    (
        image,
        scale,
        pad_x,
        pad_y,
        orig_w,
        orig_h,
    ) = prepare_image(
        args.image
    )

    # --------------------------------------------------------
    # RGB -> RF-DETR -> 4-channel prior
    # --------------------------------------------------------

    rf_prior = generate_rfdetr_prior(
        rfdetr,
        args.image,
    )

    # --------------------------------------------------------
    # V10.1
    # --------------------------------------------------------

    (
        semantic,
        occupancy,
        confidence,
    ) = run_v10_inference(
        model=model,
        image=image,
        rf_prior=rf_prior,
        camera=args.camera,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        orig_w=orig_w,
        orig_h=orig_h,
        device=device,
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    if semantic.shape != VOXEL_SHAPE:
        raise RuntimeError(
            f"Semantic shape {semantic.shape} "
            f"!= expected {VOXEL_SHAPE}"
        )

    if occupancy.shape != VOXEL_SHAPE:
        raise RuntimeError(
            f"Occupancy shape {occupancy.shape} "
            f"!= expected {VOXEL_SHAPE}"
        )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    print(
        "Output shape:",
        semantic.shape,
    )

    print(
        "Voxel size:",
        VOXEL_SIZE,
        "m",
    )

    print(
        "Bounds:",
        BOUNDS,
    )

    print(
        "Predicted occupied fraction:",
        f"{(semantic != 255).mean():.6f}",
    )

    for class_id in range(
        NUM_CLASSES
    ):

        count = int(
            (
                semantic
                == class_id
            ).sum()
        )

        print(
            f"{CLASS_NAMES[class_id]:20s}: "
            f"{count:,}"
        )

    print(
        f"{'empty':20s}: "
        f"{(semantic == 255).sum():,}"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_prediction(
        output_path=args.output,
        semantic=semantic,
        occupancy=occupancy,
        confidence=confidence,
        camera=args.camera,
        image_path=args.image,
    )

    # Copy RGB image alongside result
    output_path = Path(
        args.output
    )

    image_copy = (
        output_path.parent
        / (
            output_path.stem
            + "_image"
            + Path(
                args.image
            ).suffix
        )
    )

    shutil.copy2(
        args.image,
        image_copy,
    )

    print(
        f"[SAVED] {image_copy}"
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()