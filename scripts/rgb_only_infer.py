#!/usr/bin/env python3

"""
V10.1 RGB -> 3D voxel inference

Pipeline:

    RGB image
        |
        v
    V10.1 RGBToSemanticVoxelModelV10
        |
        +--> occupancy probability [320,30,195]
        |
        +--> semantic probabilities [4,320,30,195]
        |
        v
    predicted semantic voxel grid

V10.1 is RGB-only:
    - NO RF-DETR prior
    - NO LiDAR prior

Output NPZ contains:
    semantic
    occupancy
    semantic_confidence
    voxel_size
    origin
    bounds
    camera
    timestamp
    image_path
    class_names
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch


# ============================================================
# CONFIG
# ============================================================

NUM_CLASSES = 4

CLASS_NAMES = {
    0: "loose_trash",
    1: "mud_spill",
    2: "public_greenway",
    3: "sidewalk",
}

OCCUPANCY_THRESHOLD = 0.50


# ============================================================
# FIND V10.1 TRAINING SOURCE
# ============================================================

def find_model_source(repo_root: Path):
    """
    Find the actual V10.1 training source by looking for:

        class RGBToSemanticVoxelModelV10

    This deliberately does NOT assume a filename such as
    train_v10_1.py.
    """

    candidates = []

    for path in repo_root.rglob("*.py"):

        # Don't inspect this inference script.
        if path.resolve() == Path(__file__).resolve():
            continue

        # Ignore virtual environments / caches.
        parts = set(path.parts)

        if ".git" in parts:
            continue

        if "__pycache__" in parts:
            continue

        if "site-packages" in parts:
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        if "class RGBToSemanticVoxelModelV10" in text:
            candidates.append(path)

    if len(candidates) == 0:
        raise FileNotFoundError(
            "\nCould not automatically find the V10.1 training source.\n"
            "Pass it explicitly with:\n\n"
            "  --model-source /path/to/your/V10.1_training.py\n"
        )

    if len(candidates) > 1:

        print("\n[MODEL SOURCE] Multiple V10.1 sources found:")

        for i, path in enumerate(candidates):
            print(f"  [{i}] {path}")

        raise RuntimeError(
            "\nMultiple files contain RGBToSemanticVoxelModelV10.\n"
            "Use --model-source to select the correct V10.1 training file."
        )

    print(
        f"[MODEL SOURCE] Automatically found:\n"
        f"  {candidates[0]}"
    )

    return candidates[0]


def load_model_module(source_path: Path):

    source_path = source_path.resolve()

    print(
        f"[MODEL SOURCE] Loading:\n"
        f"  {source_path}"
    )

    spec = importlib.util.spec_from_file_location(
        "v10_1_training_module",
        source_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not create import specification for {source_path}"
        )

    module = importlib.util.module_from_spec(spec)

    # Make sure imports inside the training script can resolve.
    sys.path.insert(
        0,
        str(source_path.parent),
    )

    spec.loader.exec_module(module)

    if not hasattr(
        module,
        "RGBToSemanticVoxelModelV10",
    ):
        raise AttributeError(
            "Loaded source does not contain "
            "RGBToSemanticVoxelModelV10"
        )

    return module


# ============================================================
# CHECKPOINT DISCOVERY
# ============================================================

def find_checkpoint(repo_root: Path):

    candidates = []

    # Prefer hazard checkpoint.
    for pattern in (
        "**/best_hazard.pth",
        "**/best_full.pth",
    ):

        found = list(repo_root.glob(pattern))

        for path in found:
            if path not in candidates:
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "\nCould not automatically find a V10.1 checkpoint.\n"
            "Pass it explicitly with:\n\n"
            "  --checkpoint /path/to/best_hazard.pth\n"
        )

    # Prefer best_hazard.
    hazard = [
        p for p in candidates
        if p.name == "best_hazard.pth"
    ]

    if hazard:
        checkpoint = hazard[0]
    else:
        checkpoint = candidates[0]

    print(
        f"[CHECKPOINT] Automatically selected:\n"
        f"  {checkpoint}"
    )

    return checkpoint


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(model, checkpoint_path, device):

    print()
    print("=" * 80)
    print("LOADING CHECKPOINT")
    print("=" * 80)
    print(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"{checkpoint_path} does not contain "
            "'model_state_dict'"
        )

    state_dict = checkpoint["model_state_dict"]

    # Handle checkpoints that may have been saved from
    # DataParallel / DistributedDataParallel.
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    if missing:
        print("\n[WARNING] Missing checkpoint keys:")
        for key in missing[:30]:
            print(" ", key)

        if len(missing) > 30:
            print(
                f"  ... and {len(missing) - 30} more"
            )

    if unexpected:
        print("\n[WARNING] Unexpected checkpoint keys:")
        for key in unexpected[:30]:
            print(" ", key)

        if len(unexpected) > 30:
            print(
                f"  ... and {len(unexpected) - 30} more"
            )

    print(
        f"\n[CHECKPOINT] Epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    if "metrics" in checkpoint:
        metrics = checkpoint["metrics"]

        if isinstance(metrics, dict):
            print("[CHECKPOINT] Stored metrics:")

            for key in (
                "loss",
                "miou",
                "hazard_miou",
            ):
                if key in metrics:
                    print(
                        f"  {key}: {metrics[key]}"
                    )

    print(
        "\n[CHECKPOINT] Geometry stored in checkpoint:"
    )

    for key in (
        "voxel_shape",
        "voxel_size",
        "bounds",
        "source_voxel_shape",
        "source_bounds",
    ):
        if key in checkpoint:
            print(
                f"  {key}: {checkpoint[key]}"
            )

    print(
        "\n[CHECKPOINT] V10.1 configuration:"
    )

    for key in (
        "rgb_only_ablation",
        "rfdetr_prior_used",
        "lidar_prior_used",
        "foreground_only_semantic_task",
    ):
        if key in checkpoint:
            print(
                f"  {key}: {checkpoint[key]}"
            )

    return checkpoint


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image_path,
    image_size,
):
    """
    Exact V10.1 letterbox preprocessing.

    Returns:

        image_tensor
        scale
        pad_x
        pad_y
        orig_w
        orig_h
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    orig_w, orig_h = image.size

    scale = min(
        image_size / orig_w,
        image_size / orig_h,
    )

    new_w = max(
        1,
        int(round(orig_w * scale)),
    )

    new_h = max(
        1,
        int(round(orig_h * scale)),
    )

    image_resized = image.resize(
        (new_w, new_h),
        Image.Resampling.BILINEAR,
    )

    canvas = Image.new(
        "RGB",
        (image_size, image_size),
        (114, 114, 114),
    )

    pad_x = (
        image_size - new_w
    ) // 2

    pad_y = (
        image_size - new_h
    ) // 2

    canvas.paste(
        image_resized,
        (pad_x, pad_y),
    )

    arr = (
        np.asarray(canvas)
        .astype(np.float32)
        / 255.0
    )

    # ImageNet normalization used by V10.1.
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    arr = (arr - mean) / std

    arr = np.transpose(
        arr,
        (2, 0, 1),
    )

    tensor = torch.from_numpy(
        arr
    ).float()

    return (
        tensor,
        float(scale),
        float(pad_x),
        float(pad_y),
        float(orig_w),
        float(orig_h),
    )


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def run_inference(
    model,
    module,
    image_path,
    camera_name,
    device,
    occupancy_threshold,
):
    """
    Run the actual V10.1 RGB -> 3D voxel model.
    """

    if camera_name not in (
        "CAM2",
        "CAM6",
    ):
        raise ValueError(
            "camera must be CAM2 or CAM6"
        )

    camera_id_value = (
        0
        if camera_name == "CAM2"
        else 1
    )

    # V10.1 inference does NOT horizontally flip.
    flip_value = False

    (
        image,
        scale,
        pad_x,
        pad_y,
        orig_w,
        orig_h,
    ) = preprocess_image(
        image_path,
        module.IMAGE_SIZE,
    )

    image = (
        image
        .unsqueeze(0)
        .to(device)
    )

    camera_id = torch.tensor(
        [camera_id_value],
        dtype=torch.long,
        device=device,
    )

    flip = torch.tensor(
        [flip_value],
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

    print()
    print("=" * 80)
    print("3D VOXEL INFERENCE")
    print("=" * 80)

    print(f"Image:       {image_path}")
    print(f"Camera:      {camera_name}")
    print(f"Image input: {tuple(image.shape)}")
    print(f"Device:      {device}")

    # --------------------------------------------------------
    # ACTUAL V10.1 MODEL INFERENCE
    # --------------------------------------------------------

    if device.type == "cuda":

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            (
                occupancy_logits,
                semantic_logits,
                coarse_logits,
            ) = model(
                image,
                camera_id,
                flip,
                scale,
                pad_x,
                pad_y,
                orig_w,
                orig_h,
            )

    else:

        (
            occupancy_logits,
            semantic_logits,
            coarse_logits,
        ) = model(
            image,
            camera_id,
            flip,
            scale,
            pad_x,
            pad_y,
            orig_w,
            orig_h,
        )

    # --------------------------------------------------------
    # OCCUPANCY
    # --------------------------------------------------------

    occupancy_probability = (
        torch.sigmoid(
            occupancy_logits
        )
        .squeeze(0)
        .squeeze(0)
        .float()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # SEMANTIC
    # --------------------------------------------------------

    semantic_probability = (
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
        semantic_probability
        .argmax(axis=0)
        .astype(np.int64)
    )

    semantic_confidence = (
        semantic_probability
        .max(axis=0)
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # FINAL OCCUPANCY MASK
    # --------------------------------------------------------

    predicted_occupancy = (
        occupancy_probability
        >= occupancy_threshold
    )

    # V10.1 has NO semantic background class.
    #
    # Therefore:
    #
    #   occupied voxel -> 0..3 semantic class
    #   empty voxel    -> 255
    #
    semantic_prediction = (
        semantic_argmax
        .astype(np.uint8)
    )

    semantic_prediction[
        ~predicted_occupancy
    ] = 255

    # --------------------------------------------------------
    # PRINT STATS
    # --------------------------------------------------------

    total_voxels = (
        predicted_occupancy.size
    )

    occupied_voxels = int(
        predicted_occupancy.sum()
    )

    print()
    print("[OUTPUT]")
    print(
        "  Occupancy shape:",
        occupancy_probability.shape,
    )
    print(
        "  Semantic shape:",
        semantic_prediction.shape,
    )

    print(
        f"  Occupied voxels: "
        f"{occupied_voxels:,} / "
        f"{total_voxels:,}"
    )

    print(
        f"  Occupied fraction: "
        f"{occupied_voxels / total_voxels:.6f}"
    )

    print()
    print("[CLASS COUNTS]")

    for class_id, class_name in CLASS_NAMES.items():

        count = int(
            np.logical_and(
                predicted_occupancy,
                semantic_argmax == class_id,
            ).sum()
        )

        print(
            f"  {class_id}: "
            f"{class_name:20s} "
            f"{count:,}"
        )

    return {
        "occupancy_probability":
            occupancy_probability,

        "semantic_prediction":
            semantic_prediction,

        "semantic_confidence":
            semantic_confidence,

        "semantic_probability":
            semantic_probability,

        "predicted_occupancy":
            predicted_occupancy,
    }


# ============================================================
# SAVE NPZ
# ============================================================

def save_prediction(
    output_path,
    result,
    module,
    image_path,
    camera_name,
    timestamp,
):
    """
    Save the prediction using the same NPZ schema
    as the V10.1 training inference.
    """

    occupancy = result[
        "occupancy_probability"
    ]

    semantic = result[
        "semantic_prediction"
    ]

    semantic_confidence = result[
        "semantic_confidence"
    ]

    voxel_shape = tuple(
        module.VOXEL_SHAPE
    )

    bounds = np.asarray(
        module.BOUNDS,
        dtype=np.float32,
    )

    voxel_size = np.float32(
        module.VOXEL_SIZE
    )

    origin = np.array(
        [
            bounds[0],
            bounds[2],
            bounds[4],
        ],
        dtype=np.float32,
    )

    if tuple(occupancy.shape) != voxel_shape:

        raise RuntimeError(
            "\nMODEL OUTPUT GEOMETRY DOES NOT MATCH "
            "V10.1 TARGET GRID\n\n"
            f"Expected: {voxel_shape}\n"
            f"Got:      {occupancy.shape}\n"
        )

    np.savez_compressed(
        output_path,

        # Final semantic prediction:
        # 0..3 = class
        # 255   = predicted empty
        semantic=semantic,

        # Continuous occupancy probability.
        occupancy=occupancy.astype(
            np.float32
        ),

        # Maximum semantic probability.
        semantic_confidence=semantic_confidence,

        voxel_size=voxel_size,

        origin=origin,

        bounds=bounds,

        camera=np.array(
            camera_name
        ),

        timestamp=np.int64(
            timestamp
        ),

        image_path=np.array(
            str(image_path)
        ),

        class_names=np.array(
            [
                CLASS_NAMES[i]
                for i in range(NUM_CLASSES)
            ]
        ),
    )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)
    print(output_path)

    print()
    print("NPZ keys:")

    with np.load(
        output_path,
        allow_pickle=True,
    ) as data:

        for key in data.files:
            print(
                f"  {key:25s} "
                f"{data[key].shape if hasattr(data[key], 'shape') else ''}"
            )


# ============================================================
# OPTIONAL GT COMPARISON
# ============================================================

def print_gt_info(gt_path):

    if gt_path is None:
        return

    gt_path = Path(gt_path)

    if not gt_path.exists():
        raise FileNotFoundError(
            f"GT NPZ does not exist: {gt_path}"
        )

    with np.load(
        gt_path,
        allow_pickle=True,
    ) as gt:

        print()
        print("=" * 80)
        print("GROUND TRUTH")
        print("=" * 80)

        for key in gt.files:

            value = gt[key]

            if hasattr(value, "shape"):
                print(
                    f"  {key:25s} "
                    f"shape={value.shape}"
                )
            else:
                print(
                    f"  {key}: {value}"
                )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "V10.1 RGB -> 3D semantic voxel inference"
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Input rectified RGB image",
    )

    parser.add_argument(
        "--camera",
        required=True,
        choices=["CAM2", "CAM6"],
        help="Camera corresponding to the input image",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "V10.1 best_hazard.pth or best_full.pth. "
            "If omitted, the project is searched automatically."
        ),
    )

    parser.add_argument(
        "--model-source",
        type=Path,
        default=None,
        help=(
            "Actual V10.1 training .py containing "
            "RGBToSemanticVoxelModelV10. "
            "If omitted, it is searched automatically."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output prediction NPZ",
    )

    parser.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="Optional GT NPZ to inspect",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Occupancy threshold (default: 0.50)",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    repo_root = Path(
        __file__
    ).resolve().parents[1]

    if not args.image.exists():
        raise FileNotFoundError(
            f"Input image not found:\n{args.image}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if args.cpu:
        device = torch.device("cpu")

    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print(
        f"[DEVICE] {device}"
    )

    if device.type == "cuda":
        print(
            f"[GPU] "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Model source
    # --------------------------------------------------------

    if args.model_source is not None:

        model_source = (
            args.model_source.resolve()
        )

    else:

        model_source = find_model_source(
            repo_root
        )

    module = load_model_module(
        model_source
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The model constructor normally has:
    #
    #     USE_PRETRAINED = True
    #
    # We already have the trained weights in the checkpoint,
    # so don't make torchvision try to download ImageNet
    # weights again.
    # --------------------------------------------------------

    if hasattr(
        module,
        "USE_PRETRAINED",
    ):
        module.USE_PRETRAINED = False

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    if args.checkpoint is not None:

        checkpoint_path = (
            args.checkpoint.resolve()
        )

    else:

        checkpoint_path = find_checkpoint(
            repo_root
        )

    # --------------------------------------------------------
    # Build exact V10.1 model
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("BUILDING V10.1 MODEL")
    print("=" * 80)

    model = (
        module.RGBToSemanticVoxelModelV10()
        .to(device)
    )

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    checkpoint = load_checkpoint(
        model,
        checkpoint_path,
        device,
    )

    model.eval()

    # --------------------------------------------------------
    # Verify model geometry
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("V10.1 MODEL GEOMETRY")
    print("=" * 80)

    print(
        "Source grid:",
        module.SOURCE_VOXEL_SHAPE,
    )

    print(
        "Source bounds:",
        module.SOURCE_BOUNDS,
    )

    print(
        "Target grid:",
        module.VOXEL_SHAPE,
    )

    print(
        "Target bounds:",
        module.BOUNDS,
    )

    print(
        "Voxel size:",
        module.VOXEL_SIZE,
    )

    print(
        "Latent shape:",
        module.LATENT_SHAPE,
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    result = run_inference(
        model=model,
        module=module,
        image_path=args.image,
        camera_name=args.camera,
        device=device,
        occupancy_threshold=args.threshold,
    )

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    if args.output is not None:

        output_path = (
            args.output.resolve()
        )

    else:

        output_path = (
            args.image.parent
            / f"{args.image.stem}_v10_1_voxels.npz"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    try:
        timestamp = int(
            args.image.stem
        )
    except ValueError:
        timestamp = -1

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_prediction(
        output_path=output_path,
        result=result,
        module=module,
        image_path=args.image,
        camera_name=args.camera,
        timestamp=timestamp,
    )

    # --------------------------------------------------------
    # Optional GT
    # --------------------------------------------------------

    print_gt_info(
        args.gt
    )

    print()
    print("=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)

    print(
        f"Prediction:\n  {output_path}"
    )


if __name__ == "__main__":
    main()