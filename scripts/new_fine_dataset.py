
#!/usr/bin/env python3
"""
Generate independent semantic voxel NPZ files for CAM2 and CAM6.

Pipeline:
    RGB + LiDAR
       |-- RF-DETR -> semantic segmentation
       |-- MapAnything -> camera-relative 3D points
       |-- Voxelisation -> semantic + occupancy
       |-- Compressed NPZ

Timing:
    - Individual time for every generated NPZ
    - Image loading
    - LiDAR loading
    - RF-DETR inference + segmentation
    - MapAnything inference + preprocessing
    - Voxelisation
    - NPZ compression / saving
    - Total end-to-end NPZ generation time
    - CSV log and overall statistics

Run:
    python -u generate_semantic_voxels.py
    python -u generate_semantic_voxels.py --limit 10
    python -u generate_semantic_voxels.py --camera CAM2
    python -u generate_semantic_voxels.py --overwrite --limit 5
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from rfdetr import RFDETRSegSmall
from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs


# ============================================================
# CONFIG
# ============================================================

BAG_ROOT = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/hazard_test_101"
)

EXTRACTED = BAG_ROOT / "extracted"
OUTPUT_DIR = BAG_ROOT / "semantic_voxels_new_roi_5cm"

CAMERAS = ["CAM2", "CAM6"]

SYNC_TOLERANCE_NS = 50_000_000

LIDAR_MAX_RANGE = 50.0
LIDAR_MIN_CAMERA_DEPTH = 0.05
BORDER_MARGIN_PIXELS = 2

VOXEL_SIZE = 0.05

X_MIN, X_MAX = -8.0, 8.0
Y_MIN, Y_MAX = 0.0, 2.0
Z_MIN, Z_MAX = 0.25, 10.0

EXPECTED_GRID_SHAPE = (320, 40, 195)

EXPECTED_GRID_BOUNDS = (
    -8.0, 8.0,
    0.0, 2.0,
    0.25, 10.0,
)

RFDETR_MODEL_PATH = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/"
    "2D-Seg-Road-Nissan.v13i.coco-segmentation/"
    "outputs/checkpoint_best_total.pth"
)

RFDETR_CONFIDENCE_THRESHOLD = 0.30

CLASS_NAMES = {
    0: "fallen_fence",
    1: "loose_trash",
    2: "mud_spill",
    3: "public_greenway",
    4: "sidewalk",
}

OUTPUT_CLASS_IDS = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
}

OUTPUT_CLASS_NAMES = np.array(
    [
        "background",
        "fallen_fence",
        "loose_trash",
        "mud_spill",
        "public_greenway",
        "sidewalk",
    ],
    dtype="<U32",
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# EXTRINSICS
# ============================================================

LIDAR_TO_CAMERA = {
    "CAM2": np.array([
        [0.64901045, 0.76012775, -0.03148390, -0.18778682],
        [0.09876220, -0.12521404, -0.98720184, -0.85765461],
        [-0.75434174, 0.63759489, -0.15633711, -0.93159365],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64),

    "CAM6": np.array([
        [-0.67094235, -0.74109244, -0.02486673, 0.26797529],
        [-0.11488898, 0.13702719, -0.98388214, -0.69426488],
        [0.73255504, -0.65727128, -0.17708070, -0.91874840],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64),
}


# ============================================================
# TIMING HELPERS
# ============================================================

def sync_gpu():
    """Synchronise GPU before reading elapsed time."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


class Timer:
    """Accurate elapsed wall-clock timer, including GPU work."""

    def __enter__(self):
        sync_gpu()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sync_gpu()
        self.elapsed = time.perf_counter() - self.start


TIMING_FIELDS = [
    "camera",
    "timestamp",
    "npz_path",
    "image_loading_s",
    "lidar_loading_s",
    "rfdetr_s",
    "mapanything_s",
    "voxelisation_s",
    "npz_saving_s",
    "other_s",
    "total_s",
    "occupied_voxels",
    "hazard_voxels",
]


def print_timing(timings, running_average, processed):
    print("\n  GENERATION TIMING")
    print("  " + "-" * 47)

    labels = [
        ("Image loading", "image_loading_s"),
        ("LiDAR loading", "lidar_loading_s"),
        ("RF-DETR", "rfdetr_s"),
        ("MapAnything", "mapanything_s"),
        ("Voxelisation", "voxelisation_s"),
        ("NPZ saving", "npz_saving_s"),
        ("Other overhead", "other_s"),
    ]

    for label, key in labels:
        print(f"  {label:<22} {timings[key]:>10.3f} s")

    print("  " + "-" * 47)
    print(f"  {'TOTAL NPZ TIME':<22} {timings['total_s']:>10.3f} s")
    print(f"  {'Running average':<22} {running_average:>10.3f} s/NPZ")
    print(f"  {'Completed':<22} {processed:>10,} NPZ")


def print_summary(records, wall_time, skipped, failed):
    print("\n" + "=" * 80)
    print("GENERATION SUMMARY")
    print("=" * 80)

    print(f"Generated: {len(records):,}")
    print(f"Skipped:   {skipped:,}")
    print(f"Failed:    {failed:,}")

    if not records:
        print("No new NPZ files were generated.")
        return

    totals = np.array(
        [r["total_s"] for r in records],
        dtype=np.float64,
    )

    print("\nPER-NPZ GENERATION TIME")
    print(f"  Average:  {np.mean(totals):.3f} s")
    print(f"  Median:   {np.median(totals):.3f} s")
    print(f"  Minimum:  {np.min(totals):.3f} s")
    print(f"  Maximum:  {np.max(totals):.3f} s")
    print(f"  Std dev:  {np.std(totals):.3f} s")
    print(f"  P95:      {np.percentile(totals, 95):.3f} s")

    print("\nAVERAGE TIME BREAKDOWN")

    keys = [
        ("Image loading", "image_loading_s"),
        ("LiDAR loading", "lidar_loading_s"),
        ("RF-DETR", "rfdetr_s"),
        ("MapAnything", "mapanything_s"),
        ("Voxelisation", "voxelisation_s"),
        ("NPZ saving", "npz_saving_s"),
        ("Other", "other_s"),
    ]

    for label, key in keys:
        avg = np.mean([r[key] for r in records])
        percentage = 100.0 * avg / np.mean(totals)
        print(f"  {label:<20} {avg:8.3f} s  ({percentage:5.1f}%)")

    print("\nBY CAMERA")

    for camera in CAMERAS:
        values = [
            r["total_s"]
            for r in records
            if r["camera"] == camera
        ]

        if values:
            print(
                f"  {camera}: {len(values):,} files, "
                f"mean={np.mean(values):.3f} s, "
                f"median={np.median(values):.3f} s"
            )

    processing_time = float(np.sum(totals))

    print("\nTHROUGHPUT")
    print(f"  Generation time: {processing_time:.2f} s")
    print(f"  Full run time:    {wall_time:.2f} s")
    print(f"  Generation rate:  {len(records)/processing_time:.3f} NPZ/s")
    print(f"  Including setup:  {len(records)/wall_time:.3f} NPZ/s")


# ============================================================
# CALIBRATION / FILES
# ============================================================

def parse_array(value):
    value = str(value).strip()

    for wrapper in [
        "np.float64(",
        "np.float32(",
        "np.int64(",
        "np.int32(",
    ]:
        value = value.replace(wrapper, "")

    value = (
        value.replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace(",", " ")
    )

    values = np.fromstring(
        value,
        sep=" ",
        dtype=np.float64,
    )

    if values.size == 0:
        raise RuntimeError(f"Could not parse array: {value}")

    return values


def load_intrinsics(camera):
    path = EXTRACTED / camera / "camera_info.csv"

    with open(path, "r") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"No camera info found: {path}")

    row = rows[0]
    P = parse_array(row["P"])

    if P.size != 12:
        raise RuntimeError(f"{camera}: invalid P size {P.size}")

    K = P.reshape(3, 4)[:, :3]

    return K, int(row["width"]), int(row["height"])


def load_images(camera):
    directory = EXTRACTED / camera / "images_rect"
    records = []

    for path in directory.glob("*.png"):
        try:
            timestamp = int(path.stem)
        except ValueError:
            continue

        records.append((timestamp, path))

    records.sort(key=lambda x: x[0])

    if not records:
        raise RuntimeError(f"No images found: {directory}")

    return records


def load_lidar_files():
    directory = EXTRACTED / "lidar"
    records = []

    for path in directory.glob("*.ply"):
        try:
            timestamp = int(path.stem)
        except ValueError:
            continue

        records.append((timestamp, path))

    records.sort(key=lambda x: x[0])

    if not records:
        raise RuntimeError(f"No LiDAR files found: {directory}")

    return records


def nearest_timestamp(target, records):
    timestamps = np.asarray(
        [x[0] for x in records],
        dtype=np.int64,
    )

    index = int(np.searchsorted(timestamps, target))
    candidates = []

    if index > 0:
        candidates.append(index - 1)

    if index < len(records):
        candidates.append(index)

    if not candidates:
        return None, None, None

    best = min(
        candidates,
        key=lambda i: abs(int(records[i][0]) - int(target)),
    )

    timestamp = int(records[best][0])

    return (
        timestamp,
        records[best][1],
        abs(timestamp - int(target)),
    )


# ============================================================
# ASCII PLY
# ============================================================

def load_ascii_ply(path):
    header = []

    with open(path, "r") as f:
        while True:
            line = f.readline()

            if not line:
                raise RuntimeError(f"Unexpected EOF: {path}")

            line = line.strip()
            header.append(line)

            if line == "end_header":
                break

    if not header or header[0] != "ply":
        raise RuntimeError(f"Invalid PLY header: {path}")

    if "format ascii 1.0" not in header:
        raise RuntimeError(
            f"Only ASCII PLY is supported: {path}"
        )

    vertex_count = None
    properties = []
    in_vertex = False

    for line in header:
        parts = line.split()

        if (
            len(parts) == 3
            and parts[0] == "element"
            and parts[1] == "vertex"
        ):
            vertex_count = int(parts[2])
            in_vertex = True
            continue

        if (
            in_vertex
            and len(parts) >= 3
            and parts[0] == "property"
        ):
            properties.append(parts[-1])

        elif (
            in_vertex
            and len(parts) >= 2
            and parts[0] == "element"
        ):
            break

    if vertex_count is None:
        raise RuntimeError(f"No vertex count: {path}")

    if vertex_count == 0:
        return np.empty((0, len(properties))), properties

    data = np.loadtxt(
        path,
        skiprows=len(header),
        max_rows=vertex_count,
        ndmin=2,
    )

    return data, properties


def load_lidar_ply(path):
    data, properties = load_ascii_ply(path)

    xyz = data[
        :,
        [
            properties.index("x"),
            properties.index("y"),
            properties.index("z"),
        ],
    ].astype(np.float64)

    valid = np.isfinite(xyz).all(axis=1)
    ranges = np.linalg.norm(xyz, axis=1)

    return xyz[valid & (ranges <= LIDAR_MAX_RANGE)]


# ============================================================
# LIDAR DEPTH
# ============================================================

def transform_points(points, T):
    homogeneous = np.concatenate(
        [
            points,
            np.ones((len(points), 1)),
        ],
        axis=1,
    )

    return (homogeneous @ T.T)[:, :3]


def project_lidar(points, camera, K, width, height):
    camera_points = transform_points(
        points,
        LIDAR_TO_CAMERA[camera],
    )

    X = camera_points[:, 0]
    Y = camera_points[:, 1]
    Z = camera_points[:, 2]

    valid = (
        np.isfinite(camera_points).all(axis=1)
        & (Z > LIDAR_MIN_CAMERA_DEPTH)
    )

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    if len(Z) == 0:
        return (
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.float64),
        )

    u = np.rint(
        K[0, 0] * X / Z + K[0, 2]
    ).astype(np.intp)

    v = np.rint(
        K[1, 1] * Y / Z + K[1, 2]
    ).astype(np.intp)

    inside = (
        (u >= BORDER_MARGIN_PIXELS)
        & (u < width - BORDER_MARGIN_PIXELS)
        & (v >= BORDER_MARGIN_PIXELS)
        & (v < height - BORDER_MARGIN_PIXELS)
    )

    return u[inside], v[inside], Z[inside]


def lidar_depth_image(
    points,
    camera,
    K,
    width,
    height,
):
    u, v, depths = project_lidar(
        points, camera, K, width, height
    )

    depth = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    if len(u) == 0:
        return depth

    pixel_ids = v.astype(np.int64) * width + u.astype(np.int64)

    order = np.lexsort((depths, pixel_ids))
    sorted_pixels = pixel_ids[order]

    keep = np.ones(len(order), dtype=bool)

    if len(order) > 1:
        keep[1:] = sorted_pixels[1:] != sorted_pixels[:-1]

    selected = order[keep]

    depth[v[selected], u[selected]] = depths[selected].astype(
        np.float32
    )

    return depth


# ============================================================
# RF-DETR
# ============================================================

def normalise_rfdetr_mask(mask, image_height, image_width):
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    mask = np.squeeze(np.asarray(mask))

    if mask.dtype != np.bool_:
        mask = mask > 0.5

    if mask.ndim != 2:
        raise ValueError(
            f"Unexpected RF-DETR mask shape: {mask.shape}"
        )

    if mask.shape != (image_height, image_width):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    return mask


def run_rfdetr_segmentation(model, image):
    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    with torch.inference_mode():
        detections = model.predict([image_bgr])[0]

    image_height, image_width = image.shape[:2]

    if not hasattr(detections, "mask"):
        raise RuntimeError(
            "RF-DETR returned no segmentation masks."
        )

    masks = detections.mask
    class_ids = detections.class_id
    confidences = detections.confidence

    point_class = np.full(
        (image_height, image_width),
        -1,
        dtype=np.int16,
    )

    if masks is None:
        if len(class_ids) > 0:
            raise RuntimeError(
                "RF-DETR detections exist but masks are None."
            )
        return point_class

    ordered = sorted(
        range(len(class_ids)),
        key=lambda i: float(confidences[i]),
        reverse=True,
    )

    for detection_index in ordered:
        confidence = float(confidences[detection_index])
        class_id = int(class_ids[detection_index])

        if confidence < RFDETR_CONFIDENCE_THRESHOLD:
            continue

        if class_id not in CLASS_NAMES:
            continue

        mask = normalise_rfdetr_mask(
            masks[detection_index],
            image_height,
            image_width,
        )

        assign = mask & (point_class == -1)
        point_class[assign] = class_id

    return point_class


# ============================================================
# MAPANYTHING
# ============================================================

def run_mapanything_single(
    model,
    image,
    lidar,
    camera,
    K,
    width,
    height,
):
    depth = lidar_depth_image(
        lidar,
        camera,
        K,
        width,
        height,
    )

    view = {
        "img": image,
        "intrinsics": K.astype(np.float32),
        "depth_z": torch.from_numpy(depth).float(),
        "is_metric_scale": torch.tensor(
            [True],
            device=DEVICE,
        ),
    }

    processed = preprocess_inputs([view])

    with torch.no_grad():
        predictions = model.infer(
            processed,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
            confidence_percentile=10,
            use_multiview_confidence=False,
            ignore_calibration_inputs=False,
            ignore_depth_inputs=False,
            ignore_pose_inputs=True,
            ignore_depth_scale_inputs=False,
            ignore_pose_scale_inputs=True,
        )

    if len(predictions) != 1:
        raise RuntimeError(
            f"Expected 1 MapAnything prediction, "
            f"got {len(predictions)}"
        )

    prediction = predictions[0]

    points = (
        prediction["pts3d_cam"]
        .detach()
        .cpu()
        .numpy()
    )

    while points.ndim > 3 and points.shape[0] == 1:
        points = points[0]

    if points.ndim != 3 or points.shape[-1] != 3:
        raise RuntimeError(
            f"Unexpected pts3d_cam shape: {points.shape}"
        )

    h, w = points.shape[:2]
    mask = prediction.get("mask")

    if mask is None:
        valid_mask = np.ones((h, w), dtype=bool)
    else:
        valid_mask = np.squeeze(
            mask.detach().cpu().numpy()
        ).astype(bool)

        if valid_mask.shape != (h, w):
            valid_mask = np.ones((h, w), dtype=bool)

    return points, valid_mask, (h, w)


# ============================================================
# VOXELISATION
# ============================================================

def voxel_grid_shape():
    return (
        int(np.ceil((X_MAX - X_MIN) / VOXEL_SIZE)),
        int(np.ceil((Y_MAX - Y_MIN) / VOXEL_SIZE)),
        int(np.ceil((Z_MAX - Z_MIN) / VOXEL_SIZE)),
    )


def validate_voxel_grid():
    shape = voxel_grid_shape()

    if shape != EXPECTED_GRID_SHAPE:
        raise RuntimeError(
            f"Unexpected grid shape {shape}; "
            f"expected {EXPECTED_GRID_SHAPE}"
        )

    if not np.isclose(VOXEL_SIZE, 0.05, atol=1e-12):
        raise RuntimeError("Voxel size must be 0.05 m")

    bounds = np.array([
        X_MIN, X_MAX,
        Y_MIN, Y_MAX,
        Z_MIN, Z_MAX,
    ])

    if not np.allclose(
        bounds,
        EXPECTED_GRID_BOUNDS,
        atol=1e-12,
    ):
        raise RuntimeError("Voxel bounds mismatch")


def points_to_voxel_indices(points):
    ix = np.floor(
        (points[:, 0] - X_MIN) / VOXEL_SIZE
    ).astype(np.int32)

    iy = np.floor(
        (points[:, 1] - Y_MIN) / VOXEL_SIZE
    ).astype(np.int32)

    iz = np.floor(
        (points[:, 2] - Z_MIN) / VOXEL_SIZE
    ).astype(np.int32)

    nx, ny, nz = voxel_grid_shape()

    inside = (
        (ix >= 0) & (ix < nx)
        & (iy >= 0) & (iy < ny)
        & (iz >= 0) & (iz < nz)
    )

    return ix, iy, iz, inside


def build_semantic_voxel_map(
    points,
    valid_mask,
    segmentation_map,
):
    h, w = points.shape[:2]

    if segmentation_map.shape != (h, w):
        segmentation_map = cv2.resize(
            segmentation_map.astype(np.int16),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    valid = (
        valid_mask
        & np.isfinite(points).all(axis=-1)
    )

    flat_points = points[valid].astype(np.float32)
    flat_labels = segmentation_map[valid].astype(np.int16)

    ix, iy, iz, inside = points_to_voxel_indices(
        flat_points
    )

    ix = ix[inside]
    iy = iy[inside]
    iz = iz[inside]
    flat_labels = flat_labels[inside]

    nx, ny, nz = voxel_grid_shape()

    semantic = np.full(
        (nx, ny, nz),
        -1,
        dtype=np.int16,
    )

    occupancy = np.zeros(
        (nx, ny, nz),
        dtype=np.uint8,
    )

    voxel_linear = (ix * ny + iy) * nz + iz

    # Group reconstructed points by voxel.
    # This replaces the original repeated full-array scan,
    # while retaining majority-vote semantic assignment.
    unique_voxels, inverse = np.unique(
        voxel_linear,
        return_inverse=True,
    )

    if len(unique_voxels) == 0:
        return semantic, occupancy

    occupancy.reshape(-1)[unique_voxels] = 1

    # Background class = 0.
    # RF-DETR class IDs 0..4 map to output IDs 1..5.
    output_labels = flat_labels.astype(np.int32) + 1

    # Ignore points that have no detected class when voting.
    hazard = (
        (output_labels >= 1)
        & (output_labels <= 5)
    )

    # Efficient majority vote over 5 semantic classes.
    counts = np.zeros(
        (len(unique_voxels), 5),
        dtype=np.int32,
    )

    np.add.at(
        counts,
        (
            inverse[hazard],
            output_labels[hazard] - 1,
        ),
        1,
    )

    has_hazard = counts.sum(axis=1) > 0

    voxel_labels = np.zeros(
        len(unique_voxels),
        dtype=np.int16,
    )

    # np.argmax chooses the smallest class ID for ties,
    # consistent with np.unique + np.argmax.
    voxel_labels[has_hazard] = (
        np.argmax(counts[has_hazard], axis=1) + 1
    ).astype(np.int16)

    semantic.reshape(-1)[unique_voxels] = voxel_labels

    return semantic, occupancy


# ============================================================
# SAVE
# ============================================================

def save_voxel_map(
    path,
    semantic,
    occupancy,
    camera,
    timestamp,
    image_path,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        semantic=semantic,
        occupancy=occupancy,
        voxel_size=np.float32(VOXEL_SIZE),
        origin=np.array(
            [X_MIN, Y_MIN, Z_MIN],
            dtype=np.float32,
        ),
        bounds=np.array(
            [
                X_MIN, X_MAX,
                Y_MIN, Y_MAX,
                Z_MIN, Z_MAX,
            ],
            dtype=np.float32,
        ),
        camera=np.array(camera),
        timestamp=np.int64(timestamp),
        image_path=np.array(str(image_path)),
        class_names=OUTPUT_CLASS_NAMES,
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate semantic voxel NPZ files with timing."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of newly generated NPZ files; 0=all.",
    )

    parser.add_argument(
        "--camera",
        choices=["CAM2", "CAM6", "both"],
        default="both",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing NPZ files.",
    )

    parser.add_argument(
        "--timing-csv",
        type=Path,
        default=None,
        help="Path for generation timing CSV.",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if args.limit < 0:
        raise ValueError("--limit must be >= 0")

    selected_cameras = (
        CAMERAS
        if args.camera == "both"
        else [args.camera]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timing_csv = (
        args.timing_csv
        if args.timing_csv is not None
        else OUTPUT_DIR / "generation_times.csv"
    )

    timing_csv.parent.mkdir(parents=True, exist_ok=True)

    run_start = time.perf_counter()

    print("=" * 80)
    print("SEMANTIC VOXEL GENERATION WITH PER-NPZ TIMING")
    print("=" * 80)
    print(f"Device:       {DEVICE}")
    print(f"Cameras:      {selected_cameras}")
    print(f"Output:       {OUTPUT_DIR}")
    print(f"Timing CSV:   {timing_csv}")
    print(f"Limit:        {args.limit or 'ALL'}")
    print(f"Overwrite:    {args.overwrite}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    validate_voxel_grid()

    nx, ny, nz = voxel_grid_shape()

    print(
        f"Voxel grid: {nx} x {ny} x {nz} "
        f"= {nx * ny * nz:,} voxels"
    )
    print(f"Voxel size: {VOXEL_SIZE} m")

    # --------------------------------------------------------
    # Load calibration and file records
    # --------------------------------------------------------

    calibration = {}
    camera_records = {}

    for camera in selected_cameras:
        K, width, height = load_intrinsics(camera)

        calibration[camera] = {
            "K": K,
            "width": width,
            "height": height,
        }

        camera_records[camera] = load_images(camera)

        print(
            f"{camera}: {width}x{height}, "
            f"{len(camera_records[camera]):,} images"
        )

    lidar_records = load_lidar_files()

    print(f"LiDAR scans: {len(lidar_records):,}")

    # --------------------------------------------------------
    # Load RF-DETR once
    # --------------------------------------------------------

    print("\nLoading RF-DETR...")

    with Timer() as rf_load:
        rfdetr_model = RFDETRSegSmall.from_checkpoint(
            str(RFDETR_MODEL_PATH)
        )

        rfdetr_model.inference(
            dtype=torch.float16,
            batch_size=1,
        )

    print(
        f"RF-DETR setup: {rf_load.elapsed:.3f} s"
    )

    # --------------------------------------------------------
    # Load MapAnything once
    # --------------------------------------------------------

    print("\nLoading MapAnything...")

    with Timer() as map_load:
        model = (
            MapAnything
            .from_pretrained("facebook/map-anything")
            .to(DEVICE)
        )
        model.eval()

    print(
        f"MapAnything setup: {map_load.elapsed:.3f} s"
    )

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    timing_records = []
    skipped = 0
    failed = 0
    total_processed = 0

    stop_requested = False

    with open(timing_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=TIMING_FIELDS,
        )

        writer.writeheader()
        csv_file.flush()

        for camera in selected_cameras:

            print("\n" + "=" * 80)
            print(f"PROCESSING {camera}")
            print("=" * 80)

            records = camera_records[camera]
            camera_output = OUTPUT_DIR / camera
            camera_output.mkdir(parents=True, exist_ok=True)

            for image_index, (
                image_timestamp,
                image_path,
            ) in enumerate(records, start=1):

                if args.limit and total_processed >= args.limit:
                    stop_requested = True
                    break

                output_path = (
                    camera_output / f"{image_timestamp}.npz"
                )

                if output_path.exists() and not args.overwrite:
                    skipped += 1
                    continue

                print("\n" + "-" * 80)
                print(
                    f"[{image_index}/{len(records)}] "
                    f"{camera} | {image_timestamp}"
                )

                # Find nearest LiDAR before starting
                # generation. Timestamp lookup is treated
                # as dataset bookkeeping.
                (
                    lidar_timestamp,
                    lidar_path,
                    lidar_delta,
                ) = nearest_timestamp(
                    image_timestamp,
                    lidar_records,
                )

                if lidar_path is None:
                    print("  [SKIP] No LiDAR available")
                    skipped += 1
                    continue

                if lidar_delta > SYNC_TOLERANCE_NS:
                    print(
                        f"  WARNING: LiDAR offset "
                        f"{lidar_delta / 1e6:.2f} ms "
                        f"exceeds 50 ms tolerance."
                    )

                # The existing pipeline proceeds with the
                # nearest scan even when tolerance is exceeded.

                timings = {}
                frame_start = time.perf_counter()

                try:
                    # ----------------------------------------
                    # Image loading
                    # ----------------------------------------

                    with Timer() as t:
                        with Image.open(image_path) as im:
                            image = np.asarray(
                                im.convert("RGB")
                            )

                    timings["image_loading_s"] = t.elapsed

                    # ----------------------------------------
                    # LiDAR loading
                    # ----------------------------------------

                    with Timer() as t:
                        lidar = load_lidar_ply(lidar_path)

                    timings["lidar_loading_s"] = t.elapsed

                    # ----------------------------------------
                    # RF-DETR
                    # ----------------------------------------

                    print("  Running RF-DETR...")

                    with Timer() as t:
                        segmentation_map = (
                            run_rfdetr_segmentation(
                                rfdetr_model,
                                image,
                            )
                        )

                    timings["rfdetr_s"] = t.elapsed

                    # ----------------------------------------
                    # MapAnything
                    # ----------------------------------------

                    print("  Running MapAnything...")

                    with Timer() as t:
                        (
                            points,
                            valid_mask,
                            point_shape,
                        ) = run_mapanything_single(
                            model,
                            image,
                            lidar,
                            camera,
                            calibration[camera]["K"],
                            calibration[camera]["width"],
                            calibration[camera]["height"],
                        )

                    timings["mapanything_s"] = t.elapsed

                    # ----------------------------------------
                    # Voxelisation
                    # ----------------------------------------

                    print("  Voxelising...")

                    with Timer() as t:
                        (
                            semantic,
                            occupancy,
                        ) = build_semantic_voxel_map(
                            points,
                            valid_mask,
                            segmentation_map,
                        )

                        occupied_count = int(
                            np.count_nonzero(occupancy)
                        )

                        hazard_count = int(
                            np.count_nonzero(semantic > 0)
                        )

                    timings["voxelisation_s"] = t.elapsed

                    # ----------------------------------------
                    # NPZ saving
                    # ----------------------------------------

                    print("  Saving NPZ...")

                    with Timer() as t:
                        save_voxel_map(
                            output_path,
                            semantic,
                            occupancy,
                            camera,
                            image_timestamp,
                            image_path,
                        )

                    timings["npz_saving_s"] = t.elapsed

                    # ----------------------------------------
                    # Total generation time
                    # ----------------------------------------

                    sync_gpu()

                    timings["total_s"] = (
                        time.perf_counter() - frame_start
                    )

                    stage_total = sum(
                        timings[key]
                        for key in [
                            "image_loading_s",
                            "lidar_loading_s",
                            "rfdetr_s",
                            "mapanything_s",
                            "voxelisation_s",
                            "npz_saving_s",
                        ]
                    )

                    timings["other_s"] = max(
                        0.0,
                        timings["total_s"] - stage_total,
                    )

                    # ----------------------------------------
                    # Record CSV
                    # ----------------------------------------

                    record = {
                        "camera": camera,
                        "timestamp": image_timestamp,
                        "npz_path": str(output_path),
                        **timings,
                        "occupied_voxels": occupied_count,
                        "hazard_voxels": hazard_count,
                    }

                    writer.writerow(record)
                    csv_file.flush()

                    timing_records.append(record)
                    total_processed += 1

                    running_average = np.mean(
                        [r["total_s"] for r in timing_records]
                    )

                    print(
                        f"  Occupied voxels: {occupied_count:,}"
                    )

                    print(
                        f"  Hazard voxels:   {hazard_count:,}"
                    )

                    print_timing(
                        timings,
                        running_average,
                        total_processed,
                    )

                    print(f"\n  SAVED -> {output_path}")

                except Exception as exc:
                    failed += 1

                    # Remove incomplete output generated by
                    # a failed save operation.
                    if output_path.exists():
                        try:
                            output_path.unlink()
                        except OSError:
                            pass

                    print(
                        f"  [FAILED] {camera} "
                        f"{image_timestamp}: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    raise

                finally:
                    # Preserve original cache-clearing behavior.
                    if DEVICE.type == "cuda":
                        torch.cuda.empty_cache()

            if stop_requested:
                break

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    wall_time = time.perf_counter() - run_start

    print_summary(
        timing_records,
        wall_time,
        skipped,
        failed,
    )

    print("\nOUTPUT FILES")
    print(f"  Voxels: {OUTPUT_DIR}")
    print(f"  Timing: {timing_csv}")

    print("\nDONE")


if __name__ == "__main__":
    main()
