#!/usr/bin/env python3

"""
Generate ONE high-resolution semantic 3D voxel map for EVERY image in CAM2 and CAM6.

NEW ROI:
    X = -8.0 .. 8.0 m
    Y =  0.0 .. 2.0 m
    Z =  0.25 .. 10.0 m

Voxel size:
    0.05 m (5 cm)

Grid:
    320 x 40 x 195
    = 2,496,000 voxels per image

Every image is processed independently.
CAM2 and CAM6 are never merged.
LiDAR is used as sparse depth input to MapAnything.
RF-DETR provides the 2D semantic labels.
MapAnything pts3d_cam is camera-relative.
"""

import csv
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

CAMERAS = ["CAM2", "CAM6"]

SYNC_TOLERANCE_NS = 50_000_000

LIDAR_MAX_RANGE = 50.0
LIDAR_MIN_CAMERA_DEPTH = 0.05
BORDER_MARGIN_PIXELS = 2


# ============================================================
# VOXEL GRID
# ============================================================

VOXEL_SIZE = 0.05  # 5 cm

# NEW ROI
X_MIN, X_MAX = -8.0, 8.0
Y_MIN, Y_MAX = 0.0, 2.0
Z_MIN, Z_MAX = 0.25, 10.0

# Expected grid:
# X: 16.00 / 0.05 = 320
# Y:  2.00 / 0.05 = 40
# Z:  9.75 / 0.05 = 195
EXPECTED_GRID_SHAPE = (320, 40, 195)

EXPECTED_GRID_BOUNDS = (
    -8.0, 8.0,
    0.0, 2.0,
    0.25, 10.0,
)

OUTPUT_DIR = BAG_ROOT / "semantic_voxels_new_roi_5cm"


# ============================================================
# RF-DETR
# ============================================================

RFDETR_MODEL_PATH = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/"
    "2D-Seg-Road-Nissan.v13i.coco-segmentation/"
    "outputs/checkpoint_best_total.pth"
)

RFDETR_CONFIDENCE_THRESHOLD = 0.30

# RF-DETR dataset IDs
CLASS_NAMES = {
    0: "fallen_fence",
    1: "loose_trash",
    2: "mud_spill",
    3: "public_greenway",
    4: "sidewalk",
}

# Output voxel IDs
# 0 = background
# 1..5 = hazard classes
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
# LiDAR -> CAMERA EXTRINSICS
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
        value
        .replace(")", "")
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
        raise RuntimeError(
            f"Could not parse array: {value}"
        )

    return values


def load_intrinsics(camera):
    path = EXTRACTED / camera / "camera_info.csv"

    with open(path, "r") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(
            f"No camera info found: {path}"
        )

    row = rows[0]

    P = parse_array(row["P"])

    if P.size != 12:
        raise RuntimeError(
            f"{camera}: invalid P size {P.size}"
        )

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
        raise RuntimeError(
            f"No images found: {directory}"
        )

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
        raise RuntimeError(
            f"No LiDAR files found: {directory}"
        )

    return records


def nearest_timestamp(target, records):
    timestamps = np.asarray(
        [x[0] for x in records],
        dtype=np.int64,
    )

    index = int(
        np.searchsorted(timestamps, target)
    )

    candidates = []

    if index > 0:
        candidates.append(index - 1)

    if index < len(records):
        candidates.append(index)

    if not candidates:
        return None, None, None

    best = min(
        candidates,
        key=lambda i: abs(
            int(records[i][0]) - int(target)
        ),
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
                raise RuntimeError(
                    f"Unexpected EOF: {path}"
                )

            line = line.strip()
            header.append(line)

            if line == "end_header":
                break

    vertex_count = None

    for line in header:
        parts = line.split()

        if (
            len(parts) == 3
            and parts[0] == "element"
            and parts[1] == "vertex"
        ):
            vertex_count = int(parts[2])
            break

    if vertex_count is None:
        raise RuntimeError(
            f"No vertex count found: {path}"
        )

    properties = []
    in_vertex = False

    for line in header:
        parts = line.split()

        if (
            len(parts) >= 3
            and parts[0] == "element"
            and parts[1] == "vertex"
        ):
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

    data = np.loadtxt(
        path,
        skiprows=len(header),
        max_rows=vertex_count,
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

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

    ranges = np.linalg.norm(xyz, axis=1)

    return xyz[
        ranges <= LIDAR_MAX_RANGE
    ]


# ============================================================
# LiDAR DEPTH
# ============================================================

def transform_points(points, T):
    homogeneous = np.concatenate(
        [
            points,
            np.ones(
                (len(points), 1)
            ),
        ],
        axis=1,
    )

    return (
        homogeneous @ T.T
    )[:, :3]


def project_lidar(
    points,
    camera,
    K,
    width,
    height,
):
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
        K[0, 0] * X / Z
        + K[0, 2]
    ).astype(np.intp)

    v = np.rint(
        K[1, 1] * Y / Z
        + K[1, 2]
    ).astype(np.intp)

    inside = (
        (u >= BORDER_MARGIN_PIXELS)
        & (u < width - BORDER_MARGIN_PIXELS)
        & (v >= BORDER_MARGIN_PIXELS)
        & (v < height - BORDER_MARGIN_PIXELS)
    )

    return (
        u[inside],
        v[inside],
        Z[inside],
    )


def lidar_depth_image(
    points,
    camera,
    K,
    width,
    height,
):
    u, v, depths = project_lidar(
        points,
        camera,
        K,
        width,
        height,
    )

    depth = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    if len(u) == 0:
        return depth

    pixel_ids = (
        v.astype(np.int64) * width
        + u.astype(np.int64)
    )

    order = np.lexsort(
        (depths, pixel_ids)
    )

    sorted_pixels = pixel_ids[order]

    keep = np.ones(
        len(order),
        dtype=bool,
    )

    if len(order) > 1:
        keep[1:] = (
            sorted_pixels[1:]
            != sorted_pixels[:-1]
        )

    selected = order[keep]

    depth[
        v[selected],
        u[selected],
    ] = depths[selected].astype(
        np.float32
    )

    return depth


# ============================================================
# RF-DETR
# ============================================================

def normalise_rfdetr_mask(
    mask,
    image_height,
    image_width,
):
    if torch.is_tensor(mask):
        mask = (
            mask
            .detach()
            .cpu()
            .numpy()
        )

    mask = np.asarray(mask)
    mask = np.squeeze(mask)

    if mask.dtype != np.bool_:
        mask = mask > 0.5

    if mask.ndim != 2:
        raise ValueError(
            f"Unexpected RF-DETR mask shape: "
            f"{mask.shape}"
        )

    if mask.shape != (
        image_height,
        image_width,
    ):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (
                image_width,
                image_height,
            ),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    return mask


def run_rfdetr_segmentation(
    model,
    image,
):
    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    with torch.inference_mode():
        detections_batch = model.predict(
            [image_bgr]
        )

    detections = detections_batch[0]

    image_height, image_width = (
        image.shape[:2]
    )

    if not hasattr(
        detections,
        "mask",
    ):
        raise RuntimeError(
            "RF-DETR returned no segmentation masks."
        )

    masks = detections.mask
    class_ids = detections.class_id
    confidences = detections.confidence

    ordered = sorted(
        range(len(class_ids)),
        key=lambda i: float(
            confidences[i]
        ),
        reverse=True,
    )

    point_class = np.full(
        (
            image_height,
            image_width,
        ),
        -1,
        dtype=np.int16,
    )

    for detection_index in ordered:

        confidence = float(
            confidences[detection_index]
        )

        class_id = int(
            class_ids[detection_index]
        )

        if (
            confidence
            < RFDETR_CONFIDENCE_THRESHOLD
        ):
            continue

        if class_id not in CLASS_NAMES:
            continue

        mask = normalise_rfdetr_mask(
            masks[detection_index],
            image_height,
            image_width,
        )

        assign = (
            mask
            & (point_class == -1)
        )

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
        "intrinsics": K.astype(
            np.float32
        ),
        "depth_z": torch.from_numpy(
            depth
        ).float(),
        "is_metric_scale": torch.tensor(
            [True],
            device=DEVICE,
        ),
    }

    processed = preprocess_inputs(
        [view]
    )

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

    # Camera-relative 3D coordinates
    points = (
        prediction["pts3d_cam"]
        .detach()
        .cpu()
        .numpy()
    )

    while (
        points.ndim > 3
        and points.shape[0] == 1
    ):
        points = points[0]

    if (
        points.ndim != 3
        or points.shape[-1] != 3
    ):
        raise RuntimeError(
            f"Unexpected pts3d_cam shape: "
            f"{points.shape}"
        )

    h, w = points.shape[:2]

    mask = prediction.get("mask")

    if mask is None:
        valid_mask = np.ones(
            (h, w),
            dtype=bool,
        )

    else:
        valid_mask = np.squeeze(
            mask.detach()
            .cpu()
            .numpy()
        ).astype(bool)

        if valid_mask.shape != (
            h,
            w,
        ):
            valid_mask = np.ones(
                (h, w),
                dtype=bool,
            )

    return points, valid_mask, (h, w)


# ============================================================
# VOXELISATION
# ============================================================

def voxel_grid_shape():
    nx = int(
        np.ceil(
            (X_MAX - X_MIN)
            / VOXEL_SIZE
        )
    )

    ny = int(
        np.ceil(
            (Y_MAX - Y_MIN)
            / VOXEL_SIZE
        )
    )

    nz = int(
        np.ceil(
            (Z_MAX - Z_MIN)
            / VOXEL_SIZE
        )
    )

    return nx, ny, nz


def validate_voxel_grid():

    shape = voxel_grid_shape()

    if shape != EXPECTED_GRID_SHAPE:
        raise RuntimeError(
            f"Unexpected 5 cm voxel grid: "
            f"{shape}; expected "
            f"{EXPECTED_GRID_SHAPE}"
        )

    if not np.isclose(
        VOXEL_SIZE,
        0.05,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"Unexpected voxel size: "
            f"{VOXEL_SIZE}; expected 0.05 m"
        )

    actual_bounds = np.array(
        [
            X_MIN,
            X_MAX,
            Y_MIN,
            Y_MAX,
            Z_MIN,
            Z_MAX,
        ],
        dtype=np.float64,
    )

    expected_bounds = np.array(
        EXPECTED_GRID_BOUNDS,
        dtype=np.float64,
    )

    if not np.allclose(
        actual_bounds,
        expected_bounds,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"Bounds mismatch: "
            f"actual={actual_bounds}, "
            f"expected={expected_bounds}"
        )

    extent = np.array(
        [
            X_MAX - X_MIN,
            Y_MAX - Y_MIN,
            Z_MAX - Z_MIN,
        ],
        dtype=np.float64,
    )

    expected_extent = (
        np.array(
            shape,
            dtype=np.float64,
        )
        * VOXEL_SIZE
    )

    if not np.allclose(
        extent,
        expected_extent,
        atol=1e-8,
    ):
        raise RuntimeError(
            f"Voxel geometry mismatch: "
            f"bounds extent={extent}, "
            f"shape*voxel_size="
            f"{expected_extent}"
        )


def points_to_voxel_indices(
    points,
):
    ix = np.floor(
        (points[:, 0] - X_MIN)
        / VOXEL_SIZE
    ).astype(np.int32)

    iy = np.floor(
        (points[:, 1] - Y_MIN)
        / VOXEL_SIZE
    ).astype(np.int32)

    iz = np.floor(
        (points[:, 2] - Z_MIN)
        / VOXEL_SIZE
    ).astype(np.int32)

    nx, ny, nz = voxel_grid_shape()

    inside = (
        (ix >= 0)
        & (ix < nx)
        & (iy >= 0)
        & (iy < ny)
        & (iz >= 0)
        & (iz < nz)
    )

    return ix, iy, iz, inside


def build_semantic_voxel_map(
    points,
    valid_mask,
    segmentation_map,
):
    h, w = points.shape[:2]

    if segmentation_map.shape != (
        h,
        w,
    ):
        segmentation_map = cv2.resize(
            segmentation_map.astype(
                np.int16
            ),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    valid = (
        valid_mask
        & np.isfinite(points).all(
            axis=-1
        )
    )

    flat_points = points[valid].astype(
        np.float32
    )

    flat_labels = segmentation_map[
        valid
    ].astype(np.int16)

    ix, iy, iz, inside = (
        points_to_voxel_indices(
            flat_points
        )
    )

    ix = ix[inside]
    iy = iy[inside]
    iz = iz[inside]

    flat_labels = flat_labels[
        inside
    ]

    nx, ny, nz = voxel_grid_shape()

    # -1 = unknown/unobserved
    #  0 = observed point, no hazard
    #  1..5 = hazard class

    semantic = np.full(
        (nx, ny, nz),
        -1,
        dtype=np.int16,
    )

    # 1 = at least one reconstructed
    # point in this voxel
    occupancy = np.zeros(
        (nx, ny, nz),
        dtype=np.uint8,
    )

    # Linear voxel index
    voxel_linear = (
        (ix * ny + iy) * nz
        + iz
    )

    unique_voxels = np.unique(
        voxel_linear
    )

    for linear in unique_voxels:

        selected = (
            voxel_linear == linear
        )

        vx = int(
            linear // (ny * nz)
        )

        remainder = int(
            linear % (ny * nz)
        )

        vy = remainder // nz
        vz = remainder % nz

        occupancy[
            vx,
            vy,
            vz,
        ] = 1

        labels = flat_labels[
            selected
        ]

        # RF-DETR -1 means no detected
        # hazard
        hazard_labels = labels[
            labels >= 0
        ]

        if len(hazard_labels) == 0:

            semantic[
                vx,
                vy,
                vz,
            ] = 0

            continue

        output_labels = np.array(
            [
                OUTPUT_CLASS_IDS[
                    int(label)
                ]
                for label in hazard_labels
                if int(label)
                in OUTPUT_CLASS_IDS
            ],
            dtype=np.int16,
        )

        if len(output_labels) == 0:

            semantic[
                vx,
                vy,
                vz,
            ] = 0

        else:

            values, counts = np.unique(
                output_labels,
                return_counts=True,
            )

            semantic[
                vx,
                vy,
                vz,
            ] = int(
                values[
                    np.argmax(counts)
                ]
            )

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
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        semantic=semantic,
        occupancy=occupancy,
        voxel_size=np.float32(
            VOXEL_SIZE
        ),
        origin=np.array(
            [
                X_MIN,
                Y_MIN,
                Z_MIN,
            ],
            dtype=np.float32,
        ),
        bounds=np.array(
            [
                X_MIN,
                X_MAX,
                Y_MIN,
                Y_MAX,
                Z_MIN,
                Z_MAX,
            ],
            dtype=np.float32,
        ),
        camera=np.array(camera),
        timestamp=np.int64(
            timestamp
        ),
        image_path=np.array(
            str(image_path)
        ),
        class_names=OUTPUT_CLASS_NAMES,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "PER-IMAGE SEMANTIC VOXEL GENERATION"
    )
    print("=" * 80)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    nx, ny, nz = voxel_grid_shape()

    validate_voxel_grid()

    print(
        f"Voxel size: "
        f"{VOXEL_SIZE:.3f} m (5 cm)"
    )

    print(
        f"Voxel grid: "
        f"{nx} x {ny} x {nz}"
    )

    print(
        f"Total voxels per image: "
        f"{nx * ny * nz:,}"
    )

    print(
        f"Bounds: "
        f"X={X_MIN}..{X_MAX}, "
        f"Y={Y_MIN}..{Y_MAX}, "
        f"Z={Z_MIN}..{Z_MAX}"
    )

    print(
        f"Expected grid shape: "
        f"{EXPECTED_GRID_SHAPE}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )

    print()

    print("IMPORTANT:")
    print(
        "  Every image is processed independently."
    )
    print(
        "  MapAnything receives ONE image at a time."
    )
    print(
        "  CAM2 and CAM6 remain completely separate."
    )

    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Calibration / image records
    # --------------------------------------------------------

    calibration = {}
    camera_records = {}

    for camera in CAMERAS:

        K, width, height = (
            load_intrinsics(
                camera
            )
        )

        calibration[camera] = {
            "K": K,
            "width": width,
            "height": height,
        }

        camera_records[camera] = (
            load_images(camera)
        )

        print(
            f"{camera}: "
            f"{width} x {height}, "
            f"{len(camera_records[camera]):,} images"
        )

    lidar_records = load_lidar_files()

    print(
        f"LiDAR files: "
        f"{len(lidar_records):,}"
    )

    # --------------------------------------------------------
    # Load RF-DETR once
    # --------------------------------------------------------

    print(
        "\nLoading RF-DETR..."
    )

    rfdetr_model = (
        RFDETRSegSmall.from_checkpoint(
            str(RFDETR_MODEL_PATH)
        )
    )

    print(
        "Compiling RF-DETR FP16 inference..."
    )

    rfdetr_model.inference(
        dtype=torch.float16,
        batch_size=1,
    )

    print(
        "RF-DETR ready."
    )

    # --------------------------------------------------------
    # Load MapAnything once
    # --------------------------------------------------------

    print(
        "\nLoading MapAnything..."
    )

    model = (
        MapAnything
        .from_pretrained(
            "facebook/map-anything"
        )
        .to(DEVICE)
    )

    model.eval()

    print(
        "MapAnything loaded."
    )

    # --------------------------------------------------------
    # Process every image
    # --------------------------------------------------------

    total_processed = 0
    total_skipped = 0

    for camera in CAMERAS:

        print(
            "\n" + "=" * 80
        )

        print(
            f"PROCESSING ALL "
            f"{camera} IMAGES"
        )

        print(
            "=" * 80
        )

        camera_output = (
            OUTPUT_DIR / camera
        )

        camera_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        records = camera_records[
            camera
        ]

        for image_index, (
            image_timestamp,
            image_path,
        ) in enumerate(records):

            output_path = (
                camera_output
                / f"{image_timestamp}.npz"
            )

            if output_path.exists():

                print(
                    f"[{image_index + 1}/"
                    f"{len(records)}] "
                    f"{camera} "
                    f"{image_timestamp}: "
                    f"already exists, "
                    f"skipping"
                )

                total_skipped += 1
                continue

            print(
                "\n" + "-" * 80
            )

            print(
                f"[{image_index + 1}/"
                f"{len(records)}] "
                f"{camera}"
            )

            print(
                f"Timestamp: "
                f"{image_timestamp}"
            )

            print(
                f"Image: "
                f"{image_path}"
            )

            # ------------------------------------------------
            # Load image
            # ------------------------------------------------

            image = np.asarray(
                Image.open(
                    image_path
                ).convert("RGB")
            )

            # ------------------------------------------------
            # Find synchronized LiDAR
            # ------------------------------------------------

            (
                lidar_timestamp,
                lidar_path,
                lidar_delta,
            ) = nearest_timestamp(
                image_timestamp,
                lidar_records,
            )

            if lidar_path is None:

                print(
                    "  WARNING: no "
                    "synchronized LiDAR; "
                    "skipping image."
                )

                total_skipped += 1
                continue

            print(
                f"LiDAR: "
                f"{lidar_path.name} "
                f"(delta="
                f"{lidar_delta / 1e6:.2f} ms)"
            )

            if (
                lidar_delta
                > SYNC_TOLERANCE_NS
            ):

                print(
                    "  WARNING: LiDAR "
                    "synchronization exceeds "
                    "tolerance."
                )

            lidar = load_lidar_ply(
                lidar_path
            )

            # ------------------------------------------------
            # RF-DETR
            # ------------------------------------------------

            print(
                "  RF-DETR segmentation..."
            )

            segmentation_map = (
                run_rfdetr_segmentation(
                    rfdetr_model,
                    image,
                )
            )

            # ------------------------------------------------
            # MapAnything
            # ------------------------------------------------

            print(
                "  MapAnything single-image "
                "RGB + LiDAR inference..."
            )

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

            # ------------------------------------------------
            # Voxelise
            # ------------------------------------------------

            print(
                "  Voxelising..."
            )

            (
                semantic,
                occupancy,
            ) = build_semantic_voxel_map(
                points,
                valid_mask,
                segmentation_map,
            )

            occupied_count = int(
                occupancy.sum()
            )

            hazard_count = int(
                np.count_nonzero(
                    semantic > 0
                )
            )

            print(
                f"  Occupied voxels: "
                f"{occupied_count:,}"
            )

            print(
                f"  Hazard voxels: "
                f"{hazard_count:,}"
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            save_voxel_map(
                output_path,
                semantic,
                occupancy,
                camera,
                image_timestamp,
                image_path,
            )

            print(
                f"  SAVED -> "
                f"{output_path}"
            )

            total_processed += 1

            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    print(
        "\n" + "=" * 80
    )

    print("DONE")

    print(
        "=" * 80
    )

    print(
        f"Processed: "
        f"{total_processed:,}"
    )

    print(
        f"Skipped:   "
        f"{total_skipped:,}"
    )

    print(
        f"Output:    "
        f"{OUTPUT_DIR}"
    )

    print()

    print(
        "Each image now has its own "
        "independent camera-relative "
        "semantic voxel map."
    )


if __name__ == "__main__":
    main()