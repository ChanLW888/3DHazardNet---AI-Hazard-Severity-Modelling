
#!/usr/bin/env python3
"""
MCAV cached 3DHazardNet -> GPS DBSCAN -> Severity -> Hazard Map.

Uses already generated:
    sample_XXXXX_CAM2/
        image_original.*
        prediction.npz
        rfdetr_prior.npz
        metadata.json

No neural-network inference is performed.

Outputs:
    frame_events.csv
    clusters.csv
    route.csv
    hazard_map.html
    pipeline_summary.json

    cluster_XXXX/
        events.csv
        metadata.json
        severity_metrics.json
        representative_rgb.jpg
        representative_overlay.jpg
        voxel_points_by_class.npz
        voxel_3d.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import importlib.util
import json
import math

from collections import Counter, defaultdict
from pathlib import Path

import cv2
import folium
import numpy as np

from sklearn.cluster import DBSCAN


# ============================================================
# CONFIGURATION
# ============================================================

CLASSES = (
    "loose_trash",
    "mud_spill",
    "public_greenway",
    "sidewalk",
)

HAZARDS = (
    "loose_trash",
    "mud_spill",
    "fallen_fence",
)

ASSETS = (
    "public_greenway",
    "sidewalk",
)

VOXEL_ID = {
    name: i
    for i, name in enumerate(CLASSES)
}

# OpenCV uses BGR.
BGR = {
    "loose_trash": (0, 0, 255),
    "mud_spill": (0, 165, 255),
    "fallen_fence": (220, 70, 180),
    "public_greenway": (50, 205, 50),
    "sidewalk": (220, 220, 0),
}

STAR_COLOUR = {
    None: "#777777",
    1: "#2ca25f",
    2: "#99d8c9",
    3: "#fec44f",
    4: "#fe9929",
    5: "#de2d26",
}

DEFAULT_BASE = Path(
    "/fs04/scratch2/bt60/lcha_data/"
    "map-anything/personal_data/hazard_test_101"
)


# ============================================================
# PATHS AND IMPORTS
# ============================================================

def find_existing_default(suffix):
    bases = (
        DEFAULT_BASE,
        Path(
            "/scratch/bt60/lcha_data/"
            "map-anything/personal_data/hazard_test_101"
        ),
        Path(
            "/home/lcha0115/bt60_scratch/lcha_data/"
            "map-anything/personal_data/hazard_test_101"
        ),
    )

    for base in bases:
        candidate = base / suffix

        if candidate.exists():
            return candidate

    return DEFAULT_BASE / suffix


def load_severity(script):
    script = Path(script).expanduser().resolve()

    if not script.is_file():
        raise FileNotFoundError(
            f"severity.py not found: {script}\n"
            "Supply --severity-script /path/to/severity.py"
        )

    spec = importlib.util.spec_from_file_location(
        "cached_mcav_severity",
        str(script),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import {script}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = (
        "calculate_hazard_severity",
        "calculate_location_severity",
        "save_severity_metrics",
    )

    for name in required:
        if not hasattr(module, name):
            raise RuntimeError(
                f"{script} missing {name}"
            )

    return module


# ============================================================
# SAVE UTILITIES
# ============================================================

def write_csv(path, rows, fields):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".partial"
    )

    with tmp.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)

    tmp.replace(path)


def write_json(path, obj):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".partial"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    tmp.replace(path)


# ============================================================
# GPS LOADING
# ============================================================

def find_column(columns, names):
    lower = {
        c.strip().lower(): c
        for c in columns
    }

    for name in names:
        if name in lower:
            return lower[name]

    for name in names:
        for col in columns:
            if name in col.strip().lower():
                return col

    return None


def load_gps(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"GPS CSV missing: {csv_path}"
        )

    with csv_path.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        tcol = find_column(
            fields,
            (
                "ros_timestamp_ns",
                "timestamp_ns",
                "timestamp",
                "time_ns",
            ),
        )

        latcol = find_column(
            fields,
            (
                "latitude_deg",
                "latitude",
                "lat",
            ),
        )

        loncol = find_column(
            fields,
            (
                "longitude_deg",
                "longitude",
                "lon",
                "lng",
            ),
        )

        if not all((tcol, latcol, loncol)):
            raise ValueError(
                f"Cannot identify GPS columns: {fields}"
            )

        records = []

        for row in reader:
            try:
                # Avoid float conversion of nanosecond timestamps.
                timestamp = int(row[tcol])

                latitude = float(row[latcol])
                longitude = float(row[loncol])

                valid = (
                    math.isfinite(latitude)
                    and math.isfinite(longitude)
                    and -90 <= latitude <= 90
                    and -180 <= longitude <= 180
                    and (
                        latitude != 0
                        or longitude != 0
                    )
                )

                if valid:
                    records.append(
                        (
                            timestamp,
                            latitude,
                            longitude,
                        )
                    )

            except (ValueError, KeyError, TypeError):
                continue

    if len(records) < 2:
        raise ValueError(
            "Fewer than two valid GPS fixes"
        )

    records.sort(
        key=lambda row: row[0]
    )

    # Keep one record for each timestamp.
    deduplicated = {
        record[0]: record
        for record in records
    }

    records = list(
        deduplicated.values()
    )

    timestamps = np.array(
        [row[0] for row in records],
        dtype=np.int64,
    )

    latitudes = np.array(
        [row[1] for row in records],
        dtype=np.float64,
    )

    longitudes = np.array(
        [row[2] for row in records],
        dtype=np.float64,
    )

    print(
        f"GPS fixes: {len(records):,} | "
        f"{timestamps[0]} to {timestamps[-1]}"
    )

    return (
        timestamps,
        latitudes,
        longitudes,
    )


def gps_at(
    gps_ts,
    gps_lat,
    gps_lon,
    timestamp,
    max_dt_ms,
):
    timestamp = int(timestamp)

    # Do not invent positions outside the GPS recording.
    if (
        timestamp < int(gps_ts[0])
        or timestamp > int(gps_ts[-1])
    ):
        return None

    index = int(
        np.searchsorted(
            gps_ts,
            timestamp,
        )
    )

    index = max(
        0,
        min(index, len(gps_ts) - 1),
    )

    candidates = {
        index,
        max(0, index - 1),
    }

    nearest = min(
        candidates,
        key=lambda i: abs(
            int(gps_ts[i]) - timestamp
        ),
    )

    delta_ms = (
        abs(
            int(gps_ts[nearest]) - timestamp
        )
        / 1e6
    )

    if delta_ms > max_dt_ms:
        return None

    # Interpolate between GPS fixes.
    if (
        index == 0
        or int(gps_ts[index]) == timestamp
    ):
        latitude = float(
            gps_lat[index]
        )

        longitude = float(
            gps_lon[index]
        )

    else:
        previous = index - 1

        alpha = (
            timestamp - int(gps_ts[previous])
        ) / (
            int(gps_ts[index])
            - int(gps_ts[previous])
        )

        latitude = float(
            gps_lat[previous]
            + alpha * (
                gps_lat[index]
                - gps_lat[previous]
            )
        )

        longitude = float(
            gps_lon[previous]
            + alpha * (
                gps_lon[index]
                - gps_lon[previous]
            )
        )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "gps_timestamp": int(
            gps_ts[nearest]
        ),
        "gps_dt_ms": round(
            delta_ms,
            3,
        ),
    }


# ============================================================
# FIND ALL SAVED INFERENCE FRAMES
# ============================================================

def image_from_folder(folder, metadata):
    saved = sorted(
        folder.glob("image_original.*")
    )

    saved = [
        p
        for p in saved
        if p.suffix.lower()
        in (".png", ".jpg", ".jpeg", ".webp")
    ]

    if saved:
        return saved[0]

    original = metadata.get(
        "original_image"
    )

    if original and Path(original).is_file():
        return Path(original)

    return None


def list_saved_samples(
    root,
    camera,
    start_ts,
    end_ts,
):
    items = []
    stats = Counter()

    folders = sorted(
        root.glob("sample_*")
    )

    for folder in folders:
        if not folder.is_dir():
            continue

        stats["sample_folders"] += 1

        metadata_path = (
            folder / "metadata.json"
        )

        prediction_path = (
            folder / "prediction.npz"
        )

        prior_path = (
            folder / "rfdetr_prior.npz"
        )

        if not all(
            p.is_file()
            for p in (
                metadata_path,
                prediction_path,
                prior_path,
            )
        ):
            stats["missing_saved_files"] += 1
            continue

        try:
            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            index = int(
                metadata["sample_index"]
            )

            cam = str(
                metadata["camera"]
            ).upper()

            timestamp = int(
                metadata["timestamp"]
            )

            if cam not in ("CAM2", "CAM6"):
                stats["unsupported_camera"] += 1
                continue

            expected_prefix = (
                f"sample_{index:05d}_{cam}"
            )

            if not folder.name.startswith(
                expected_prefix
            ):
                stats["folder_metadata_mismatch"] += 1
                continue

            if (
                camera != "all"
                and cam != camera
            ):
                continue

            if (
                start_ts is not None
                and timestamp < start_ts
            ):
                continue

            if (
                end_ts is not None
                and timestamp > end_ts
            ):
                continue

            image = image_from_folder(
                folder,
                metadata,
            )

            if image is None:
                stats["missing_rgb"] += 1
                continue

            items.append({
                "index": index,
                "camera": cam,
                "timestamp": timestamp,
                "folder": folder,
                "image": image,
            })

        except (
            OSError,
            KeyError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):
            stats["bad_metadata"] += 1

    # The master sample index is NOT assumed chronological.
    items.sort(
        key=lambda row: (
            row["timestamp"],
            row["camera"],
            row["index"],
        )
    )

    seen = set()
    unique = []

    for sample in items:
        key = (
            sample["index"],
            sample["camera"],
        )

        if key not in seen:
            unique.append(sample)
            seen.add(key)

    stats["usable_samples"] = len(unique)

    return unique, stats


# ============================================================
# READ CACHED RF-DETR SEGMENTATION PRIORS
# ============================================================

def read_prior(
    folder,
    threshold,
    min_hazard_pixels,
    min_asset_pixels,
):
    """
    The saved full prior has channels:

    0 background
    1 fallen_fence
    2 loose_trash
    3 mud_spill
    4 public_greenway
    5 sidewalk

    A hazard event requires sufficient pixels above
    the confidence threshold.

    No RF-DETR model is loaded.
    """

    prior_path = (
        folder / "rfdetr_prior.npz"
    )

    with np.load(
        prior_path,
        allow_pickle=False,
    ) as data:

        if "full_rfdetr_prior" in data:
            full = data["full_rfdetr_prior"]

            if (
                full.ndim != 3
                or full.shape[0] != 6
            ):
                raise ValueError(
                    f"Invalid full prior: {full.shape}"
                )

            channels = {
                "fallen_fence": full[1],
                "loose_trash": full[2],
                "mud_spill": full[3],
                "public_greenway": full[4],
                "sidewalk": full[5],
            }

        elif "prior" in data:
            prior = data["prior"]

            if (
                prior.ndim != 3
                or prior.shape[0] != 4
            ):
                raise ValueError(
                    f"Invalid prior: {prior.shape}"
                )

            channels = {
                cls: prior[i]
                for i, cls in enumerate(CLASSES)
            }

        else:
            raise ValueError(
                "No RF-DETR channels in saved NPZ"
            )

        evidence = {}

        for cls, plane in channels.items():
            mask = (
                plane >= threshold
            )

            count = int(
                np.count_nonzero(mask)
            )

            required = (
                min_hazard_pixels
                if cls in HAZARDS
                else min_asset_pixels
            )

            if count >= required:
                evidence[cls] = {
                    "confidence": round(
                        float(
                            np.max(plane[mask])
                        ),
                        6,
                    ),
                    "pixel_count": count,
                }

    return evidence


# ============================================================
# GENERATE HAZARD FRAME EVENTS
# ============================================================

def make_events(samples, gps, args):
    gps_ts, gps_lat, gps_lon = gps

    events = []
    stats = Counter()

    for n, sample in enumerate(
        samples,
        start=1,
    ):
        try:
            evidence = read_prior(
                sample["folder"],
                args.threshold,
                args.min_hazard_pixels,
                args.min_asset_pixels,
            )

        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
        ):
            stats["bad_priors"] += 1
            continue

        hazard_names = [
            cls
            for cls in HAZARDS
            if cls in evidence
        ]

        # Sidewalk/greenway alone must NOT create an event.
        if not hazard_names:
            stats["no_2d_hazard"] += 1
            continue

        gps_position = gps_at(
            gps_ts,
            gps_lat,
            gps_lon,
            sample["timestamp"],
            args.gps_max_dt_ms,
        )

        if gps_position is None:
            stats["gps_unmatched"] += 1
            continue

        event = {
            **sample,
            **gps_position,
            "evidence": evidence,
            "hazards": hazard_names,
            "assets": [
                cls
                for cls in ASSETS
                if cls in evidence
            ],
        }

        events.append(event)

        if n % 1000 == 0:
            print(
                f"Scanned {n:,}/{len(samples):,} | "
                f"hazard events: {len(events):,}",
                flush=True,
            )

    stats["events"] = len(events)

    return events, stats


# ============================================================
# GPS DBSCAN
# ============================================================

def cluster_events(
    events,
    eps_m,
    min_samples,
):
    """
    DBSCAN clusters FRAME EVENTS, not masks.

    One image with multiple hazard classes
    counts as one frame event.

    Same local tangent-plane GPS approximation
    as the original hazard_dataset_pipeline.py.
    """

    if not events:
        return []

    latitude0 = float(
        np.mean([
            e["latitude"]
            for e in events
        ])
    )

    longitude0 = float(
        np.mean([
            e["longitude"]
            for e in events
        ])
    )

    rad = math.pi / 180.0
    earth_radius = 6371000.0

    coordinates = np.array([
        (
            (
                e["longitude"] - longitude0
            )
            * rad
            * earth_radius
            * math.cos(latitude0 * rad),

            (
                e["latitude"] - latitude0
            )
            * rad
            * earth_radius,
        )
        for e in events
    ], dtype=np.float64)

    labels = DBSCAN(
        eps=eps_m,
        min_samples=min_samples,
        metric="euclidean",
    ).fit_predict(coordinates)

    # Preserve isolated DBSCAN noise points.
    next_label = int(
        np.max(labels)
    ) + 1

    grouped = defaultdict(list)

    for index, label in enumerate(labels):
        if label < 0:
            label = next_label
            next_label += 1

        grouped[int(label)].append(
            events[index]
        )

    groups = sorted(
        grouped.values(),
        key=lambda group: min(
            event["timestamp"]
            for event in group
        ),
    )

    locations = []

    for location_id, group in enumerate(
        groups,
        start=1,
    ):
        location = {
            "location_id": location_id,

            "latitude": float(
                np.mean([
                    e["latitude"]
                    for e in group
                ])
            ),

            "longitude": float(
                np.mean([
                    e["longitude"]
                    for e in group
                ])
            ),

            "events": group,

            "hazard_classes": sorted({
                cls
                for e in group
                for cls in e["hazards"]
            }),

            "asset_classes": sorted({
                cls
                for e in group
                for cls in e["assets"]
            }),

            "first_timestamp": min(
                e["timestamp"]
                for e in group
            ),

            "last_timestamp": max(
                e["timestamp"]
                for e in group
            ),
        }

        locations.append(location)

    return locations


# ============================================================
# READ SAVED 3D VOXEL PREDICTIONS
# ============================================================

def load_prediction(folder):
    with np.load(
        folder / "prediction.npz",
        allow_pickle=False,
    ) as data:

        required = {
            "semantic",
            "occupancy_binary",
            "origin",
            "voxel_size",
        }

        missing = (
            required - set(data.files)
        )

        if missing:
            raise ValueError(
                f"Missing prediction fields: "
                f"{sorted(missing)}"
            )

        semantic = np.asarray(
            data["semantic"],
            dtype=np.uint8,
        )

        occupancy = np.asarray(
            data["occupancy_binary"],
            dtype=bool,
        )

        origin = np.asarray(
            data["origin"],
            dtype=np.float64,
        ).reshape(3)

        voxel_size = float(
            data["voxel_size"]
        )

    if (
        semantic.ndim != 3
        or occupancy.shape != semantic.shape
        or not (0 < voxel_size < 1)
    ):
        raise ValueError(
            "Invalid voxel grid, occupancy, or voxel size"
        )

    return (
        semantic,
        occupancy,
        origin,
        voxel_size,
    )


def points_for_classes(prediction):
    """
    Convert occupied voxel centres to camera-centric XYZ.

    Original voxel storage: [X, Y, Z].

    XYZ = origin + (voxel_index + 0.5) * voxel_size.
    """

    (
        semantic,
        occupancy,
        origin,
        voxel_size,
    ) = prediction

    result = {}

    for cls in CLASSES:
        indices = np.argwhere(
            occupancy
            & (
                semantic == VOXEL_ID[cls]
            )
        )

        points = (
            origin
            + (
                indices.astype(np.float32)
                + 0.5
            )
            * voxel_size
        )

        result[cls] = points.astype(
            np.float32
        )

    return result


def concatenate(points, classes):
    parts = [
        points[cls]
        for cls in classes
        if len(points[cls])
    ]

    if not parts:
        return np.empty(
            (0, 3),
            dtype=np.float32,
        )

    return np.concatenate(
        parts,
        axis=0,
    )


# ============================================================
# SELECT REPRESENTATIVE FRAMES
# ============================================================

def choose_representative(
    location,
    cls,
    limit,
):
    candidates = sorted(
        [
            e
            for e in location["events"]
            if cls in e["evidence"]
        ],
        key=lambda e: (
            -e["evidence"][cls]["confidence"],
            -e["evidence"][cls]["pixel_count"],
            e["timestamp"],
        ),
    )

    if not candidates:
        return None, None

    # Fallen fence is 2D-only.
    if cls == "fallen_fence":
        return candidates[0], None

    # Prefer high-confidence evidence with actual 3D support.
    for event in candidates[:limit]:
        try:
            prediction = load_prediction(
                event["folder"]
            )

            semantic = prediction[0]
            occupancy = prediction[1]

            has_3d_class = np.any(
                occupancy
                & (
                    semantic == VOXEL_ID[cls]
                )
            )

            if has_3d_class:
                return event, prediction

        except (
            OSError,
            ValueError,
            KeyError,
        ):
            continue

    # Do not invent 3D severity.
    return candidates[0], None


# ============================================================
# IMAGES / 2D SEGMENTATION OVERLAY
# ============================================================

def write_image(path, image):
    success = cv2.imwrite(
        str(path),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 88],
    )

    if not success:
        raise OSError(
            f"Cannot save image: {path}"
        )


def make_overlay(
    image_path,
    prior_path,
    threshold,
):
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise OSError(
            f"Cannot read image: {image_path}"
        )

    height, width = image.shape[:2]

    with np.load(
        prior_path,
        allow_pickle=False,
    ) as data:

        if "full_rfdetr_prior" in data:
            full = data["full_rfdetr_prior"]

            channels = {
                "fallen_fence": full[1],
                "loose_trash": full[2],
                "mud_spill": full[3],
                "public_greenway": full[4],
                "sidewalk": full[5],
            }

        else:
            prior = data["prior"]

            channels = {
                cls: prior[i]
                for i, cls in enumerate(CLASSES)
            }

    output = image.copy()

    for cls in (*ASSETS, *HAZARDS):
        if cls not in channels:
            continue

        plane = channels[cls]
        side = plane.shape[0]

        # Undo the prior's letterbox transformation.
        scale = min(
            side / width,
            side / height,
        )

        new_width = max(
            1,
            round(width * scale),
        )

        new_height = max(
            1,
            round(height * scale),
        )

        pad_x = (
            side - new_width
        ) // 2

        pad_y = (
            side - new_height
        ) // 2

        cropped = plane[
            pad_y:pad_y + new_height,
            pad_x:pad_x + new_width,
        ]

        if cropped.size == 0:
            continue

        mask = (
            cv2.resize(
                cropped.astype(np.float32),
                (width, height),
            )
            >= threshold
        )

        if not np.any(mask):
            continue

        colour = np.asarray(
            BGR[cls],
            dtype=np.float32,
        )

        output[mask] = np.clip(
            output[mask].astype(np.float32)
            * 0.62
            + colour * 0.38,
            0,
            255,
        ).astype(np.uint8)

    cv2.rectangle(
        output,
        (0, 0),
        (width, 45),
        (245, 245, 245),
        -1,
    )

    cv2.putText(
        output,
        "Cached RF-DETR semantic prior",
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(
            0.35,
            min(0.65, width / 1800.0),
        ),
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    return output


# ============================================================
# 3D VOXEL VISUALISATION
# ============================================================

def make_3d_view(
    path,
    points,
    max_points,
):
    try:
        import plotly.graph_objects as go

    except ImportError as exc:
        raise RuntimeError(
            "Install plotly or use --no-3d-html"
        ) from exc

    traces = []

    for cls in (
        "loose_trash",
        "mud_spill",
        "sidewalk",
        "public_greenway",
    ):
        xyz = points[cls]

        if not len(xyz):
            continue

        if len(xyz) > max_points:
            indices = np.linspace(
                0,
                len(xyz) - 1,
                max_points,
                dtype=int,
            )

            xyz = xyz[indices]

        b, g, r = BGR[cls]

        traces.append(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                name=cls,
                marker={
                    "size": (
                        2.1
                        if cls in ASSETS
                        else 3.3
                    ),
                    "color": f"rgb({r},{g},{b})",
                    "opacity": 0.8,
                },
            )
        )

    fig = go.Figure(
        data=traces
    )

    fig.update_layout(
        title=(
            "3DHazardNet - Single Camera Frame"
        ),
        scene={
            "xaxis_title": "X right (m)",
            "yaxis_title": "Y down (m)",
            "zaxis_title": "Z forward (m)",
            "aspectmode": "data",
        },
        margin=dict(
            l=0,
            r=0,
            t=45,
            b=0,
        ),
    )

    fig.write_html(
        str(path),
        include_plotlyjs="cdn",
    )


# ============================================================
# LOCATION SEVERITY PROCESSING
# ============================================================

def process_location(
    location,
    output_root,
    severity,
    args,
):
    location_id = location["location_id"]

    cluster_dir = (
        output_root
        / f"cluster_{location_id:04d}"
    )

    cluster_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save every frame event associated with this location.
    # --------------------------------------------------------

    event_rows = []

    for event in location["events"]:
        event_rows.append({
            "index": event["index"],
            "camera": event["camera"],
            "timestamp": event["timestamp"],
            "latitude": event["latitude"],
            "longitude": event["longitude"],
            "hazards": ";".join(
                event["hazards"]
            ),
            "assets": ";".join(
                event["assets"]
            ),
            "best_hazard_confidence": max(
                event["evidence"][cls]["confidence"]
                for cls in event["hazards"]
            ),
            "source_folder": str(
                event["folder"]
            ),
        })

    write_csv(
        cluster_dir / "events.csv",
        event_rows,
        (
            "index",
            "camera",
            "timestamp",
            "latitude",
            "longitude",
            "hazards",
            "assets",
            "best_hazard_confidence",
            "source_folder",
        ),
    )

    # --------------------------------------------------------
    # Score hazards using 3D voxels.
    # --------------------------------------------------------

    chosen = {}
    scores = {}

    for cls in location["hazard_classes"]:
        representative, prediction = (
            choose_representative(
                location,
                cls,
                args.max_rep_candidates,
            )
        )

        if representative is None:
            continue

        chosen[cls] = {
            "index": representative["index"],
            "camera": representative["camera"],
            "timestamp": representative["timestamp"],
            "confidence": representative[
                "evidence"
            ][cls]["confidence"],
            "predicted_3d": (
                prediction is not None
            ),
        }

        # V10.1 cannot predict fallen fences.
        # Also do not score a 2D-only detection if
        # the corresponding voxel class was absent.
        if (
            cls == "fallen_fence"
            or prediction is None
        ):
            continue

        points_by_class = points_for_classes(
            prediction
        )

        scene_points = concatenate(
            points_by_class,
            CLASSES,
        )

        asset_points = concatenate(
            points_by_class,
            ASSETS,
        )

        confidences = [
            event["evidence"][cls]["confidence"]
            for event in location["events"]
            if cls in event["hazards"]
        ]

        # Exact severity.py API.
        scores[cls] = severity.calculate_hazard_severity(
            class_name=cls,
            hazard_points=points_by_class[cls],
            scene_points=scene_points,
            asset_points=asset_points,
            metric_scale_known=True,
            class_event_count=len(confidences),
            total_event_count=len(
                location["events"]
            ),
            mean_confidence=float(
                np.mean(confidences)
            ),
        )

    # Original severity.py uses the maximum class score,
    # with no mixed-hazard bonus.
    rating = severity.calculate_location_severity(
        scores.values()
    )

    note = (
        "RF-DETR hazard frame event -> GPS DBSCAN -> "
        "saved V10.1 camera-only voxels. "
        "Geometry uses nominal 5 cm voxel centres; "
        "physical accuracy is not externally validated. "
        "Each class is scored using ONE aligned camera "
        "frame, not concatenated unregistered views. "
        "RF-DETR confidences are approximated from "
        "cached per-pixel confidence priors. "
        "Fallen fence is 2D-only and cannot receive "
        "3D voxel severity."
    )

    severity_document = {
        **rating,
        "hazards": scores,
        "source": "cached_v10_1_voxels",
        "scale_mode": "nominal_metric_voxel_grid",
        "metric_scale_known": True,
        "note": note,
    }

    severity.save_severity_metrics(
        cluster_dir / "severity_metrics.json",
        severity_document,
    )

    # --------------------------------------------------------
    # Select a representative image for the map popup.
    # --------------------------------------------------------

    dominant = rating.get(
        "dominant_hazard"
    )

    if dominant not in chosen:
        dominant = max(
            chosen,
            key=lambda name: chosen[name][
                "confidence"
            ],
        )

    representative_info = chosen[dominant]

    representative = next(
        event
        for event in location["events"]
        if (
            event["index"]
            == representative_info["index"]
            and event["camera"]
            == representative_info["camera"]
        )
    )

    image = cv2.imread(
        str(representative["image"])
    )

    if image is None:
        raise OSError(
            f"Cannot read RGB: "
            f"{representative['image']}"
        )

    write_image(
        cluster_dir / "representative_rgb.jpg",
        image,
    )

    try:
        overlay = make_overlay(
            representative["image"],
            representative["folder"]
            / "rfdetr_prior.npz",
            args.threshold,
        )

        write_image(
            cluster_dir / "representative_overlay.jpg",
            overlay,
        )

    except Exception as exc:
        print(
            f"Warning: overlay fallback for "
            f"cluster {location_id}: {exc}"
        )

        write_image(
            cluster_dir / "representative_overlay.jpg",
            image,
        )

    # --------------------------------------------------------
    # Save the representative 3D scene.
    #
    # Do not combine camera-coordinate voxel grids from
    # different frames without extrinsic pose alignment.
    # --------------------------------------------------------

    point_stats = {}

    try:
        prediction = load_prediction(
            representative["folder"]
        )

        points_by_class = points_for_classes(
            prediction
        )

        point_stats = {
            cls: len(points)
            for cls, points
            in points_by_class.items()
        }

        np.savez_compressed(
            cluster_dir / "voxel_points_by_class.npz",
            **points_by_class,
        )

        if not args.no_3d_html:
            make_3d_view(
                cluster_dir / "voxel_3d.html",
                points_by_class,
                args.max_plot_points,
            )

    except (OSError, ValueError, KeyError) as exc:
        print(
            f"Warning: representative voxel "
            f"visualisation unavailable: {exc}"
        )

    # --------------------------------------------------------
    # Save location metadata.
    # --------------------------------------------------------

    metadata = {
        "location_id": location_id,
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "first_timestamp": location["first_timestamp"],
        "last_timestamp": location["last_timestamp"],
        "frame_events": len(
            location["events"]
        ),
        "hazard_classes": location["hazard_classes"],
        "asset_classes_2d": location["asset_classes"],
        "representative_class": dominant,
        "representative_camera": representative["camera"],
        "representative_sample_index": representative["index"],
        "representative_timestamp": representative["timestamp"],
        "class_representatives": chosen,
        "representative_voxel_counts": point_stats,
        "severity_score": rating.get("severity_score"),
        "severity_stars": rating.get("severity_stars"),
        "severity_level": rating.get("severity_level"),
        "severity_dominant_hazard": rating.get(
            "dominant_hazard"
        ),
        "note": note,
    }

    write_json(
        cluster_dir / "metadata.json",
        metadata,
    )

    return metadata


# ============================================================
# BUILD GPS ROUTE
# ============================================================

def save_route(
    gps,
    samples,
    path,
):
    timestamps, latitudes, longitudes = gps

    if not samples:
        return []

    # Route uses ALL eligible frame timestamps,
    # not only frames containing hazards.
    first = min(
        sample["timestamp"]
        for sample in samples
    )

    last = max(
        sample["timestamp"]
        for sample in samples
    )

    selected = np.where(
        (timestamps >= first)
        & (timestamps <= last)
    )[0]

    # Limit map geometry to approximately 5000 GPS fixes.
    stride = max(
        1,
        math.ceil(
            len(selected) / 5000
        ),
    )

    selected = selected[::stride]

    route = [
        (
            float(latitudes[i]),
            float(longitudes[i]),
        )
        for i in selected
    ]

    rows = [
        {
            "timestamp": int(timestamps[i]),
            "latitude": float(latitudes[i]),
            "longitude": float(longitudes[i]),
        }
        for i in selected
    ]

    write_csv(
        path,
        rows,
        (
            "timestamp",
            "latitude",
            "longitude",
        ),
    )

    return route


# ============================================================
# MAP HELPERS
# ============================================================

def small_jpg_data_uri(path):
    image = cv2.imread(
        str(path)
    )

    if image is None:
        return ""

    height, width = image.shape[:2]

    thumbnail_width = min(
        width,
        460,
    )

    thumbnail = cv2.resize(
        image,
        (
            thumbnail_width,
            max(
                1,
                round(
                    height
                    * thumbnail_width
                    / width
                ),
            ),
        ),
    )

    success, encoded = cv2.imencode(
        ".jpg",
        thumbnail,
        [cv2.IMWRITE_JPEG_QUALITY, 74],
    )

    if not success:
        return ""

    return (
        "data:image/jpeg;base64,"
        + base64.b64encode(
            encoded
        ).decode("ascii")
    )


# ============================================================
# BUILD INTERACTIVE SATELLITE HAZARD MAP
# ============================================================

def make_map(
    output_root,
    rows,
    route,
):
    if rows:
        centre = [
            float(np.mean([
                row["latitude"]
                for row in rows
            ])),
            float(np.mean([
                row["longitude"]
                for row in rows
            ])),
        ]

    elif route:
        centre = [
            float(np.mean([
                p[0]
                for p in route
            ])),
            float(np.mean([
                p[1]
                for p in route
            ])),
        ]

    else:
        print(
            "No GPS positions to map"
        )
        return

    fmap = folium.Map(
        location=centre,
        zoom_start=17,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/World_Imagery/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=True,
        show=True,
        max_zoom=19,
    ).add_to(fmap)

    folium.TileLayer(
        "OpenStreetMap",
        name="Street map",
        show=False,
    ).add_to(fmap)

    # Vehicle route.
    if len(route) > 1:
        folium.PolyLine(
            route,
            color="#2399e8",
            weight=4,
            opacity=0.88,
            tooltip="MCAV GPS route",
        ).add_to(fmap)

        folium.CircleMarker(
            route[0],
            radius=6,
            color="#00c853",
            fill=True,
            fill_opacity=1,
            tooltip="Route start",
        ).add_to(fmap)

        folium.CircleMarker(
            route[-1],
            radius=6,
            color="#333333",
            fill=True,
            fill_opacity=1,
            tooltip="Route end",
        ).add_to(fmap)

    # Separate layer for each severity category.
    layers = {}

    for stars in (
        1,
        2,
        3,
        4,
        5,
        None,
    ):
        title = (
            "Unrated (2D-only / no voxel match)"
            if stars is None
            else f"{stars} star hazard"
        )

        layer = folium.FeatureGroup(
            name=title,
            show=True,
        )

        layer.add_to(fmap)

        layers[stars] = layer

    # Cluster pins and information popups.
    for row in rows:
        cluster_id = row["location_id"]

        stars = row["severity_stars"]

        colour = STAR_COLOUR.get(
            stars,
            "#777777",
        )

        folder = (
            output_root
            / f"cluster_{cluster_id:04d}"
        )

        overlay_url = small_jpg_data_uri(
            folder / "representative_overlay.jpg"
        )

        if overlay_url:
            image_html = (
                f'<img src="{overlay_url}" '
                'style="max-width:460px;width:100%;'
                'border-radius:6px">'
            )
        else:
            image_html = ""

        score = row.get(
            "severity_score"
        )

        score_text = (
            "Unrated"
            if score is None
            else f"{float(score):.1f} / 100"
        )

        classes = ", ".join(
            cls.replace("_", " ").title()
            for cls in row["hazard_classes"]
        )

        voxel_viewer = (
            folder / "voxel_3d.html"
        )

        viewer_html = ""

        if voxel_viewer.is_file():
            viewer_html = (
                f'<a href="cluster_{cluster_id:04d}/'
                'voxel_3d.html" target="_blank">'
                'Open 3D voxel viewer</a>'
            )

        popup_html = (
            '<div style="width:460px;'
            'font-family:Arial,sans-serif">'

            f'<h3>Location {cluster_id:04d}</h3>'

            f'<b>Hazards:</b> {classes}<br>'

            f'<b>Frames:</b> '
            f'{row["frame_events"]}<br>'

            f'<b>Severity:</b> '
            f'{score_text}<br>'

            f'<b>Level:</b> '
            f'{row["severity_level"]}<br>'

            f'<b>Representative:</b> '
            f'{row["representative_camera"]} '
            f'frame {row["representative_sample_index"]}'
            '<br>'

            f'{image_html}'
            f'<p>{viewer_html}</p>'

            '<p style="color:#777;font-size:11px">'
            'Severity uses one camera-aligned 5 cm voxel '
            'grid. GPS indicates the vehicle position, '
            'not the surveyed hazard position.'
            '</p>'

            '</div>'
        )

        icon_html = (
            f'<div style="background:{colour};'
            'border:2px solid white;'
            'border-radius:50%;'
            'width:30px;height:30px;'
            'line-height:30px;'
            'font-weight:bold;'
            'text-align:center;color:white;'
            'box-shadow:0 1px 4px #333">'

            f'{stars if stars else "?"}'

            '</div>'
        )

        folium.Marker(
            location=[
                row["latitude"],
                row["longitude"],
            ],
            tooltip=(
                f"Cluster {cluster_id:04d} | "
                f"{score_text} | {classes}"
            ),
            popup=folium.Popup(
                popup_html,
                max_width=510,
            ),
            icon=folium.DivIcon(
                html=icon_html,
                icon_size=(34, 34),
                icon_anchor=(17, 17),
            ),
        ).add_to(
            layers[stars]
        )

    folium.LayerControl(
        collapsed=False
    ).add_to(fmap)

    positions = route + [
        (
            row["latitude"],
            row["longitude"],
        )
        for row in rows
    ]

    if len(positions) > 1:
        fmap.fit_bounds(
            positions,
            padding=(30, 30),
        )

    path = (
        output_root / "hazard_map.html"
    )

    fmap.save(
        str(path)
    )

    print(
        "Interactive hazard map:",
        path,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process saved 3DHazardNet predictions and "
            "RF-DETR priors into a GPS/DBSCAN severity map."
        )
    )

    parser.add_argument(
        "--inference-root",
        type=Path,
        default=find_existing_default(
            "v10_1_sequence_inference"
        ),
    )

    parser.add_argument(
        "--gps-csv",
        type=Path,
        default=find_existing_default(
            "extracted/gps/position.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--severity-script",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "severity.py"
        ),
    )

    parser.add_argument(
        "--camera",
        choices=(
            "all",
            "CAM2",
            "CAM6",
        ),
        default="all",
    )

    parser.add_argument(
        "--start-ts",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--end-ts",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Cached RF-DETR 2D confidence threshold",
    )

    parser.add_argument(
        "--min-hazard-pixels",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--min-asset-pixels",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--gps-max-dt-ms",
        type=float,
        default=1500.0,
    )

    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=7.0,
        help="DBSCAN radius in metres",
    )

    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-rep-candidates",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=30000,
    )

    parser.add_argument(
        "--no-3d-html",
        action="store_true",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Debug with first N cached samples",
    )

    args = parser.parse_args()

    if (
        not 0 <= args.threshold <= 1
        or args.dbscan_eps <= 0
        or args.dbscan_min_samples < 1
        or args.gps_max_dt_ms <= 0
    ):
        parser.error(
            "Invalid confidence, DBSCAN or GPS settings"
        )

    if (
        args.max_rep_candidates < 1
        or args.min_hazard_pixels < 1
        or args.min_asset_pixels < 1
    ):
        parser.error(
            "Pixel and candidate counts must be >= 1"
        )

    if (
        args.start_ts is not None
        and args.end_ts is not None
        and args.start_ts > args.end_ts
    ):
        parser.error(
            "start-ts must be <= end-ts"
        )

    inference_root = (
        args.inference_root
        .expanduser()
        .resolve()
    )

    if not inference_root.is_dir():
        raise FileNotFoundError(
            f"Cached inference root missing:\n"
            f"{inference_root}"
        )

    output_root = (
        args.output
        or (
            inference_root.parent
            / "hazard_clusters_cached_v10"
        )
    ).expanduser().resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("MCAV CACHED HAZARD PIPELINE")
    print("=" * 72)
    print("Cached predictions:", inference_root)
    print("GPS:", args.gps_csv)
    print("Output:", output_root)

    # --------------------------------------------------------
    # Load existing scoring implementation.
    # --------------------------------------------------------

    severity = load_severity(
        args.severity_script
    )

    # --------------------------------------------------------
    # Load GPS.
    # --------------------------------------------------------

    gps = load_gps(
        args.gps_csv.expanduser().resolve()
    )

    # --------------------------------------------------------
    # Discover all saved inference samples.
    # --------------------------------------------------------

    samples, sample_stats = list_saved_samples(
        inference_root,
        args.camera,
        args.start_ts,
        args.end_ts,
    )

    if args.max_frames is not None:
        samples = samples[:args.max_frames]

    print(
        "Sample scan:",
        dict(sample_stats),
    )

    print(
        f"Usable cached samples: {len(samples):,}"
    )

    if not samples:
        raise RuntimeError(
            "No cached samples found. Check inference root, "
            "camera and timestamp filters."
        )

    # --------------------------------------------------------
    # Stage 1: 2D hazard events.
    # --------------------------------------------------------

    events, event_stats = make_events(
        samples,
        gps,
        args,
    )

    print(
        "Hazard scan:",
        dict(event_stats),
    )

    print(
        f"GPS-matched hazard frames: {len(events):,}"
    )

    frame_event_rows = []

    for event in events:
        frame_event_rows.append({
            "sample_index": event["index"],
            "timestamp": event["timestamp"],
            "camera": event["camera"],
            "latitude": event["latitude"],
            "longitude": event["longitude"],
            "gps_dt_ms": event["gps_dt_ms"],
            "hazard_classes": ";".join(
                event["hazards"]
            ),
            "asset_classes": ";".join(
                event["assets"]
            ),
            "conf_loose_trash": event["evidence"]
            .get("loose_trash", {})
            .get("confidence"),
            "conf_mud_spill": event["evidence"]
            .get("mud_spill", {})
            .get("confidence"),
            "conf_fallen_fence": event["evidence"]
            .get("fallen_fence", {})
            .get("confidence"),
            "sample_folder": str(
                event["folder"]
            ),
        })

    write_csv(
        output_root / "frame_events.csv",
        frame_event_rows,
        (
            "sample_index",
            "timestamp",
            "camera",
            "latitude",
            "longitude",
            "gps_dt_ms",
            "hazard_classes",
            "asset_classes",
            "conf_loose_trash",
            "conf_mud_spill",
            "conf_fallen_fence",
            "sample_folder",
        ),
    )

    # --------------------------------------------------------
    # Stage 2: GPS DBSCAN clustering.
    # --------------------------------------------------------

    locations = cluster_events(
        events,
        args.dbscan_eps,
        args.dbscan_min_samples,
    )

    print(
        f"DBSCAN clusters: {len(locations):,}"
    )

    # --------------------------------------------------------
    # Stage 3: 3D voxel severity by cluster.
    # --------------------------------------------------------

    rows = []

    fields = (
        "location_id",
        "latitude",
        "longitude",
        "first_timestamp",
        "last_timestamp",
        "frame_events",
        "hazard_classes",
        "asset_classes_2d",
        "representative_camera",
        "representative_sample_index",
        "severity_score",
        "severity_stars",
        "severity_level",
        "severity_dominant_hazard",
    )

    for n, location in enumerate(
        locations,
        start=1,
    ):
        print(
            f"[{n}/{len(locations)}] "
            f"Cluster {location['location_id']:04d}: "
            f"{len(location['events'])} events | "
            f"{','.join(location['hazard_classes'])}",
            flush=True,
        )

        metadata = process_location(
            location,
            output_root,
            severity,
            args,
        )

        rows.append(metadata)

        # Incrementally preserve the cluster CSV.
        if n % 10 == 0:
            write_csv(
                output_root / "clusters.csv",
                rows,
                fields,
            )

    write_csv(
        output_root / "clusters.csv",
        rows,
        fields,
    )

    # --------------------------------------------------------
    # Stage 4: Full GPS route.
    # --------------------------------------------------------

    route = save_route(
        gps,
        samples,
        output_root / "route.csv",
    )

    # --------------------------------------------------------
    # Stage 5: Interactive map.
    # --------------------------------------------------------

    make_map(
        output_root,
        rows,
        route,
    )

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    summary = {
        "cached_inference_root": str(inference_root),
        "gps_file": str(args.gps_csv),
        "saved_samples_found": len(samples),
        "hazard_frame_events": len(events),
        "dbscan_clusters": len(locations),
        "dbscan_eps_m": args.dbscan_eps,
        "dbscan_min_samples": args.dbscan_min_samples,
        "two_d_threshold": args.threshold,
        "min_hazard_pixels": args.min_hazard_pixels,
        "gps_max_dt_ms": args.gps_max_dt_ms,
        "unrated_clusters": sum(
            row["severity_stars"] is None
            for row in rows
        ),
        "assumptions": (
            "GPS coordinates represent vehicle positions, "
            "not precisely georeferenced hazard positions. "
            "Saved RF-DETR priors are per-pixel composites, "
            "not original instance detections. "
            "Each 3D severity class uses a single "
            "camera-aligned voxel grid."
        ),
    }

    write_json(
        output_root / "pipeline_summary.json",
        summary,
    )

    print("=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)
    print("Saved samples:", len(samples))
    print("Hazard frame events:", len(events))
    print("GPS clusters:", len(locations))
    print(
        "Map:",
        output_root / "hazard_map.html",
    )


if __name__ == "__main__":
    main()
