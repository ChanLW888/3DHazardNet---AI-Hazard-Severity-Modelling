import shutil
from pathlib import Path

import pandas as pd
from matplotlib.path import Path as PolygonPath


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/hazard_test_101"
)

GPS_FILE = (
    DATASET_ROOT
    / "extracted"
    / "gps"
    / "position.csv"
)

NPZ_ROOT = (
    DATASET_ROOT
    / "semantic_voxels_new_roi_5cm"
)

OUTPUT_ROOT = (
    DATASET_ROOT
    / "semantic_voxels_new_roi_5cm_filtered"
)

FILTER_LOG = OUTPUT_ROOT / "filter_log.csv"


# ============================================================
# GPS MATCHING
# ============================================================

# Maximum allowed difference between NPZ timestamp and GPS timestamp.
# 50 ms = 50,000,000 ns
GPS_TOLERANCE_NS = 100_000_000


# ============================================================
# EXCLUSION POLYGON
# ============================================================

# (latitude, longitude)

EXCLUSION_POLYGON = [
    (-37.90712150409077, 145.12898230839627),
    (-37.91447687545938, 145.12758593368648),
    (-37.91605713483157, 145.14020961150348),
    (-37.90661172062127, 145.14191003554470),
]

# PolygonPath uses (x, y)
# Therefore:
# x = longitude
# y = latitude

POLYGON = PolygonPath([
    (lon, lat)
    for lat, lon in EXCLUSION_POLYGON
])


# ============================================================
# LOAD GPS
# ============================================================

print("Loading GPS:")
print(GPS_FILE)

gps = pd.read_csv(GPS_FILE)

print(f"GPS rows: {len(gps):,}")

required_columns = {
    "ros_timestamp_ns",
    "latitude_deg",
    "longitude_deg",
}

missing = required_columns - set(gps.columns)

if missing:
    raise RuntimeError(
        f"Missing GPS columns: {missing}"
    )


# Keep only what we need
gps = gps[
    [
        "ros_timestamp_ns",
        "latitude_deg",
        "longitude_deg",
    ]
].copy()

gps["ros_timestamp_ns"] = (
    gps["ros_timestamp_ns"]
    .astype("int64")
)

# Sort by timestamp for nearest matching
gps = gps.sort_values("ros_timestamp_ns").reset_index(drop=True)


# ============================================================
# FIND CAMERA FOLDERS
# ============================================================

camera_dirs = [
    NPZ_ROOT / "CAM2",
    NPZ_ROOT / "CAM6",
]

for camera_dir in camera_dirs:

    if not camera_dir.exists():
        print(
            f"[WARNING] Camera folder does not exist: "
            f"{camera_dir}"
        )


# ============================================================
# CREATE OUTPUT ROOT
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FILTER
# ============================================================

kept = 0
discarded = 0
missing_gps = 0
invalid_timestamp = 0

log_rows = []


for camera_dir in camera_dirs:

    if not camera_dir.exists():
        continue

    camera_name = camera_dir.name

    print()
    print("=" * 60)
    print(f"PROCESSING {camera_name}")
    print("=" * 60)

    # --------------------------------------------------------
    # Output camera directory
    # --------------------------------------------------------

    output_camera_dir = (
        OUTPUT_ROOT / camera_name
    )

    output_camera_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Find NPZ files
    # --------------------------------------------------------

    npz_files = sorted(
        camera_dir.glob("*.npz")
    )

    print(
        f"NPZ files found: {len(npz_files):,}"
    )

    # --------------------------------------------------------
    # Process every NPZ
    # --------------------------------------------------------

    for npz_path in npz_files:

        # ====================================================
        # Extract timestamp from filename
        # ====================================================

        try:
            timestamp = int(npz_path.stem)

        except ValueError:

            print(
                f"[WARNING] Invalid timestamp: "
                f"{npz_path.name}"
            )

            invalid_timestamp += 1

            log_rows.append({
                "camera": camera_name,
                "npz": npz_path.name,
                "timestamp_ns": "",
                "gps_timestamp_ns": "",
                "gps_delta_ms": "",
                "latitude": "",
                "longitude": "",
                "inside_exclusion": "",
                "action": "INVALID_TIMESTAMP",
            })

            continue


        # ====================================================
        # Find nearest GPS timestamp
        # ====================================================

        # Use pandas nearest merge on a single timestamp
        timestamp_df = pd.DataFrame({
            "ros_timestamp_ns": [timestamp]
        })

        match = pd.merge_asof(
            timestamp_df,
            gps,
            on="ros_timestamp_ns",
            direction="nearest",
            tolerance=GPS_TOLERANCE_NS,
        )

        gps_row = match.iloc[0]


        # ====================================================
        # Check whether GPS was found
        # ====================================================

        if pd.isna(gps_row["latitude_deg"]):

            print(
                f"[WARNING] No GPS within "
                f"{GPS_TOLERANCE_NS / 1e6:.1f} ms: "
                f"{npz_path.name}"
            )

            missing_gps += 1

            log_rows.append({
                "camera": camera_name,
                "npz": npz_path.name,
                "timestamp_ns": timestamp,
                "gps_timestamp_ns": "",
                "gps_delta_ms": "",
                "latitude": "",
                "longitude": "",
                "inside_exclusion": "",
                "action": "NO_GPS_MATCH",
            })

            continue


        # ====================================================
        # GPS information
        # ====================================================

        lat = float(
            gps_row["latitude_deg"]
        )

        lon = float(
            gps_row["longitude_deg"]
        )

        gps_timestamp = int(
            gps_row["ros_timestamp_ns"]
        )

        delta_ms = (
            abs(timestamp - gps_timestamp)
            / 1e6
        )


        # ====================================================
        # Check exclusion polygon
        # ====================================================

        inside = POLYGON.contains_point(
            (lon, lat)
        )


        # ====================================================
        # KEEP / DISCARD
        # ====================================================

        if inside:

            discarded += 1

            action = "DISCARD"

            print(
                f"[DISCARD] {camera_name}/"
                f"{npz_path.name} "
                f"lat={lat:.8f}, "
                f"lon={lon:.8f}"
            )

        else:

            kept += 1

            action = "KEEP"

            shutil.copy2(
                npz_path,
                output_camera_dir / npz_path.name
            )


        # ====================================================
        # Log
        # ====================================================

        log_rows.append({
            "camera": camera_name,
            "npz": npz_path.name,
            "timestamp_ns": timestamp,
            "gps_timestamp_ns": gps_timestamp,
            "gps_delta_ms": delta_ms,
            "latitude": lat,
            "longitude": lon,
            "inside_exclusion": inside,
            "action": action,
        })


# ============================================================
# SAVE FILTER LOG
# ============================================================

log_df = pd.DataFrame(log_rows)

log_df.to_csv(
    FILTER_LOG,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("FILTER COMPLETE")
print("=" * 60)

print(
    f"Original NPZs:       "
    f"{len(log_df):,}"
)

print(
    f"Kept:                "
    f"{kept:,}"
)

print(
    f"Discarded:           "
    f"{discarded:,}"
)

print(
    f"Missing GPS:         "
    f"{missing_gps:,}"
)

print(
    f"Invalid timestamps:  "
    f"{invalid_timestamp:,}"
)

print()
print("Filtered dataset:")
print(OUTPUT_ROOT)

print()
print("Filter log:")
print(FILTER_LOG)