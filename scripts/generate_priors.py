#!/usr/bin/env python3

"""Generate cached RF-DETR + LiDAR priors for V5, with projection checks.

The generated priors are saved at PRIOR_SIZE x PRIOR_SIZE.

For verification, this script ALSO saves raw-resolution LiDAR overlays
BEFORE dilation, Gaussian blur, or resizing. This lets you visually
confirm that the LiDAR projection used to generate the V5 prior is correct.

Projection:
    p_camera = T_lidar_to_camera @ p_lidar
    u = fx * X/Z + cx
    v = fy * Y/Z + cy
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from rfdetr import RFDETRSegSmall


# ============================================================
# CONFIG
# ============================================================

BAG_ROOT = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/hazard_test_101"
)

EXTRACTED = BAG_ROOT / "extracted"
NPZ_ROOT = BAG_ROOT / "semantic_voxels_filtered"
OUTPUT_ROOT = BAG_ROOT / "v5_priors_letterboxed"

# Separate visual verification output.
CHECK_ROOT = BAG_ROOT / "v5_priors_letterboxed_projection_check"

CAMERAS = ["CAM2", "CAM6"]

SYNC_TOLERANCE_NS = 50_000_000
LIDAR_MAX_RANGE = 50.0
LIDAR_MIN_CAMERA_DEPTH = 0.05
BORDER_MARGIN_PIXELS = 2

# V5 prior resolution.
PRIOR_SIZE = 192

RFDETR_CONFIDENCE_THRESHOLD = 0.30

RFDETR_MODEL_PATH = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/"
    "2D-Seg-Road-Nissan.v13i.coco-segmentation/"
    "outputs/checkpoint_best_total.pth"
)

LIDAR_DILATION_KERNEL = 5
LIDAR_BLUR_KERNEL = 11
LIDAR_DEPTH_MAX = 30.0

# Save a raw LiDAR projection check every N generated samples.
# Set to 0 to disable. 1 = every sample.
PROJECTION_CHECK_EVERY = 100

# Also save the first few frames from each camera, regardless of the
# interval above. This makes it easy to inspect immediately.
PROJECTION_CHECK_FIRST_N_PER_CAMERA = 3

PROJECTION_POINT_RADIUS = 2
PROJECTION_DEPTH_VIS_MAX = 30.0


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
# CALIBRATION / TIMESTAMPS
# ============================================================

def parse_array(v):
    v = str(v).strip()

    for w in [
        "np.float64(",
        "np.float32(",
        "np.int64(",
        "np.int32(",
    ]:
        v = v.replace(w, "")

    v = (
        v.replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace(",", " ")
    )

    out = np.fromstring(
        v,
        sep=" ",
        dtype=np.float64,
    )

    if out.size == 0:
        raise RuntimeError(
            f"Could not parse array: {v}"
        )

    return out


def load_intrinsics(camera):
    path = EXTRACTED / camera / "camera_info.csv"

    with open(path) as f:
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

    return (
        K,
        int(row["width"]),
        int(row["height"]),
        P.reshape(3, 4),
    )


def load_images(camera):
    out = []

    for p in (
        EXTRACTED / camera / "images_rect"
    ).glob("*.png"):

        try:
            out.append((int(p.stem), p))
        except ValueError:
            pass

    if not out:
        raise RuntimeError(
            f"No images found for {camera}"
        )

    return sorted(out)


def load_lidar_files():
    out = []

    for p in (
        EXTRACTED / "lidar"
    ).glob("*.ply"):

        try:
            out.append((int(p.stem), p))
        except ValueError:
            pass

    if not out:
        raise RuntimeError(
            "No LiDAR PLY files found"
        )

    return sorted(out)


def nearest_timestamp(target, records):
    ts = np.asarray(
        [x[0] for x in records],
        dtype=np.int64,
    )

    i = int(np.searchsorted(ts, target))
    candidates = []

    if i > 0:
        candidates.append(i - 1)

    if i < len(records):
        candidates.append(i)

    if not candidates:
        return None, None, None

    j = min(
        candidates,
        key=lambda k: abs(
            int(records[k][0]) - int(target)
        ),
    )

    return (
        int(records[j][0]),
        records[j][1],
        abs(
            int(records[j][0]) - int(target)
        ),
    )


# ============================================================
# LiDAR
# ============================================================

def load_ascii_ply(path):
    header = []

    with open(path) as f:
        while True:
            line = f.readline()

            if not line:
                raise RuntimeError(
                    f"EOF: {path}"
                )

            line = line.strip()
            header.append(line)

            if line == "end_header":
                break

    vc = None

    for line in header:
        p = line.split()

        if (
            len(p) == 3
            and p[0] == "element"
            and p[1] == "vertex"
        ):
            vc = int(p[2])
            break

    if vc is None:
        raise RuntimeError(
            f"No vertex element: {path}"
        )

    props = []
    inv = False

    for line in header:
        p = line.split()

        if (
            len(p) >= 3
            and p[0] == "element"
            and p[1] == "vertex"
        ):
            inv = True
            continue

        if (
            inv
            and len(p) >= 3
            and p[0] == "property"
        ):
            props.append(p[-1])

        elif (
            inv
            and len(p) >= 2
            and p[0] == "element"
        ):
            break

    data = np.loadtxt(
        path,
        skiprows=len(header),
        max_rows=vc,
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data, props


def load_lidar(path):
    data, props = load_ascii_ply(path)

    xyz = data[
        :,
        [
            props.index("x"),
            props.index("y"),
            props.index("z"),
        ],
    ].astype(np.float64)

    r = np.linalg.norm(xyz, axis=1)

    return xyz[r <= LIDAR_MAX_RANGE]


# ============================================================
# LiDAR projection
# ============================================================

def project_lidar(
    points,
    camera,
    K,
    w,
    h,
):
    hp = np.concatenate(
        [
            points,
            np.ones(
                (len(points), 1)
            ),
        ],
        axis=1,
    )

    cp = (
        hp
        @ LIDAR_TO_CAMERA[camera].T
    )[:, :3]

    X = cp[:, 0]
    Y = cp[:, 1]
    Z = cp[:, 2]

    valid = (
        np.isfinite(cp).all(axis=1)
        & (Z > LIDAR_MIN_CAMERA_DEPTH)
    )

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    if len(Z) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            0,
        )

    u = np.rint(
        K[0, 0] * X / Z
        + K[0, 2]
    ).astype(np.int64)

    v = np.rint(
        K[1, 1] * Y / Z
        + K[1, 2]
    ).astype(np.int64)

    inside = (
        (u >= BORDER_MARGIN_PIXELS)
        & (u < w - BORDER_MARGIN_PIXELS)
        & (v >= BORDER_MARGIN_PIXELS)
        & (v < h - BORDER_MARGIN_PIXELS)
    )

    return (
        u[inside],
        v[inside],
        Z[inside],
        int(valid.sum()),
    )


# ============================================================
# Projection verification overlay
# ============================================================

def save_projection_check(
    image_rgb,
    u,
    v,
    depth,
    path,
):
    """
    Save raw-resolution RGB + projected LiDAR.

    IMPORTANT:
    This is generated directly from the exact u/v/depth arrays
    used by lidar_prior(), BEFORE dilation, blur, or resize.
    """

    image = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    if len(u):
        depth_norm = np.clip(
            depth / PROJECTION_DEPTH_VIS_MAX,
            0.0,
            1.0,
        )

        values = (
            depth_norm * 255
        ).astype(np.uint8)

        colours = cv2.applyColorMap(
            values.reshape(-1, 1),
            cv2.COLORMAP_TURBO,
        ).reshape(-1, 3)

        # Draw far points first so near points remain visible.
        order = np.argsort(depth)[::-1]

        for i in order:
            cv2.circle(
                image,
                (int(u[i]), int(v[i])),
                PROJECTION_POINT_RADIUS,
                tuple(
                    int(x)
                    for x in colours[i]
                ),
                -1,
                lineType=cv2.LINE_AA,
            )

    cv2.imwrite(
        str(path),
        image,
    )


# ============================================================
# LiDAR prior
# ============================================================

def lidar_prior(
    points,
    camera,
    K,
    w,
    h,
):
    u, v, d, _ = project_lidar(
        points,
        camera,
        K,
        w,
        h,
    )

    depth = np.zeros(
        (h, w),
        np.float32,
    )

    support = np.zeros(
        (h, w),
        np.float32,
    )

    if len(u) == 0:
        return (
            depth,
            support,
            u,
            v,
            d,
        )

    ids = (
        v.astype(np.int64) * w
        + u.astype(np.int64)
    )

    order = np.lexsort(
        (d, ids)
    )

    sp = ids[order]

    keep = np.ones(
        len(order),
        bool,
    )

    if len(order) > 1:
        keep[1:] = (
            sp[1:] != sp[:-1]
        )

    sel = order[keep]

    depth[
        v[sel],
        u[sel]
    ] = np.minimum(
        d[sel],
        LIDAR_DEPTH_MAX,
    ).astype(np.float32)

    support[
        v[sel],
        u[sel]
    ] = 1.0

    kernel = np.ones(
        (
            LIDAR_DILATION_KERNEL,
            LIDAR_DILATION_KERNEL,
        ),
        np.uint8,
    )

    support_d = cv2.dilate(
        support,
        kernel,
    )

    depth_b = cv2.GaussianBlur(
        depth,
        (
            LIDAR_BLUR_KERNEL,
            LIDAR_BLUR_KERNEL,
        ),
        0,
    )

    depth_n = np.zeros_like(
        depth_b
    )

    m = support_d > 0

    depth_n[m] = (
        depth_b[m]
        / LIDAR_DEPTH_MAX
    ).clip(0, 1)

    support_s = cv2.GaussianBlur(
        support_d,
        (
            LIDAR_BLUR_KERNEL,
            LIDAR_BLUR_KERNEL,
        ),
        0,
    ).clip(0, 1)

    return (
        depth_n,
        support_s,
        u,
        v,
        d,
    )


# ============================================================
# RF-DETR
# ============================================================

def norm_mask(mask, h, w):
    if torch.is_tensor(mask):
        mask = (
            mask.detach()
            .cpu()
            .numpy()
        )

    mask = np.squeeze(
        np.asarray(mask)
    )

    if mask.dtype != np.bool_:
        mask = mask > 0.5

    if mask.shape != (h, w):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    return mask


def rf_prior(det, h, w):
    out = np.zeros(
        (6, h, w),
        np.float32,
    )

    ids = det.class_id
    conf = det.confidence
    masks = det.mask

    assigned = np.zeros(
        (h, w),
        bool,
    )

    order = sorted(
        range(len(ids)),
        key=lambda i: float(conf[i]),
        reverse=True,
    )

    for i in order:

        c = float(conf[i])
        cid = int(ids[i])

        if (
            c < RFDETR_CONFIDENCE_THRESHOLD
            or cid not in range(5)
        ):
            continue

        m = norm_mask(
            masks[i],
            h,
            w,
        )

        a = m & (~assigned)

        out[cid + 1, a] = c
        assigned[a] = True

    out[0] = (
        1.0
        - out[1:].max(axis=0)
    )

    return out


# ============================================================
# Dataset samples
# ============================================================

def find_samples():
    out = []

    for cam in CAMERAS:

        for p in (
            NPZ_ROOT / cam
        ).glob("*.npz"):

            try:
                ts = int(p.stem)
            except ValueError:
                continue

            with np.load(
                p,
                allow_pickle=True,
            ) as d:

                ip = d["image_path"]

                if isinstance(
                    ip,
                    np.ndarray,
                ):
                    ip = ip.item()

            out.append(
                (
                    cam,
                    ts,
                    Path(str(ip)),
                )
            )

    return sorted(
        out,
        key=lambda x: (
            x[0],
            x[1],
        ),
    )


# ============================================================
# Letterbox prior to match V5 RGB preprocessing
# ============================================================

def letterbox_prior(prior, target_size):
    """
    Apply the same aspect-ratio-preserving resize + padding
    used by the V5 RGB training pipeline.

    prior:
        [C, H, W] in original camera-image coordinates.

    Returns:
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
    ).unsqueeze(0)  # [1,C,H,W]

    prior_tensor = torch.nn.functional.interpolate(
        prior_tensor,
        size=(nh, nw),
        mode="bilinear",
        align_corners=False,
    )

    pad_x = (target_size - nw) // 2
    pad_y = (target_size - nh) // 2

    # Zero padding means "no prior information" in the padded
    # region. This matches the semantic meaning of the prior.
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


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECK_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    cal = {}
    cams = {}

    for cam in CAMERAS:

        K, w, h, P = load_intrinsics(
            cam
        )

        cal[cam] = (
            K,
            w,
            h,
            P,
        )

        cams[cam] = load_images(
            cam
        )

        print(
            f"{cam}: "
            f"{w}x{h}, "
            f"{len(cams[cam]):,} images"
        )

        print("K:")
        print(K)
        print("P:")
        print(P)

    lidars = load_lidar_files()
    samples = find_samples()

    print(
        "Samples:",
        len(samples),
    )

    print(
        "Loading RF-DETR..."
    )

    model = (
        RFDETRSegSmall.from_checkpoint(
            str(RFDETR_MODEL_PATH)
        )
    )

    # We call model.predict([bgr]) with exactly one image at a time,
    # so RF-DETR must be optimized/compiled for batch size 1.
    model.inference(
        dtype=torch.float16,
        batch_size=1,
    )

    generated = 0
    skipped = 0

    projection_checks = {
        cam: 0
        for cam in CAMERAS
    }

    for n, (
        cam,
        ts,
        image_path,
    ) in enumerate(
        samples,
        1,
    ):

        out = (
            OUTPUT_ROOT
            / cam
            / f"{ts}.npz"
        )

        out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if out.exists():
            skipped += 1
            continue

        image = np.asarray(
            Image.open(
                image_path
            ).convert("RGB")
        )

        h, w = image.shape[:2]

        bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )

        with torch.inference_mode():
            det = model.predict(
                [bgr]
            )[0]

        rp = rf_prior(
            det,
            h,
            w,
        )

        _, lp, delta = nearest_timestamp(
            ts,
            lidars,
        )

        if lp is None:
            print(
                f"[SKIP] {cam} {ts}: "
                "no LiDAR file available"
            )
            skipped += 1
            continue

        # Only use LiDAR when the nearest scan is sufficiently
        # close in time to the RGB/NPZ frame.
        #
        # This prevents temporally misaligned LiDAR from becoming
        # a misleading geometric prior for V5.
        if delta > SYNC_TOLERANCE_NS:
            print(
                f"[SKIP] {cam} {ts}: "
                f"nearest LiDAR is "
                f"{delta / 1e6:.2f} ms away "
                f"(limit={SYNC_TOLERANCE_NS / 1e6:.1f} ms)"
            )
            skipped += 1
            continue

        pts = load_lidar(lp)

        K, _, _, _ = cal[cam]

        (
            dp,
            sp,
            u,
            v,
            d,
        ) = lidar_prior(
            pts,
            cam,
            K,
            w,
            h,
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Save the check from the SAME projection arrays
        # used to create the actual LiDAR prior.
        #
        # It is therefore a direct test of this generator's
        # LiDAR projection, not a separate implementation.
        # ----------------------------------------------------

        should_check = (
            projection_checks[cam]
            < PROJECTION_CHECK_FIRST_N_PER_CAMERA
            or (
                PROJECTION_CHECK_EVERY > 0
                and generated % PROJECTION_CHECK_EVERY == 0
            )
        )

        if should_check:

            check_dir = (
                CHECK_ROOT / cam
            )

            check_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            check_path = (
                check_dir
                / f"{ts}_overlay.png"
            )

            save_projection_check(
                image,
                u,
                v,
                d,
                check_path,
            )

            projection_checks[cam] += 1

            print(
                f"[CHECK] {cam} {ts} -> "
                f"{check_path}"
            )

        prior = np.concatenate(
            [
                rp,
                dp[None],
                sp[None],
            ],
            axis=0,
        ).astype(np.float16)

        # This is the actual prior used by V5.
        # prior has 8 channels:
        #   0..5 = RF-DETR semantic priors
        #   6    = LiDAR depth prior
        #   7    = LiDAR support prior
        #
        # IMPORTANT:
        # Match the RGB training pipeline's aspect-ratio-preserving
        # letterbox transform instead of directly squashing the
        # original camera image into a square.
        prior = letterbox_prior(
            prior,
            PRIOR_SIZE,
        ).astype(np.float16)

        np.savez_compressed(
            out,
            prior=prior,
            camera=np.array(cam),
            timestamp=np.int64(ts),
            image_path=np.array(
                str(image_path)
            ),
        )

        generated += 1

        if generated % 100 == 0:
            print(
                f"[PRIOR] "
                f"generated={generated:,} "
                f"skipped={skipped:,}"
            )

    print()
    print("Done.")
    print(
        "generated=",
        generated,
        "skipped=",
        skipped,
    )
    print(
        f"LiDAR synchronization threshold: "
        f"{SYNC_TOLERANCE_NS / 1e6:.1f} ms"
    )
    print(
        "priors=",
        OUTPUT_ROOT,
    )
    print(
        "projection checks=",
        CHECK_ROOT,
    )


if __name__ == "__main__":
    main()
