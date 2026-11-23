#!/usr/bin/env python3
"""
Severity model for MCAV hazard mapping.

The model supports two operating modes:

1) metric_lidar
   Used when the 3D reconstruction is metric-scale assisted (for example by
   synchronized LiDAR depth supplied to MapAnything). Geometry terms use
   metres / square metres / cubic metres as approximate measurements.

2) relative_reconstruction
   Used when the reconstruction scale has not been validated. Geometry is
   normalized by the local scene diagonal, so the score remains useful for
   relative triage without pretending reconstruction units are metres.

The location score is simply the maximum class score. There is deliberately
NO hard-coded mixed-hazard bonus or minimum star rating.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors


SUPPORTED_CLASSES = {
    "mud_spill",
    "loose_trash",
    "fallen_fence",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _as_points(points) -> np.ndarray:
    if points is None:
        return np.empty((0, 3), dtype=np.float32)

    arr = np.asarray(points, dtype=np.float32)

    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    arr = arr.reshape(-1, 3)
    arr = arr[np.isfinite(arr).all(axis=1)]
    return arr


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _saturating(value: float, reference: float) -> float:
    """phi(x) = 1 - exp(-x/x0), bounded in [0,1]."""
    value = max(0.0, float(value))
    reference = max(1e-9, float(reference))
    return _clamp01(1.0 - math.exp(-value / reference))


def _confidence_transform(confidence: float) -> float:
    """Smoothly map detector confidence to [0,1]."""
    confidence = _clamp01(confidence)
    return _clamp01(1.0 / (1.0 + math.exp(-8.0 * (confidence - 0.5))))


def _principal_extents(points: np.ndarray) -> np.ndarray:
    """Return PCA-aligned extents e1 >= e2 >= e3."""
    points = _as_points(points)

    if len(points) < 3:
        return np.zeros(3, dtype=np.float64)

    centred = points - np.mean(points, axis=0, keepdims=True)

    try:
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
        rotated = centred @ vh.T

        # Robust span: use the 5th-95th percentile rather than absolute
        # min/max so a few misregistered/outlier points from repeated views
        # cannot make a tiny hazard look physically huge.
        low = np.percentile(rotated, 5.0, axis=0)
        high = np.percentile(rotated, 95.0, axis=0)
        extents = high - low
    except np.linalg.LinAlgError:
        extents = np.max(points, axis=0) - np.min(points, axis=0)

    extents = np.sort(np.asarray(extents, dtype=np.float64))[::-1]

    if len(extents) < 3:
        extents = np.pad(extents, (0, 3 - len(extents)))

    return extents[:3]


def _scene_diagonal(scene_points) -> float:
    scene = _as_points(scene_points)

    if len(scene) == 0:
        return 1.0

    extent = np.max(scene, axis=0) - np.min(scene, axis=0)
    diagonal = float(np.linalg.norm(extent))
    return max(diagonal, 1e-6)


def calculate_geometry_metrics(
    hazard_points,
    scene_points=None,
    *,
    metric_scale_known: bool = False,
) -> Dict[str, Any]:
    hazard = _as_points(hazard_points)
    scene = _as_points(scene_points)
    extents = _principal_extents(hazard)
    e1, e2, e3 = [float(v) for v in extents]
    scene_diag = _scene_diagonal(scene)

    footprint = e1 * e2
    bbox_volume = e1 * e2 * e3

    if len(hazard) > 0:
        centroid = np.mean(hazard, axis=0)
        minimum = np.min(hazard, axis=0)
        maximum = np.max(hazard, axis=0)
    else:
        centroid = minimum = maximum = None

    result = {
        "point_count": int(len(hazard)),
        "scene_point_count": int(len(scene)),
        "metric_scale_validated": bool(metric_scale_known),
        "scale_mode": "metric_lidar" if metric_scale_known else "relative_reconstruction",
        "units": "metres" if metric_scale_known else "reconstruction_units",
        "centroid": None if centroid is None else [float(v) for v in centroid],
        "bbox_min": None if minimum is None else [float(v) for v in minimum],
        "bbox_max": None if maximum is None else [float(v) for v in maximum],
        "principal_extent_1": e1,
        "principal_extent_2": e2,
        "principal_extent_3": e3,
        "footprint_proxy": float(footprint),
        "bbox_volume_proxy": float(bbox_volume),
        "extent_method": "PCA 5th-95th percentile span",
        "scene_diagonal": float(scene_diag),
        "relative_length": float(e1 / scene_diag),
        "relative_width": float(e2 / scene_diag),
        "relative_height": float(e3 / scene_diag),
        "relative_footprint": float(footprint / (scene_diag ** 2)),
        "relative_volume": float(bbox_volume / (scene_diag ** 3)),
    }

    if metric_scale_known:
        result.update(
            {
                "length_m": e1,
                "width_m": e2,
                "height_m": e3,
                "footprint_m2": footprint,
                "bbox_volume_m3": bbox_volume,
            }
        )

    return result


def estimate_asset_interaction(
    hazard_points,
    asset_points,
    scene_points=None,
    *,
    metric_scale_known: bool = False,
    max_sample: int = 8000,
) -> Dict[str, Any]:
    """
    Estimate how strongly hazard points interact with sidewalk/greenway points.

    This is intentionally called an interaction/obstruction proxy, not a true
    blocked-width percentage. The segmentation classes do not overlap by
    design, so nearest-neighbour proximity is used instead.
    """
    hazard = _as_points(hazard_points)
    assets = _as_points(asset_points)

    if len(hazard) == 0 or len(assets) == 0:
        return {
            "interaction_ratio": 0.0,
            "distance_threshold": None,
            "sampled_hazard_points": int(min(len(hazard), max_sample)),
            "asset_points": int(len(assets)),
        }

    if len(hazard) > max_sample:
        idx = np.linspace(0, len(hazard) - 1, max_sample, dtype=int)
        hazard_sample = hazard[idx]
    else:
        hazard_sample = hazard

    if len(assets) > max_sample * 3:
        idx = np.linspace(0, len(assets) - 1, max_sample * 3, dtype=int)
        asset_sample = assets[idx]
    else:
        asset_sample = assets

    if metric_scale_known:
        threshold = 0.35  # metres; initial engineering configuration value
    else:
        threshold = 0.035 * _scene_diagonal(scene_points)

    nn = NearestNeighbors(
        n_neighbors=1,
        algorithm="auto",
    )
    nn.fit(asset_sample)
    distances, _ = nn.kneighbors(hazard_sample)
    distances = distances[:, 0]

    ratio = float(np.mean(distances <= threshold))

    return {
        "interaction_ratio": _clamp01(ratio),
        "distance_threshold": float(threshold),
        "sampled_hazard_points": int(len(hazard_sample)),
        "asset_points": int(len(asset_sample)),
        "median_nearest_asset_distance": float(np.median(distances)),
    }


# ---------------------------------------------------------------------
# Geometry score
# ---------------------------------------------------------------------

def _geometry_component(
    class_name: str,
    geometry: Dict[str, Any],
    *,
    metric_scale_known: bool,
) -> Dict[str, float]:
    """Hazard-specific nonlinear geometric score G_h."""

    if metric_scale_known:
        length = float(geometry.get("length_m", 0.0))
        width = float(geometry.get("width_m", 0.0))
        height = float(geometry.get("height_m", 0.0))
        area = float(geometry.get("footprint_m2", 0.0))
        volume = float(geometry.get("bbox_volume_m3", 0.0))

        if class_name == "loose_trash":
            # Conservative physical references so a small can/bag stays low.
            # Footprint dominates because ground coverage is the most useful
            # size cue for council clean-up severity.
            terms = {
                "volume_term": _saturating(volume, 2.0),
                "area_term": _saturating(area, 5.0),
                "height_term": _saturating(height, 1.0),
            }
            score = (
                0.30 * terms["volume_term"]
                + 0.50 * terms["area_term"]
                + 0.20 * terms["height_term"]
            )

        elif class_name == "mud_spill":
            terms = {
                "area_term": _saturating(area, 6.0),
                "length_term": _saturating(length, 4.0),
                "height_term": _saturating(height, 0.25),
            }
            score = (
                0.55 * terms["area_term"]
                + 0.25 * terms["length_term"]
                + 0.20 * terms["height_term"]
            )

        else:  # fallen_fence
            terms = {
                "length_term": _saturating(length, 4.0),
                "height_term": _saturating(height, 1.2),
                "area_term": _saturating(area, 5.0),
            }
            score = (
                0.50 * terms["length_term"]
                + 0.25 * terms["height_term"]
                + 0.25 * terms["area_term"]
            )

    else:
        length = float(geometry.get("relative_length", 0.0))
        height = float(geometry.get("relative_height", 0.0))
        area = float(geometry.get("relative_footprint", 0.0))
        volume = float(geometry.get("relative_volume", 0.0))

        # Reference ratios are dataset-scale starting values. They make the
        # score dimensionless and allow useful relative triage before metric
        # scale is validated.
        if class_name == "loose_trash":
            terms = {
                "volume_term": _saturating(volume, 0.00035),
                "area_term": _saturating(area, 0.0100),
                "height_term": _saturating(height, 0.035),
            }
            score = (
                0.30 * terms["volume_term"]
                + 0.50 * terms["area_term"]
                + 0.20 * terms["height_term"]
            )

        elif class_name == "mud_spill":
            terms = {
                "area_term": _saturating(area, 0.0100),
                "length_term": _saturating(length, 0.08),
                "height_term": _saturating(height, 0.010),
            }
            score = (
                0.55 * terms["area_term"]
                + 0.25 * terms["length_term"]
                + 0.20 * terms["height_term"]
            )

        else:  # fallen_fence
            terms = {
                "length_term": _saturating(length, 0.10),
                "height_term": _saturating(height, 0.035),
                "area_term": _saturating(area, 0.0080),
            }
            score = (
                0.50 * terms["length_term"]
                + 0.25 * terms["height_term"]
                + 0.25 * terms["area_term"]
            )

    terms["geometry_score"] = _clamp01(score)
    return terms


def score_to_stars(score: Optional[float]) -> Optional[int]:
    """Map a 0-100 score to 1-5 stars."""
    if score is None:
        return None

    s = max(0.0, min(100.0, float(score)))

    if s < 20.0:
        return 1
    if s < 40.0:
        return 2
    if s < 60.0:
        return 3
    if s < 80.0:
        return 4
    return 5


def score_to_level(score: Optional[float]) -> str:
    stars = score_to_stars(score)

    return {
        None: "unrated",
        1: "very_low",
        2: "low",
        3: "moderate",
        4: "high",
        5: "critical",
    }[stars]


# ---------------------------------------------------------------------
# Public severity API
# ---------------------------------------------------------------------

def calculate_hazard_severity(
    class_name: str,
    hazard_points,
    scene_points=None,
    *,
    asset_points=None,
    metric_scale_known: bool = False,
    class_event_count: int = 1,
    total_event_count: int = 1,
    mean_confidence: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute one class-specific severity score.

    S_h = 100 * [0.85 G_h + 0.10 O_h + 0.02 P_h + 0.02 C_h + 0.01 Q_h]

    G_h : hazard-specific nonlinear geometry term
    O_h : asset interaction / obstruction proxy
    P_h : persistence across location frame events (sqrt transformed)
    C_h : RF-DETR confidence (logistic transformed)
    Q_h : 3D support / point evidence term
    """
    if class_name not in SUPPORTED_CLASSES:
        raise ValueError(
            f"Unsupported class '{class_name}'. Expected one of {sorted(SUPPORTED_CLASSES)}"
        )

    geometry = calculate_geometry_metrics(
        hazard_points,
        scene_points,
        metric_scale_known=metric_scale_known,
    )

    if geometry["point_count"] == 0:
        return {
            "class_name": class_name,
            "severity_score": None,
            "severity_stars": None,
            "severity_level": "unrated",
            "scoring_ready": False,
            "metric_scale_known": bool(metric_scale_known),
            "scale_mode": "metric_lidar" if metric_scale_known else "relative_reconstruction",
            "geometry": geometry,
            "asset_interaction": {},
            "evidence": {
                "class_event_count": int(class_event_count),
                "total_event_count": int(max(1, total_event_count)),
                "mean_confidence": float(mean_confidence),
            },
            "score_components": {},
            "geometry_terms": {},
            "weights": {},
            "note": "No lifted 3D hazard points were available, so severity is unrated.",
        }

    interaction = estimate_asset_interaction(
        hazard_points,
        asset_points,
        scene_points,
        metric_scale_known=metric_scale_known,
    )

    geometry_terms = _geometry_component(
        class_name,
        geometry,
        metric_scale_known=metric_scale_known,
    )

    total_event_count = max(1, int(total_event_count))
    class_event_count = max(0, min(int(class_event_count), total_event_count))

    persistence_raw = class_event_count / total_event_count
    persistence_component = math.sqrt(_clamp01(persistence_raw))
    confidence_component = _confidence_transform(mean_confidence)
    support_component = _saturating(geometry["point_count"], 5000.0)
    obstruction_component = _clamp01(interaction["interaction_ratio"])
    geometry_component = _clamp01(geometry_terms["geometry_score"])

    score01 = (
        0.85 * geometry_component
        + 0.10 * obstruction_component
        + 0.02 * persistence_component
        + 0.02 * confidence_component
        + 0.01 * support_component
    )

    score = 100.0 * _clamp01(score01)
    stars = score_to_stars(score)

    return {
        "class_name": class_name,
        "severity_score": float(score),
        "severity_stars": int(stars),
        "severity_level": score_to_level(score),
        "scoring_ready": True,
        "metric_scale_known": bool(metric_scale_known),
        "scale_mode": "metric_lidar" if metric_scale_known else "relative_reconstruction",
        "geometry": geometry,
        "asset_interaction": interaction,
        "evidence": {
            "class_event_count": int(class_event_count),
            "total_event_count": int(total_event_count),
            "persistence_fraction": float(persistence_raw),
            "mean_confidence": float(mean_confidence),
        },
        "score_components": {
            "geometry": float(geometry_component),
            "asset_interaction": float(obstruction_component),
            "persistence": float(persistence_component),
            "confidence": float(confidence_component),
            "point_support": float(support_component),
        },
        "geometry_terms": geometry_terms,
        "weights": {
            "geometry": 0.85,
            "asset_interaction": 0.10,
            "persistence": 0.02,
            "confidence": 0.02,
            "point_support": 0.01,
        },
        "note": (
            "Metric-scale assisted severity. Physical dimensions should still be validated "
            "against calibration / known measurements."
            if metric_scale_known
            else
            "Relative severity from reconstruction geometry. Dimensions are normalized by "
            "local scene scale and must not be reported as metres until metric scale is validated."
        ),
    }


def calculate_location_severity(
    hazards: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Location severity equals the most severe hazard class at that location.

    There is deliberately no automatic mixed-hazard bonus or minimum star
    rating. Mixed locations naturally become severe only when at least one
    measured hazard is severe.
    """
    hazards = list(hazards)
    scored = [
        item
        for item in hazards
        if item.get("severity_score") is not None
    ]

    if not scored:
        return {
            "severity_score": None,
            "severity_stars": None,
            "severity_level": "unrated",
            "scoring_ready": False,
            "aggregation_method": "maximum_class_severity",
            "dominant_hazard": None,
        }

    dominant = max(
        scored,
        key=lambda item: float(item["severity_score"]),
    )

    score = float(dominant["severity_score"])

    return {
        "severity_score": score,
        "severity_stars": int(score_to_stars(score)),
        "severity_level": score_to_level(score),
        "scoring_ready": True,
        "aggregation_method": "maximum_class_severity",
        "dominant_hazard": dominant.get("class_name"),
        "rated_hazard_count": len(scored),
    }


def save_severity_metrics(path, metrics: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
