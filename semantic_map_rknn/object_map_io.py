"""Serialize fused object point clouds into a navigation-compatible JSON map."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def object_record(
    *,
    track_id: int,
    class_id: int,
    class_name: str,
    confidence: float,
    observation_count: int,
    first_seen: float,
    last_seen: float,
    points: np.ndarray,
    ply_path: str,
    npz_path: str,
    source: str,
    semantic_scores: dict[int, float] | None = None,
    status: str = "confirmed",
    missed_frames: int = 0,
) -> dict:
    """Build one complete object record while retaining navigation-required fields."""
    points = np.asarray(points, dtype=np.float64)
    bounds_min = np.min(points, axis=0)
    bounds_max = np.max(points, axis=0)
    centroid = np.mean(points, axis=0)
    size = bounds_max - bounds_min
    safe_label = "".join(
        char.lower() if char.isalnum() else "_" for char in class_name
    ).strip("_") or "unknown"
    return {
        "id": f"{safe_label}_{track_id}",
        "label": class_name,
        "state": status,
        "source": source,
        "x": float(centroid[0]),
        "y": float(centroid[1]),
        "z": float(centroid[2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
        "confidence": float(confidence),
        "times_seen": int(observation_count),
        "first_seen": float(first_seen),
        "last_seen": float(last_seen),
        "track_id": int(track_id),
        "class_id": int(class_id),
        "class_name": class_name,
        "centroid": centroid.tolist(),
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "point_count": int(points.shape[0]),
        "point_cloud_ply": ply_path,
        "point_cloud_npz": npz_path,
        "missed_frames": int(missed_frames),
        "semantic_scores": {
            str(class_id): float(score)
            for class_id, score in (semantic_scores or {}).items()
        },
    }


def write_semantic_object_map(
    path: str | Path,
    objects: Iterable[dict],
    *,
    frame_id: str = "map",
    source: str,
    metadata: dict | None = None,
) -> dict:
    """Write an atomic, navigation-compatible semantic object snapshot."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    object_list = list(objects)
    payload = {
        "schema_version": 1,
        "frame_id": frame_id,
        "count": len(object_list),
        "source": source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objects": object_list,
    }
    if metadata:
        payload["metadata"] = metadata
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output)
    return payload
