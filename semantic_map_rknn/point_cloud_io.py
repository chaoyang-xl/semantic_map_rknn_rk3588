"""Point-cloud file serialization shared by dataset and ROS workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_object_ply(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    """Atomically save map-frame XYZ or XYZRGB as binary little-endian PLY."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(points, dtype="<f4")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    rgb = None if colors is None else np.asarray(colors, dtype=np.uint8)
    if rgb is not None and rgb.shape != xyz.shape:
        raise ValueError("colors must have the same (N, 3) shape as points")
    temporary = output.with_suffix(output.suffix + ".tmp")
    color_properties = (
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        if rgb is not None else ""
    )
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"{color_properties}end_header\n"
    ).encode("ascii")
    with temporary.open("wb") as stream:
        stream.write(header)
        if rgb is None:
            stream.write(xyz.tobytes())
        else:
            vertices = np.empty(
                xyz.shape[0],
                dtype=[
                    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                ],
            )
            vertices["x"], vertices["y"], vertices["z"] = xyz.T
            vertices["red"], vertices["green"], vertices["blue"] = rgb.T
            stream.write(vertices.tobytes())
    temporary.replace(output)
