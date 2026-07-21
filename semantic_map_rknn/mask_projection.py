"""Project valid depth pixels selected by an instance segmentation mask."""
# 使用sam分割采用该mask投影
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semantic_map_rknn.bbox_projection import CameraIntrinsics


@dataclass(frozen=True)
class ProjectedMask:
    """One mask's camera-frame points and source image pixels."""

    points_camera: np.ndarray
    image_uv: np.ndarray


def project_mask_depth(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    pixel_stride: int = 2,
    min_depth_m: float = 0.1,
    max_depth_m: float = 10.0,
) -> ProjectedMask | None:
    """Project only mask-selected valid depth pixels into camera XYZ."""
    if depth_m.ndim != 2 or mask.ndim != 2:
        raise ValueError("depth_m and mask must both be 2D")
    if depth_m.shape != mask.shape:
        raise ValueError(
            f"mask/depth shape mismatch: mask={mask.shape}, depth={depth_m.shape}"
        )
    if intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    stride = max(1, int(pixel_stride))
    sampled_v, sampled_u = np.nonzero(mask[::stride, ::stride])
    if sampled_u.size == 0:
        return None
    u = sampled_u.astype(np.int32) * stride
    v = sampled_v.astype(np.int32) * stride
    z = depth_m[v, u].astype(np.float32, copy=False)
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    if not np.any(valid):
        return None

    u = u[valid].astype(np.float32)
    v = v[valid].astype(np.float32)
    z = z[valid]
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return ProjectedMask(
        points_camera=np.column_stack((x, y, z)).astype(np.float32),
        image_uv=np.column_stack((u, v)).astype(np.float32),
    )
