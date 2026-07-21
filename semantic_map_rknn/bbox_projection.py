"""将 YOLO 检测框内的有效深度像素向量化投影为三维点。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    '''相机内参。'''
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class ProjectedBox:
    """保留3D点及两套像素坐标，供后续反投影和 IoU 验证。"""
    points_camera: np.ndarray # (N,3) 相机坐标系下的3D点
    depth_uv: np.ndarray      # (N,2) 深度图像坐标系下的像素坐标
    image_uv: np.ndarray      # (N,2) RGB图像坐标系下的像素坐标
    depth_bbox: tuple[int, int, int, int]  # 深度图空间中的裁剪框 (left,top,right,bottom)


def image_size_from_payload(payload: dict, fallback: tuple[int, int]) -> tuple[int, int]:
    """从YOLO推理结果中提取图像尺寸."""
    shape = payload.get("image_shape")
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        try:
            height, width = int(shape[0]), int(shape[1])
            if width > 0 and height > 0:
                return width, height
        except (TypeError, ValueError):
            pass
    return fallback


def scale_and_clip_bbox(
    xyxy: list[float] | tuple[float, ...],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """
    将RGB图像上的检测框 xyxy 按分辨率比例映射到深度图像空间，并裁剪到有效范围内。
    这个函数是解决多模态传感器对齐问题的关键：即使没有精确的外参注册，
    也能通过分辨率比例进行近似映射。
    返回检测框映射到深度图像坐标系后的像素边界。

    """
    if len(xyxy) != 4:
        return None
    source_width, source_height = source_size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        return None

    try:
        x1, y1, x2, y2 = (float(value) for value in xyxy)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite([x1, y1, x2, y2])):
        return None

    # RGB 与深度尚未注册时，仅按两幅图像的分辨率进行比例近似。
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    left = max(0, min(target_width, int(np.floor(min(x1, x2) * scale_x))))
    top = max(0, min(target_height, int(np.floor(min(y1, y2) * scale_y))))
    right = max(0, min(target_width, int(np.ceil(max(x1, x2) * scale_x))))
    bottom = max(0, min(target_height, int(np.ceil(max(y1, y2) * scale_y))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def project_bbox_depth(
    depth_m: np.ndarray,
    xyxy: list[float] | tuple[float, ...],
    source_size: tuple[int, int],
    intrinsics: CameraIntrinsics,
    pixel_stride: int = 2,
    min_depth_m: float = 0.3,
    max_depth_m: float = 5.0,
) -> ProjectedBox | None:
    """
        完成从2D检测框到3D点云的全部工作
        将所有结果封装成 ProjectedBox 返回；若无任何有效深度，返回 None。

    """
    if depth_m.ndim != 2:
        raise ValueError("depth_m must be a 2D array")
    if intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    height, width = depth_m.shape
    depth_bbox = scale_and_clip_bbox(xyxy, source_size, (width, height))
    if depth_bbox is None:
        return None

    left, top, right, bottom = depth_bbox
    stride = max(1, int(pixel_stride))
    u_values = np.arange(left, right, stride, dtype=np.int32)
    v_values = np.arange(top, bottom, stride, dtype=np.int32)
    if u_values.size == 0 or v_values.size == 0:
        return None
    uu, vv = np.meshgrid(u_values, v_values)
    zz = depth_m[vv, uu].astype(np.float32, copy=False)
    # 一次过滤 0、NaN、Inf 以及相机有效量程之外的深度。
    valid = np.isfinite(zz) & (zz >= min_depth_m) & (zz <= max_depth_m)
    if not np.any(valid):
        return None

    u_valid = uu[valid].astype(np.float32)
    v_valid = vv[valid].astype(np.float32)
    z_valid = zz[valid]
    # 针孔模型：像素坐标和深度反投影到相机光学坐标系 XYZ。
    x_valid = (u_valid - intrinsics.cx) * z_valid / intrinsics.fx
    y_valid = (v_valid - intrinsics.cy) * z_valid / intrinsics.fy
    points = np.column_stack((x_valid, y_valid, z_valid)).astype(np.float32)
    depth_uv = np.column_stack((u_valid, v_valid)).astype(np.float32)

    source_width, source_height = source_size
    image_uv = np.column_stack((
        (u_valid + 0.5) * source_width / width - 0.5,
        (v_valid + 0.5) * source_height / height - 0.5,
    )).astype(np.float32)
    return ProjectedBox(points, depth_uv, image_uv, depth_bbox)


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """将ROS风格的归一化四元数 (x, y, z, w) 转换为 3X3 旋转矩阵。"""
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    scale = 2.0 / norm
    return np.array([
        [1.0 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
        [scale * (x * y + z * w), 1.0 - scale * (x * x + z * z), scale * (y * z - x * w)],
        [scale * (x * z - y * w), scale * (y * z + x * w), 1.0 - scale * (x * x + y * y)],
    ], dtype=np.float64)


def transform_points(points: np.ndarray, transform) -> np.ndarray:
    """
        应用一个 geometry_msgs/Transform 消息
        （包含 rotation 四元数和 translation 平移向量）到 (N,3) 点云
    """
    rotation = transform.rotation
    translation = transform.translation
    matrix = quaternion_matrix(rotation.x, rotation.y, rotation.z, rotation.w)
    offset = np.array([translation.x, translation.y, translation.z], dtype=np.float64)
    return (points.astype(np.float64) @ matrix.T + offset).astype(np.float32)
