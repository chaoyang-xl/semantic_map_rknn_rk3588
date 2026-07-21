"""Map-frame XY projection primitives for semantic object visualization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class XYBounds:
    """Axis-aligned bounds in map metres."""

    minimum: np.ndarray
    maximum: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return (self.minimum + self.maximum) * 0.5

    @property
    def size(self) -> np.ndarray:
        return self.maximum - self.minimum


@dataclass(frozen=True)
class TopDownLayout:
    """Metric-preserving transform from map XY coordinates to image pixels."""

    bounds: XYBounds
    pixels_per_meter: float
    width: int
    height: int
    offset_x: float
    offset_y: float

    def world_to_pixel(
        self, xy: np.ndarray, *, clip: bool = True
    ) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64)
        if points.shape[-1] != 2:
            raise ValueError("xy must end with two coordinates")
        pixels = np.empty_like(points)
        pixels[..., 0] = (
            (points[..., 0] - self.bounds.minimum[0])
            * self.pixels_per_meter
            + self.offset_x
        )
        pixels[..., 1] = self.height - 1 - (
            (points[..., 1] - self.bounds.minimum[1])
            * self.pixels_per_meter
            + self.offset_y
        )
        pixels = np.rint(pixels).astype(np.int32)
        if clip:
            pixels[..., 0] = np.clip(pixels[..., 0], 0, self.width - 1)
            pixels[..., 1] = np.clip(pixels[..., 1], 0, self.height - 1)
        return pixels


def xy_bounds(points: np.ndarray) -> XYBounds:
    """Return finite XY bounds for an XYZ or XY point array."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("points must have shape (N, 2+) ")
    finite = array[np.all(np.isfinite(array[:, :2]), axis=1), :2]
    if finite.shape[0] == 0:
        raise ValueError("points contain no finite XY coordinates")
    return XYBounds(np.min(finite, axis=0), np.max(finite, axis=0))


def make_top_down_layout(
    point_sets: list[np.ndarray],
    *,
    pixels_per_meter: float = 180.0,
    margin_m: float = 0.5,
    min_canvas_px: int = 700,
    max_canvas_px: int = 2400,
) -> TopDownLayout:
    """Fit point sets into a metric-preserving top-down canvas."""
    if pixels_per_meter <= 0.0:
        raise ValueError("pixels_per_meter must be positive")
    if margin_m < 0.0:
        raise ValueError("margin_m must not be negative")
    if min_canvas_px <= 0 or max_canvas_px < min_canvas_px:
        raise ValueError("invalid canvas limits")
    if not point_sets:
        raise ValueError("at least one point set is required")

    finite_xy = []
    for points in point_sets:
        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] < 2:
            raise ValueError("each point set must have shape (N, 2+)")
        finite_xy.append(array[np.all(np.isfinite(array[:, :2]), axis=1), :2])
    finite_xy = [points for points in finite_xy if points.shape[0] > 0]
    if not finite_xy:
        raise ValueError("point sets contain no finite XY coordinates")

    combined = np.concatenate(finite_xy, axis=0)
    minimum = np.min(combined, axis=0) - margin_m
    maximum = np.max(combined, axis=0) + margin_m
    span = np.maximum(maximum - minimum, 1e-3)
    scale = min(
        float(pixels_per_meter),
        (max_canvas_px - 1) / span[0],
        (max_canvas_px - 1) / span[1],
    )
    content_width = span[0] * scale
    content_height = span[1] * scale
    width = int(np.ceil(max(min_canvas_px, content_width + 1)))
    height = int(np.ceil(max(min_canvas_px, content_height + 1)))
    offset_x = (width - 1 - content_width) * 0.5
    offset_y = (height - 1 - content_height) * 0.5
    return TopDownLayout(
        bounds=XYBounds(minimum, maximum),
        pixels_per_meter=scale,
        width=width,
        height=height,
        offset_x=offset_x,
        offset_y=offset_y,
    )
