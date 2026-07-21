#!/usr/bin/env python3
"""Render fused semantic object point clouds as a map-frame XY view."""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from semantic_map_rknn.top_down_projection import (  # noqa: E402
    make_top_down_layout,
    xy_bounds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project object_*.npz point clouds onto map XY."
    )
    parser.add_argument("--objects-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG path; defaults to <objects parent>/semantic_objects_xy.png.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="2D object JSON; defaults to <objects parent>/semantic_objects_xy.json.",
    )
    parser.add_argument("--min-observations", type=int, default=5)
    parser.add_argument("--classes", nargs="*", default=[])
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--pixels-per-meter", type=float, default=180.0)
    parser.add_argument("--margin-m", type=float, default=0.5)
    parser.add_argument("--min-canvas-px", type=int, default=700)
    parser.add_argument("--max-canvas-px", type=int, default=2400)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument(
        "--color-mode",
        choices=("object", "rgb"),
        default="object",
        help="Use stable per-object colors by default, or saved camera RGB.",
    )
    parser.add_argument("--background", choices=("light", "dark"), default="light")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def object_color(track_id: int) -> tuple[int, int, int]:
    """Return a stable high-contrast BGR color."""
    hue = (track_id * 0.61803398875) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.78, 0.86)
    return tuple(int(round(channel * 255)) for channel in rgb[::-1])


def load_objects(args: argparse.Namespace) -> list[dict]:
    class_filter = set(args.classes)
    objects = []
    for path in sorted(args.objects_dir.glob("object_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            class_name = str(data["class_name"])
            observations = int(data["observation_count"])
            if observations < args.min_observations:
                continue
            if class_filter and class_name not in class_filter:
                continue
            points = np.asarray(data["points_map"], dtype=np.float32)
            finite = np.all(np.isfinite(points), axis=1)
            points = points[finite]
            if points.shape[0] == 0:
                continue
            rgb = None
            if "rgb" in data.files:
                candidate = np.asarray(data["rgb"], dtype=np.uint8)
                if candidate.shape[0] == finite.shape[0] and candidate.shape[1:] == (3,):
                    rgb = candidate[finite]
            objects.append({
                "track_id": int(data["track_id"]),
                "class_id": int(data["class_id"]),
                "class_name": class_name,
                "confidence": float(data["confidence"]),
                "observations": observations,
                "first_seen": float(data["first_seen"]),
                "last_seen": float(data["last_seen"]),
                "status": str(data["status"]) if "status" in data.files else "confirmed",
                "points": points,
                "rgb": rgb,
                "path": path,
            })
    objects.sort(key=lambda item: (-item["observations"], item["track_id"]))
    return objects[:args.max_objects] if args.max_objects > 0 else objects


def draw_grid(image: np.ndarray, layout, dark: bool) -> None:
    line_color = (55, 55, 55) if dark else (218, 218, 218)
    label_color = (170, 170, 170) if dark else (105, 105, 105)
    x_start = int(np.ceil(layout.bounds.minimum[0]))
    x_end = int(np.floor(layout.bounds.maximum[0]))
    y_start = int(np.ceil(layout.bounds.minimum[1]))
    y_end = int(np.floor(layout.bounds.maximum[1]))
    for x in range(x_start, x_end + 1):
        pixels = layout.world_to_pixel(
            np.asarray([[x, layout.bounds.minimum[1]], [x, layout.bounds.maximum[1]]])
        )
        cv2.line(image, tuple(pixels[0]), tuple(pixels[1]), line_color, 1, cv2.LINE_AA)
        cv2.putText(
            image, f"{x}m", (int(pixels[0, 0]) + 4, image.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, label_color, 1, cv2.LINE_AA,
        )
    for y in range(y_start, y_end + 1):
        pixels = layout.world_to_pixel(
            np.asarray([[layout.bounds.minimum[0], y], [layout.bounds.maximum[0], y]])
        )
        cv2.line(image, tuple(pixels[0]), tuple(pixels[1]), line_color, 1, cv2.LINE_AA)
        cv2.putText(
            image, f"{y}m", (8, int(pixels[0, 1]) - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, label_color, 1, cv2.LINE_AA,
        )


def draw_points(
    image: np.ndarray, pixels: np.ndarray, rgb: np.ndarray | None,
    fallback_bgr: tuple[int, int, int], radius: int,
) -> None:
    if rgb is None:
        colors = np.broadcast_to(np.asarray(fallback_bgr, dtype=np.uint8), (len(pixels), 3))
    else:
        colors = rgb[:, ::-1]
    radius = max(0, int(radius))
    offsets = [(0, 0)]
    for distance in range(1, radius + 1):
        offsets.extend(((distance, 0), (-distance, 0), (0, distance), (0, -distance)))
    for dx, dy in offsets:
        x = np.clip(pixels[:, 0] + dx, 0, image.shape[1] - 1)
        y = np.clip(pixels[:, 1] + dy, 0, image.shape[0] - 1)
        image[y, x] = colors


def fit_font_scale(lines: list[str], target_width: int) -> float:
    scale = 0.58
    while scale > 0.34:
        width = max(
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
            for line in lines
        )
        if width <= target_width:
            break
        scale -= 0.04
    return scale


def draw_center_label(
    image: np.ndarray,
    center: np.ndarray,
    lines: list[str],
    color: tuple[int, int, int],
    box_width: int,
) -> None:
    target_width = max(70, min(280, int(box_width * 0.9)))
    scale = fit_font_scale(lines, target_width)
    sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
        for line in lines
    ]
    panel_width = max(width for width, _ in sizes) + 18
    line_height = max(height for _, height in sizes) + 8
    panel_height = line_height * len(lines) + 6
    left = int(np.clip(center[0] - panel_width / 2, 2, image.shape[1] - panel_width - 2))
    top = int(np.clip(center[1] - panel_height / 2, 2, image.shape[0] - panel_height - 2))

    overlay = image.copy()
    cv2.rectangle(
        overlay, (left, top), (left + panel_width, top + panel_height),
        (248, 248, 248), cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.76, image, 0.24, 0.0, image)
    cv2.rectangle(
        image, (left, top), (left + panel_width, top + panel_height),
        color, 1, cv2.LINE_AA,
    )
    for index, (line, (width, _)) in enumerate(zip(lines, sizes)):
        x = left + (panel_width - width) // 2
        y = top + 6 + (index + 1) * line_height - 5
        cv2.putText(
            image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            scale, (25, 25, 25), 1, cv2.LINE_AA,
        )


def render(args: argparse.Namespace, objects: list[dict]) -> tuple[Path, Path]:
    layout = make_top_down_layout(
        [item["points"] for item in objects],
        pixels_per_meter=args.pixels_per_meter,
        margin_m=args.margin_m,
        min_canvas_px=args.min_canvas_px,
        max_canvas_px=args.max_canvas_px,
    )
    dark = args.background == "dark"
    background = 24 if dark else 246
    image = np.full((layout.height, layout.width, 3), background, dtype=np.uint8)
    draw_grid(image, layout, dark)

    records = []
    for item in objects:
        color = object_color(item["track_id"])
        pixels = layout.world_to_pixel(item["points"][:, :2])
        point_rgb = item["rgb"] if args.color_mode == "rgb" else None
        draw_points(image, pixels, point_rgb, color, args.point_radius)

    for item in objects:
        bounds = xy_bounds(item["points"])
        corners = layout.world_to_pixel(
            np.asarray([
                [bounds.minimum[0], bounds.minimum[1]],
                [bounds.maximum[0], bounds.maximum[1]],
            ])
        )
        left, right = sorted((int(corners[0, 0]), int(corners[1, 0])))
        top, bottom = sorted((int(corners[0, 1]), int(corners[1, 1])))
        color = object_color(item["track_id"])
        cv2.rectangle(image, (left, top), (right, bottom), color, 2, cv2.LINE_AA)
        center = layout.world_to_pixel(bounds.center.reshape(1, 2))[0]
        lines = [item["class_name"]]
        draw_center_label(
            image, center, lines, color, right - left
        )
        records.append({
            "track_id": item["track_id"],
            "class_id": item["class_id"],
            "class_name": item["class_name"],
            "confidence": item["confidence"],
            "observation_count": item["observations"],
            "first_seen": item["first_seen"],
            "last_seen": item["last_seen"],
            "status": item["status"],
            "point_count": int(item["points"].shape[0]),
            "source_npz": str(item["path"]),
            "center_map_xy": bounds.center.tolist(),
            "bounds_map_xy": {
                "minimum": bounds.minimum.tolist(),
                "maximum": bounds.maximum.tolist(),
            },
            "size_xy_m": bounds.size.tolist(),
            "height_range_m": [
                float(np.min(item["points"][:, 2])),
                float(np.max(item["points"][:, 2])),
            ],
            "rectangle_pixels": {
                "left": left, "top": top, "right": right, "bottom": bottom,
            },
        })

    output = args.output or args.objects_dir.parent / "semantic_objects_xy.png"
    json_output = args.json_output or args.objects_dir.parent / "semantic_objects_xy.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"Failed to write image: {output}")
    document = {
        "schema_version": 1,
        "frame_id": "map",
        "projection": "xy",
        "image": str(output),
        "image_size": {"width": layout.width, "height": layout.height},
        "pixels_per_meter": layout.pixels_per_meter,
        "color_mode": args.color_mode,
        "map_bounds_xy": {
            "minimum": layout.bounds.minimum.tolist(),
            "maximum": layout.bounds.maximum.tolist(),
        },
        "object_count": len(records),
        "objects": records,
    }
    json_output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output, json_output


def main() -> None:
    args = parse_args()
    if not args.objects_dir.is_dir():
        raise SystemExit(f"Objects directory does not exist: {args.objects_dir}")
    objects = load_objects(args)
    if not objects:
        raise SystemExit(
            "No objects matched. Try --min-observations 1 or remove --classes."
        )
    output, json_output = render(args, objects)
    print(f"2D semantic map: {output}")
    print(f"2D object JSON: {json_output}")
    print(
        f"Objects: {len(objects)}, points: "
        f"{sum(item['points'].shape[0] for item in objects)}"
    )
    if args.show:
        image = cv2.imread(str(output), cv2.IMREAD_COLOR)
        cv2.imshow("semantic_map_rknn XY semantic objects", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
