#!/usr/bin/env python3
"""Interactively inspect fused per-track point clouds with Open3D."""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from semantic_map_rknn.object_map_io import object_record, write_semantic_object_map

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display semantic_map_rknn objects/*.npz in one 3D scene."
    )
    parser.add_argument("--objects-dir", "--tracks-dir", dest="objects_dir", metavar="OBJECTS_DIR", type=Path, required=True)
    parser.add_argument(
        "--min-observations", type=int, default=5,
        help="Hide objects observed fewer times; use 1 to include every object.",
    )
    parser.add_argument(
        "--json-output", type=Path, default=None,
        help="Navigation JSON path; defaults to <objects parent>/semantic_objects.json.",
    )
    parser.add_argument(
        "--classes", nargs="*", default=[],
        help="Optional class filter, for example: chair couch.",
    )
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument(
        "--display-voxel-size", type=float, default=0.0,
        help="Optional display-only voxel downsampling in metres.",
    )
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument(
        "--color-mode", choices=("rgb", "track"), default="rgb",
        help="Use saved camera RGB when available, or stable per-track colors.",
    )
    parser.add_argument("--show-boxes", action="store_true")
    parser.add_argument("--show-origin", action="store_true")
    parser.add_argument("--background", choices=("dark", "light"), default="dark")
    return parser.parse_args()


def track_color(track_id: int) -> np.ndarray:
    """Generate a stable, high-contrast RGB color from a track ID."""
    hue = (track_id * 0.61803398875) % 1.0
    return np.asarray(colorsys.hsv_to_rgb(hue, 0.72, 0.95), dtype=np.float64)


def load_tracks(args: argparse.Namespace) -> list[dict]:
    tracks = []
    class_filter = set(args.classes)
    paths = list(args.objects_dir.glob("object_*.npz")) + list(args.objects_dir.glob("track*.npz"))
    for path in sorted(paths):
        data = np.load(path, allow_pickle=False)
        class_name = str(data["class_name"])
        observations = int(data["observation_count"])
        if observations < args.min_observations:
            continue
        if class_filter and class_name not in class_filter:
            continue
        points = np.asarray(data["points_map"], dtype=np.float64)
        finite = np.all(np.isfinite(points), axis=1)
        colors = None
        if "rgb" in data.files:
            candidate = np.asarray(data["rgb"], dtype=np.float64)
            if candidate.shape == points.shape:
                colors = candidate[finite] / 255.0
        points = points[finite]
        if points.size == 0:
            continue
        tracks.append({
            "track_id": int(data["track_id"]),
            "class_name": class_name,
            "class_id": int(data["class_id"]),
            "confidence": float(data["confidence"]),
            "npz_path": path,
            "observations": observations,
            "first_seen": float(data["first_seen"]),
            "last_seen": float(data["last_seen"]),
            "points": points,
            "colors": colors,
        })
    tracks.sort(key=lambda item: (-item["observations"], item["track_id"]))
    return tracks[:args.max_tracks] if args.max_tracks > 0 else tracks


def print_track_table(tracks: list[dict]) -> None:
    print("\nobject class             obs    points   frames       confidence")
    print("-----  ----------------  -----  -------  -----------  ----------")
    for item in tracks:
        frame_range = f"{item['first_seen']:.0f}-{item['last_seen']:.0f}"
        print(
            f"{item['track_id']:5d}  {item['class_name'][:16]:16s}  "
            f"{item['observations']:5d}  {len(item['points']):7d}  "
            f"{frame_range:11s}  {item['confidence']:.3f}"
        )
    print(f"\nDisplayed objects: {len(tracks)}")
    print(f"Displayed points: {sum(len(item['points']) for item in tracks)}")


def write_navigation_json(args: argparse.Namespace, tracks: list[dict]) -> Path:
    """Refresh a navigation-compatible JSON using the viewer's active filters."""
    output = args.json_output or args.objects_dir.parent / "semantic_objects.json"
    records = []
    for item in tracks:
        stem = item["npz_path"].stem
        records.append(object_record(
            track_id=item["track_id"],
            class_id=item["class_id"],
            class_name=item["class_name"],
            confidence=item["confidence"],
            observation_count=item["observations"],
            first_seen=item["first_seen"],
            last_seen=item["last_seen"],
            points=item["points"],
            ply_path=f"objects/{stem}.ply",
            npz_path=f"objects/{stem}.npz",
            source="viewer_filtered_geometric_object",
        ))
    write_semantic_object_map(
        output,
        records,
        frame_id="map",
        source="semantic_map_rknn_viewer",
        metadata={
            "objects_directory": str(args.objects_dir.resolve()),
            "min_observations": args.min_observations,
            "classes": list(args.classes),
            "max_objects": args.max_tracks,
        },
    )
    return output


def main() -> None:
    args = parse_args()
    if not args.objects_dir.is_dir():
        raise SystemExit(f"Objects directory does not exist: {args.objects_dir}")
    tracks = load_tracks(args)
    if not tracks:
        raise SystemExit(
            "No tracks matched. Try --min-observations 1 or remove --classes."
        )
    print_track_table(tracks)
    json_path = write_navigation_json(args, tracks)
    print(f"Navigation JSON: {json_path}")
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit(
            "Open3D is required. Install it with:\n  python3 -m pip install open3d"
        ) from exc
    geometries = []
    for item in tracks:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(item["points"])
        color = track_color(item["track_id"])
        if args.color_mode == "rgb" and item["colors"] is not None:
            cloud.colors = o3d.utility.Vector3dVector(item["colors"])
        else:
            cloud.paint_uniform_color(color)
        if args.display_voxel_size > 0.0:
            cloud = cloud.voxel_down_sample(args.display_voxel_size)
        geometries.append(cloud)
        if args.show_boxes:
            box = cloud.get_axis_aligned_bounding_box()
            box.color = color
            geometries.append(box)
    if args.show_origin:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        window_name="semantic_map_rknn semantic objects", width=1440, height=900
    )
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    options = visualizer.get_render_option()
    options.point_size = max(1.0, args.point_size)
    options.background_color = (
        np.asarray([0.03, 0.03, 0.03])
        if args.background == "dark"
        else np.asarray([0.95, 0.95, 0.95])
    )
    options.show_coordinate_frame = args.show_origin
    print("Controls: drag=rotate, Shift+drag=pan, wheel=zoom, Q=quit")
    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
