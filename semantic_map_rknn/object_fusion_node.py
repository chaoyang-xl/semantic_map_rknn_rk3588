#!/usr/bin/env python3
"""语义点云追踪与融合算法的 ROS 2 适配节点。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray

from semantic_map_rknn.object_tracker import (
    ObjectObservation,
    ObjectTracker,
)
from semantic_map_rknn.object_map_io import object_record, write_semantic_object_map
from semantic_map_rknn.offline_projector_node import (
    pack_rgb_uint32,
    unpack_rgb_uint32,
)
from semantic_map_rknn.point_cloud_io import save_object_ply


class ObjectFusionNode(Node):
    """按时间戳同步投影点云和检测元数据，再调用纯算法层。"""

    def __init__(self) -> None:
        super().__init__("semantic_map_rknn_fusion")
        defaults = {
            "cloud_topic": "/semantic_offline/points",
            "metadata_topic": "/semantic_offline/detections",
            "fused_cloud_topic": "/semantic_offline/fused_points",
            "objects_topic": "/semantic_offline/objects",
            "marker_topic": "/semantic_offline/object_markers",
            "publish_markers": True,
            "snapshot_path": "",
            "output_directory": "",
            "voxel_size": 0.02,
            "overlap_radius": 0.04,
            "max_centroid_distance_m": 1.0,
            "min_geometric_overlap": 0.05,
            "min_bbox_overlap": 0.0,
            "association_threshold": 0.45,
            "geometry_weight": 0.7,
            "semantic_weight": 0.3,
            "observation_cluster_eps": 0.10,
            "observation_cluster_min_points": 10,
            "max_extent_growth": 2.0,
            "association_max_points": 4096,
            "denoise_interval": 0,
            "map_merge_interval": 20,
            "map_merge_overlap": 0.80,
            "min_confirmed_observations": 3,
            "candidate_max_missed_frames": 30,
            "stale_after_s": 0.0,
            "non_fusing_labels": "person",
            "sync_buffer_size": 50,
            "snapshot_interval_s": 2.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        tracker_names = (
            "voxel_size", "overlap_radius", "max_centroid_distance_m",
            "min_geometric_overlap", "min_bbox_overlap",
            "association_threshold", "geometry_weight", "semantic_weight",
            "observation_cluster_eps", "observation_cluster_min_points",
            "max_extent_growth", "association_max_points", "denoise_interval", "map_merge_interval",
            "map_merge_overlap", "min_confirmed_observations",
            "candidate_max_missed_frames", "stale_after_s",
        )
        self._fusion = ObjectTracker(**{
            name: self.get_parameter(name).value for name in tracker_names
        })
        self._non_fusing_labels = {
            label.strip()
            for label in str(self.get_parameter("non_fusing_labels").value).split(",")
            if label.strip()
        }
        self._snapshot_path = str(self.get_parameter("snapshot_path").value).strip()
        self._output_directory = str(
            self.get_parameter("output_directory").value
        ).strip()
        # Create configured destinations immediately so launch/path problems are
        # visible before the first successful projection arrives.
        if self._output_directory:
            Path(self._output_directory).expanduser().mkdir(parents=True, exist_ok=True)
        if self._snapshot_path:
            Path(self._snapshot_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._buffer_size = max(1, int(self.get_parameter("sync_buffer_size").value))
        self._snapshot_interval_s = max(
            0.0, float(self.get_parameter("snapshot_interval_s").value)
        )
        self._last_snapshot_stamp: float | None = None
        self._projection_mode = "bbox"
        # 两个 topic 到达顺序不固定，因此使用完整 ROS 时间戳双向等待。
        self._cloud_buffer: dict[tuple[int, int], PointCloud2] = {}
        self._metadata_buffer: dict[tuple[int, int], dict] = {}

        input_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(PointCloud2, str(self.get_parameter("cloud_topic").value), self._cloud_cb, input_qos)
        self.create_subscription(String, str(self.get_parameter("metadata_topic").value), self._metadata_cb, input_qos)
        self._cloud_pub = self.create_publisher(PointCloud2, str(self.get_parameter("fused_cloud_topic").value), 10)
        self._objects_pub = self.create_publisher(String, str(self.get_parameter("objects_topic").value), 10)
        self._publish_markers = bool(self.get_parameter("publish_markers").value)
        self._marker_pub = None
        self._published_marker_ids: set[int] = set()
        if self._publish_markers:
            self._marker_pub = self.create_publisher(
                MarkerArray, str(self.get_parameter("marker_topic").value), 10
            )
        self.get_logger().info(
            "Object tracker ready: bounded nearest-neighbour geometry + "
            "semantic history, "
            f"association_max_points={self._fusion.association_max_points}, "
            f"denoise_interval={self._fusion.denoise_interval}, "
            f"non_fusing={sorted(self._non_fusing_labels)}"
        )

    def _cloud_cb(self, msg: PointCloud2) -> None:
        key = self._stamp_key(msg.header.stamp)
        self._cloud_buffer[key] = msg
        self._trim_buffer(self._cloud_buffer)
        self._try_process(key)

    def _metadata_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            key = (int(payload["stamp_sec"]), int(payload["stamp_nanosec"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().warn(f"Invalid projection metadata: {exc}")
            return
        self._metadata_buffer[key] = payload
        self._trim_buffer(self._metadata_buffer)
        self._try_process(key)

    def _try_process(self, key: tuple[int, int]) -> None:
        """同一时间戳的点云与元数据都到齐后，只处理一次。"""
        cloud = self._cloud_buffer.get(key)
        metadata = self._metadata_buffer.get(key)
        if cloud is None or metadata is None:
            return
        del self._cloud_buffer[key]
        del self._metadata_buffer[key]
        cloud_arrays = self._read_projected_cloud(cloud)
        self._projection_mode = str(
            metadata.get("projection_mode", self._projection_mode)
        )
        metadata_by_id = {int(item["detection_id"]): item for item in metadata.get("detections", [])}
        stamp = float(key[0]) + float(key[1]) * 1e-9
        observations = []
        # detection_id 用来把组合 PointCloud2 重新拆成逐对象观测。
        for detection_id_value in np.unique(cloud_arrays["detection_id"]):
            detection_id = int(detection_id_value)
            item = metadata_by_id.get(detection_id)
            if item is None:
                continue
            mask = cloud_arrays["detection_id"] == detection_id
            class_name = str(item.get("class_name", "unknown"))
            observations.append(ObjectObservation(
                detection_id=detection_id,
                class_id=int(item.get("class_id", -1)),
                class_name=class_name,
                confidence=float(item.get("confidence", 0.0)),
                stamp=stamp,
                points=cloud_arrays["xyz"][mask],
                fuse_geometry=class_name not in self._non_fusing_labels,
                colors=(
                    None
                    if cloud_arrays["rgb"] is None
                    else unpack_rgb_uint32(cloud_arrays["rgb"][mask])
                ),
            ))
        associations = self._fusion.update(observations)
        if associations:
            self._publish_state(cloud.header)
            self.get_logger().info(
                f"Associated {len(observations)} observations; "
                f"active={len(self._fusion.tracks)}, "
                f"confirmed={len(self._fusion.confirmed_tracks)}"
            )

    @staticmethod
    def _read_projected_cloud(msg: PointCloud2) -> dict[str, np.ndarray]:
        """按字段 offset 读取混合类型 PointCloud2，避免逐点解析。"""
        offsets = {field.name: field.offset for field in msg.fields}
        required = {"x", "y", "z", "detection_id"}
        if not required.issubset(offsets):
            raise ValueError(f"PointCloud2 missing fields: {sorted(required - offsets.keys())}")
        order = ">" if msg.is_bigendian else "<"
        names = ("x", "y", "z", "detection_id")
        dtype = np.dtype({
            "names": names,
            "formats": [order + "f4", order + "f4", order + "f4", order + "u4"],
            "offsets": [offsets[name] for name in names],
            "itemsize": msg.point_step,
        })
        records = np.frombuffer(msg.data, dtype=dtype, count=int(msg.width) * int(msg.height))
        result = {
            "xyz": np.column_stack((records["x"], records["y"], records["z"])).astype(np.float32),
            "detection_id": records["detection_id"].astype(np.uint32),
        }
        if "rgb" in offsets:
            rgb_dtype = np.dtype({
                "names": ("rgb",),
                "formats": [order + "u4"],
                "offsets": [offsets["rgb"]],
                "itemsize": msg.point_step,
            })
            rgb_records = np.frombuffer(
                msg.data,
                dtype=rgb_dtype,
                count=int(msg.width) * int(msg.height),
            )
            result["rgb"] = rgb_records["rgb"].astype(np.uint32)
        else:
            result["rgb"] = None
        return result

    def _publish_state(self, source_header) -> None:
        """Publish the fused cloud and refresh Replica-style object files."""
        point_rows = []
        arrays = {}
        all_tracks = sorted(
            self._fusion.tracks.values(), key=lambda item: item.track_id
        )
        tracks = [track for track in all_tracks if track.status == "confirmed"]
        for track in tracks:
            track_colors = getattr(track, "colors", None)
            packed_colors = (
                pack_rgb_uint32(track_colors)
                if track_colors is not None
                else np.full(
                    track.points.shape[0],
                    self._track_color(track.track_id),
                    dtype=np.uint32,
                )
            )
            point_rows.extend(
                (float(p[0]), float(p[1]), float(p[2]), int(color), track.track_id,
                 track.class_id, float(track.confidence), track.observation_count)
                for p, color in zip(track.points, packed_colors)
            )
            arrays[f"track_{track.track_id}"] = track.points
            if track_colors is not None:
                arrays[f"track_{track.track_id}_rgb"] = track_colors
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
            PointField(name="track_id", offset=16, datatype=PointField.UINT32, count=1),
            PointField(name="class_id", offset=20, datatype=PointField.INT32, count=1),
            PointField(name="confidence", offset=24, datatype=PointField.FLOAT32, count=1),
            PointField(name="observation_count", offset=28, datatype=PointField.UINT32, count=1),
        ]
        header = Header(stamp=source_header.stamp, frame_id=source_header.frame_id)
        self._cloud_pub.publish(point_cloud2.create_cloud(header, fields, point_rows))
        self._publish_object_markers(header, all_tracks)
        stamp = float(source_header.stamp.sec) + float(source_header.stamp.nanosec) * 1e-9
        snapshot_due = (
            self._last_snapshot_stamp is None
            or self._snapshot_interval_s <= 0.0
            or stamp - self._last_snapshot_stamp >= self._snapshot_interval_s
        )
        if snapshot_due:
            payload = self._write_object_snapshot(source_header.frame_id, tracks, arrays)
            self._objects_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
            self._last_snapshot_stamp = stamp

    def _publish_object_markers(self, header: Header, tracks: list) -> None:
        """实时发布融合对象的 3D 边界和属性，避免依赖磁盘快照刷新。"""
        if self._marker_pub is None:
            return
        markers = []
        active_ids = set()
        for track in tracks:
            bounds_min, bounds_max = track.bounds
            marker_id = int(track.track_id)
            active_ids.add(marker_id)
            packed = self._track_color(marker_id)
            red = float((packed >> 16) & 0xFF) / 255.0
            green = float((packed >> 8) & 0xFF) / 255.0
            blue = float(packed & 0xFF) / 255.0
            center = (bounds_min + bounds_max) * 0.5
            size = np.maximum(bounds_max - bounds_min, 0.02)

            box = Marker()
            box.header = header
            box.ns = "object_bounds"
            box.id = marker_id
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x = float(center[0])
            box.pose.position.y = float(center[1])
            box.pose.position.z = float(center[2])
            box.pose.orientation.w = 1.0
            box.scale.x = float(size[0])
            box.scale.y = float(size[1])
            box.scale.z = float(size[2])
            box.color.r = red
            box.color.g = green
            box.color.b = blue
            box.color.a = 0.20 if track.status == "confirmed" else 0.08
            markers.append(box)

            label = Marker()
            label.header = header
            label.ns = "object_labels"
            label.id = marker_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(center[0])
            label.pose.position.y = float(center[1])
            label.pose.position.z = float(bounds_max[2] + 0.12)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.12
            label.color.r = red
            label.color.g = green
            label.color.b = blue
            label.color.a = 1.0
            label.text = (
                f"#{track.track_id} [{track.status}] {track.class_name} "
                f"n={track.observation_count} conf={track.confidence:.2f} "
                f"pts={track.points.shape[0]}"
            )
            markers.append(label)

        for stale_id in self._published_marker_ids - active_ids:
            for namespace in ("object_bounds", "object_labels"):
                marker = Marker()
                marker.header = header
                marker.ns = namespace
                marker.id = stale_id
                marker.action = Marker.DELETE
                markers.append(marker)
        self._published_marker_ids = active_ids
        self._marker_pub.publish(MarkerArray(markers=markers))

    def _write_object_snapshot(self, frame_id: str, tracks: list, arrays: dict) -> dict:
        """Write one PLY/NPZ pair per object and a navigation-compatible JSON."""
        json_path = Path(self._snapshot_path).expanduser() if self._snapshot_path else None
        output_root = Path(self._output_directory).expanduser() if self._output_directory else None
        if output_root is None and json_path is not None:
            output_root = json_path.parent / f"{json_path.stem}_output"
        if json_path is None and output_root is not None:
            json_path = output_root / "semantic_objects.json"
        objects_directory = output_root / "objects" if output_root is not None else None
        if objects_directory is not None:
            objects_directory.mkdir(parents=True, exist_ok=True)

        records = []
        expected_object_files = set()
        for track in tracks:
            track_colors = getattr(track, "colors", None)
            safe_name = "".join(
                char if char.isalnum() or char in ("-", "_") else "_"
                for char in track.class_name
            )
            stem = f"object_{track.track_id:04d}_{safe_name}"
            expected_object_files.update({f"{stem}.ply", f"{stem}.npz"})
            if objects_directory is not None:
                ply_file = objects_directory / f"{stem}.ply"
                npz_file = objects_directory / f"{stem}.npz"
                object_arrays = {
                    "points_map": track.points,
                    "track_id": np.asarray(track.track_id, dtype=np.int32),
                    "class_id": np.asarray(track.class_id, dtype=np.int32),
                    "class_name": np.asarray(track.class_name),
                    "confidence": np.asarray(track.confidence, dtype=np.float32),
                    "observation_count": np.asarray(
                        track.observation_count, dtype=np.int32
                    ),
                    "first_seen": np.asarray(track.first_seen, dtype=np.float64),
                    "last_seen": np.asarray(track.last_seen, dtype=np.float64),
                    "status": np.asarray(getattr(track, "status", "confirmed")),
                    "missed_frames": np.asarray(
                        getattr(track, "missed_frames", 0), dtype=np.int32
                    ),
                }
                if track_colors is not None:
                    object_arrays["rgb"] = track_colors
                semantic_scores = getattr(track, "semantic_scores", {})
                object_arrays["semantic_scores"] = np.asarray(
                    json.dumps(semantic_scores, sort_keys=True)
                )
                np.savez_compressed(
                    npz_file,
                    **object_arrays,
                )
                save_object_ply(ply_file, track.points, track_colors)
                path_base = json_path.parent if json_path is not None else output_root
                ply_reference = os.path.relpath(ply_file, path_base)
                npz_reference = os.path.relpath(npz_file, path_base)
            else:
                ply_reference = f"objects/{stem}.ply"
                npz_reference = f"objects/{stem}.npz"
            records.append(object_record(
                track_id=track.track_id,
                class_id=track.class_id,
                class_name=track.class_name,
                confidence=track.confidence,
                observation_count=track.observation_count,
                first_seen=track.first_seen,
                last_seen=track.last_seen,
                points=track.points,
                ply_path=ply_reference,
                npz_path=npz_reference,
                source=f"ros_{getattr(self, '_projection_mode', 'bbox')}_object_tracking",
                semantic_scores=getattr(track, "semantic_scores", {}),
                status=getattr(track, "status", "confirmed"),
                missed_frames=getattr(track, "missed_frames", 0),
            ))

        if objects_directory is not None:
            for pattern in ("object_*.ply", "object_*.npz"):
                for stale_file in objects_directory.glob(pattern):
                    if stale_file.name not in expected_object_files:
                        stale_file.unlink()

        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_root / "fused_objects.npz", **arrays)
        if json_path is not None:
            return write_semantic_object_map(
                json_path, records, frame_id=frame_id,
                source="semantic_map_rknn_ros",
                metadata={"output_directory": str(output_root)},
            )
        return {
            "schema_version": 1, "frame_id": frame_id,
            "count": len(records), "source": "semantic_map_rknn_ros",
            "objects": records,
        }

    def _trim_buffer(self, buffer: dict) -> None:
        """限制未配对消息缓存，避免长 bag 缺帧时持续占用内存。"""
        while len(buffer) > self._buffer_size:
            del buffer[next(iter(buffer))]

    @staticmethod
    def _stamp_key(stamp) -> tuple[int, int]:
        return int(stamp.sec), int(stamp.nanosec)

    @staticmethod
    def _track_color(track_id: int) -> int:
        value = int(track_id) * 2654435761 & 0xFFFFFFFF
        return ((64 + (value & 0xBF)) << 16) | ((64 + ((value >> 8) & 0xBF)) << 8) | (64 + ((value >> 16) & 0xBF))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ObjectFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
