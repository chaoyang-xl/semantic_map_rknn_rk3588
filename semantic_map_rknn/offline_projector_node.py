#!/usr/bin/env python3
"""ROS 2 offline YOLO-box depth projection into a map-frame PointCloud2."""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
import json
from pathlib import Path

import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String

from semantic_map_rknn.bbox_projection import (
    CameraIntrinsics,
    image_size_from_payload,
    project_bbox_depth,
    transform_points,
)
from semantic_map_rknn.spatial_filter import largest_spatial_cluster_indices


def pack_rgb_uint32(colors: np.ndarray) -> np.ndarray:
    """Pack uint8 RGB rows into the ROS/PCL 0xRRGGBB representation."""
    rgb = np.asarray(colors, dtype=np.uint8)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("colors must have shape (N, 3)")
    return (
        (rgb[:, 0].astype(np.uint32) << 16)
        | (rgb[:, 1].astype(np.uint32) << 8)
        | rgb[:, 2].astype(np.uint32)
    )


def unpack_rgb_uint32(packed: np.ndarray) -> np.ndarray:
    """Unpack ROS/PCL 0xRRGGBB values into uint8 RGB rows."""
    values = np.asarray(packed, dtype=np.uint32)
    return np.column_stack(
        ((values >> 16) & 0xFF, (values >> 8) & 0xFF, values & 0xFF)
    ).astype(np.uint8)


def voxel_representative_indices(
    points: np.ndarray, voxel_size: float
) -> np.ndarray:
    """Select one deterministic representative per occupied XYZ voxel."""
    point_array = np.asarray(points, dtype=np.float32)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if voxel_size <= 0.0 or point_array.shape[0] == 0:
        return np.arange(point_array.shape[0], dtype=np.int64)
    keys = np.floor(point_array / voxel_size).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return np.sort(indices.astype(np.int64))


def sample_rgb_image(
    rgb_image: np.ndarray,
    image_uv: np.ndarray,
    source_size: tuple[int, int],
) -> np.ndarray:
    """Sample RGB at detection-image coordinates, scaling when sizes differ."""
    image = np.asarray(rgb_image, dtype=np.uint8)
    uv = np.asarray(image_uv, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb_image must have shape (H, W, 3)")
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("image_uv must have shape (N, 2)")
    source_width, source_height = source_size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source_size must be positive")
    u = np.floor((uv[:, 0] + 0.5) * image.shape[1] / source_width).astype(np.int64)
    v = np.floor((uv[:, 1] + 0.5) * image.shape[0] / source_height).astype(np.int64)
    u = np.clip(u, 0, image.shape[1] - 1)
    v = np.clip(v, 0, image.shape[0] - 1)
    return image[v, u].copy()


def detection_is_excluded(
    detection: dict,
    excluded_labels: set[str],
    excluded_class_ids: set[int],
) -> bool:
    """Return whether a detector result is disabled by label or class ID."""
    class_name = str(detection.get("class_name", "")).strip().casefold()
    class_id = int(detection.get("class_id", -1))
    return class_name in excluded_labels or class_id in excluded_class_ids


class OfflineProjectorNode(Node):
    """Match recorded detections and depth, then project complete box ROIs."""

    def __init__(self, node_name: str = "semantic_map_rknn_projector") -> None:
        super().__init__(node_name)
        '''读取参数并初始化订阅器和发布器。'''
        for name, default in (
            ("input_topic", "/yolo/results_json"),
            ("depth_topic", "/camera/depth/image_raw"),
            ("color_topic", "/camera/color/image_raw"),
            ("cloud_topic", "/semantic_offline/points"),
            ("metadata_topic", "/semantic_offline/detections"),
            ("jsonl_path", ""),
            ("save_directory", ""),
            ("camera_frame", ""),
            ("target_frame", "map"),
            ("camera_fx", 311.3878784),
            ("camera_fy", 311.3878784),
            ("camera_cx", 317.5),
            ("camera_cy", 198.5),
            ("depth_scale", 0.001),
            ("min_depth_m", 0.3),
            ("max_depth_m", 5.0),
            ("pixel_stride", 2),
            ("projection_voxel_size", 0.0),
            ("projection_cluster_eps", 0.0),
            ("projection_cluster_min_points", 10),
            ("max_time_diff_s", 0.15),
            ("detection_buffer_size", 100),
            ("color_buffer_size", 30),
            ("processing_delay_frames", 0),
            ("tf_timeout_s", 0.3),
            ("min_confidence", 0.0),
            ("excluded_labels", "person"),
            ("excluded_class_ids", ""),
            ("image_qos_reliable", False),
        ):
            self.declare_parameter(name, default)

        self._intrinsics = CameraIntrinsics(*(
            float(self.get_parameter(name).value)
            for name in ("camera_fx", "camera_fy", "camera_cx", "camera_cy")
        ))
        self._depth_scale = float(self.get_parameter("depth_scale").value)
        self._min_depth_m = float(self.get_parameter("min_depth_m").value)
        self._max_depth_m = float(self.get_parameter("max_depth_m").value)
        self._pixel_stride = max(1, int(self.get_parameter("pixel_stride").value))
        self._projection_voxel_size = max(
            0.0, float(self.get_parameter("projection_voxel_size").value)
        )
        self._projection_cluster_eps = max(
            0.0, float(self.get_parameter("projection_cluster_eps").value)
        )
        self._projection_cluster_min_points = max(
            1, int(self.get_parameter("projection_cluster_min_points").value)
        )
        self._max_time_diff_s = float(self.get_parameter("max_time_diff_s").value)
        self._tf_timeout_s = float(self.get_parameter("tf_timeout_s").value)
        self._processing_delay_frames = max(
            0, int(self.get_parameter("processing_delay_frames").value)
        )
        self._min_confidence = float(self.get_parameter("min_confidence").value)
        self._excluded_labels = {
            item.strip().casefold()
            for item in str(self.get_parameter("excluded_labels").value).split(",")
            if item.strip()
        }
        self._excluded_class_ids = {
            int(item.strip())
            for item in str(self.get_parameter("excluded_class_ids").value).split(",")
            if item.strip()
        }
        self._camera_frame = str(self.get_parameter("camera_frame").value)
        self._target_frame = str(self.get_parameter("target_frame").value)
        self._save_directory = str(self.get_parameter("save_directory").value).strip()
        if self._save_directory:
            Path(self._save_directory).expanduser().mkdir(parents=True, exist_ok=True)

        self._bridge = CvBridge()
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        buffer_size = max(1, int(self.get_parameter("detection_buffer_size").value))
        self._detections = deque(maxlen=buffer_size)
        self._depth_buffer = deque(maxlen=buffer_size)
        self._color_buffer = deque(
            maxlen=max(1, int(self.get_parameter("color_buffer_size").value))
        )
        self._handled_depth_stamps = deque(maxlen=buffer_size * 2)
        self._handled_depth_stamp_set: set[tuple[int, int]] = set()
        self._jsonl_records: list[tuple[float, dict]] = []
        self._jsonl_stamps: list[float] = []
        jsonl_path = str(self.get_parameter("jsonl_path").value).strip()
        if jsonl_path:
            self._jsonl_records = self._load_jsonl(Path(jsonl_path).expanduser())
            self._jsonl_stamps = [item[0] for item in self._jsonl_records]

        reliable_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        sensor_qos = (
            reliable_qos
            if bool(self.get_parameter("image_qos_reliable").value)
            else QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )
        self.create_subscription(
            Image, str(self.get_parameter("depth_topic").value), self._depth_cb, sensor_qos
        )
        self.create_subscription(
            Image, str(self.get_parameter("color_topic").value), self._color_cb, sensor_qos
        )
        if not self._jsonl_records:
            self.create_subscription(
                String, str(self.get_parameter("input_topic").value), self._json_cb, reliable_qos
            )
        self._cloud_pub = self.create_publisher(
            PointCloud2, str(self.get_parameter("cloud_topic").value), reliable_qos
        )
        self._metadata_pub = self.create_publisher(
            String, str(self.get_parameter("metadata_topic").value), reliable_qos
        )
        mode = f"JSONL {jsonl_path}" if self._jsonl_records else "JSON topic"
        self.get_logger().info(
            f"Offline projector ready: {mode}, stride={self._pixel_stride}, "
            f"projection_voxel={self._projection_voxel_size:.3f}m, "
            f"projection_cluster={self._projection_cluster_eps:.3f}m/"
            f"{self._projection_cluster_min_points}, "
            f"target={self._target_frame}, excluded={sorted(self._excluded_labels)}, "
            f"image_qos={'reliable' if sensor_qos.reliability == ReliabilityPolicy.RELIABLE else 'best_effort'}"
        )

    def _color_cb(self, msg: Image) -> None:
        self._color_buffer.append(msg)
        self.get_logger().info(
            f"RGB input active: {msg.width}x{msg.height}, frame={msg.header.frame_id}",
            once=True,
        )
        if self.requires_rgb and len(self._depth_buffer) > self._processing_delay_frames:
            delayed_index = -(self._processing_delay_frames + 1)
            self._process_depth(self._depth_buffer[delayed_index])

    def _load_jsonl(self, path: Path) -> list[tuple[float, dict]]:
        records = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    stamp = self._payload_stamp(payload)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    self.get_logger().warn(f"Skip JSONL line {line_number}: {exc}")
                    continue
                if stamp is not None:
                    records.append((stamp, payload))
        records.sort(key=lambda item: item[0])
        self.get_logger().info(f"Loaded {len(records)} timestamped JSON records")
        return records

    def _json_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            stamp = self._payload_stamp(payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self.get_logger().warn(f"Invalid detection JSON: {exc}")
            return
        if stamp is None:
            self.get_logger().warn("Detection JSON has no valid ROS timestamp")
            return
        self._detections.append((stamp, payload))
        # A delayed depth pipeline must be driven by _depth_cb. Processing the
        # newest depth here bypasses the delay and can request TF slightly in
        # the future relative to recorded odometry.
        if self._processing_delay_frames > 0:
            return
        if self._depth_buffer:
            depth_msg = min(
                self._depth_buffer,
                key=lambda item: abs(self._stamp_seconds(item.header.stamp) - stamp),
            )
            time_diff = abs(self._stamp_seconds(depth_msg.header.stamp) - stamp)
            if time_diff <= self._max_time_diff_s:
                self._process_depth(depth_msg)

    def _depth_cb(self, msg: Image) -> None:
        self.get_logger().info(
            f"Depth input active: {msg.width}x{msg.height}, frame={msg.header.frame_id}",
            throttle_duration_sec=10.0,
        )
        self._depth_buffer.append(msg)
        if self.requires_rgb and not self._color_buffer:
            self.get_logger().warn(
                "Waiting for first RGB frame before MobileSAM projection",
                throttle_duration_sec=5.0,
            )
            return
        if len(self._depth_buffer) <= self._processing_delay_frames:
            return
        delayed_index = -(self._processing_delay_frames + 1)
        self._process_depth(self._depth_buffer[delayed_index])

    def _process_depth(self, msg: Image) -> None:

        '''
            它负责把一帧深度图与对应的检测结果配对，
            然后一次性完成所有检测框的 3D 投影、坐标变换、点云组装和输出。
        '''
        stamp_key = (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))
        if stamp_key in self._handled_depth_stamp_set:
            return
        depth_stamp = self._stamp_seconds(msg.header.stamp)
        payload, time_diff = self._nearest_payload(depth_stamp)
        if payload is None or time_diff > self._max_time_diff_s:
            self.get_logger().warn(
                f"No detection record for depth stamp; nearest dt={time_diff:.3f}s",
                throttle_duration_sec=2.0,
            )
            return
        self.get_logger().info(
            f"Projection stage: matched detection dt={time_diff:.3f}s",
            once=True,
        )
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().error(f"Depth conversion failed: {exc}")
            return
        if depth.ndim != 2:
            self.get_logger().warn(f"Expected 2D depth image, got shape={depth.shape}")
            return
        depth_m = depth.astype(np.float32, copy=False) * self._depth_scale
        source_size = image_size_from_payload(payload, (msg.width, msg.height))
        source_frame = self._camera_frame or msg.header.frame_id

        self.get_logger().info("Projection stage: requesting map TF", once=True)
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self._target_frame,
                source_frame,
                rclpy.time.Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=self._tf_timeout_s),
            )
        except Exception as exc:
            self.get_logger().warn(
                f"TF failed at depth stamp: {source_frame}->{self._target_frame}: {exc}"
            )
            return
        self.get_logger().info("Projection stage: map TF ready", once=True)

        try:
            projection_context = self._prepare_projection_frame(msg, payload, depth_m, source_size)
        except RuntimeError as exc:
            self.get_logger().warn(str(exc), throttle_duration_sec=5.0)
            return
        point_chunks = []
        metadata = []
        for detection_id, detection in enumerate(payload.get("detections", [])):
            confidence = float(detection.get("confidence", 0.0))
            if confidence < self._min_confidence or self._is_excluded_detection(detection):
                continue
            result = self._project_detection(
                depth_m, detection, detection_id, source_size, projection_context
            )
            if result is None:
                continue
            projected, projection_metadata = result
            points_map = transform_points(projected.points_camera, tf_msg.transform)
            raw_count = points_map.shape[0]
            keep = voxel_representative_indices(
                points_map, self._projection_voxel_size
            )
            points_map = points_map[keep]
            image_uv = projected.image_uv[keep]
            voxel_count = points_map.shape[0]
            cluster_keep = largest_spatial_cluster_indices(
                points_map,
                self._projection_cluster_eps,
                self._projection_cluster_min_points,
            )
            points_map = points_map[cluster_keep]
            image_uv = image_uv[cluster_keep]
            count = points_map.shape[0]
            class_id = int(detection.get("class_id", -1))
            rgb_image = projection_context.get("rgb_image")
            if rgb_image is None:
                packed_rgb = np.full(
                    count, self._class_color(class_id), dtype=np.uint32
                )
                color_mode = "class_fallback"
            else:
                colors = sample_rgb_image(rgb_image, image_uv, source_size)
                packed_rgb = pack_rgb_uint32(colors)
                color_mode = "camera_rgb"
            point_chunks.append({
                "xyz": points_map,
                "image_uv": image_uv,
                "class_id": np.full(count, class_id, dtype=np.int32),
                "detection_id": np.full(count, detection_id, dtype=np.uint32),
                "confidence": np.full(count, confidence, dtype=np.float32),
                "rgb": packed_rgb,
            })
            metadata.append({
                "detection_id": detection_id,
                "class_id": class_id,
                "class_name": detection.get("class_name", "unknown"),
                "confidence": confidence,
                "xyxy": detection.get("xyxy", []),
                "point_count": count,
                "raw_point_count": raw_count,
                "voxel_point_count": voxel_count,
                "cluster_removed_count": voxel_count - count,
                "projection_cluster_eps": self._projection_cluster_eps,
                "projection_mode": self.projection_mode,
                "color_mode": color_mode,
                **projection_metadata,
            })

        if not point_chunks:
            self.get_logger().warn(
                "Detection matched, but no valid depth points remained in its boxes: "
                f"detections={len(payload.get('detections', []))}, "
                f"depth_range=[{self._min_depth_m:.2f}, {self._max_depth_m:.2f}]m",
                throttle_duration_sec=5.0,
            )
            return
        arrays = {
            key: np.concatenate([chunk[key] for chunk in point_chunks])
            for key in point_chunks[0]
        }
        cloud = self._make_cloud(msg.header.stamp, arrays)
        self._cloud_pub.publish(cloud)
        output_metadata = {
            "stamp_sec": msg.header.stamp.sec,
            "stamp_nanosec": msg.header.stamp.nanosec,
            "frame_id": self._target_frame,
            "source_image_size": list(source_size),
            "depth_image_size": [msg.width, msg.height],
            "time_diff_s": time_diff,
            "pixel_stride": self._pixel_stride,
            "projection_voxel_size": self._projection_voxel_size,
            "point_count": int(arrays["xyz"].shape[0]),
            "projection_mode": self.projection_mode,
            "detections": metadata,
        }
        self._metadata_pub.publish(String(data=json.dumps(output_metadata, ensure_ascii=False)))
        self._save_frame(msg.header.stamp, arrays, output_metadata)
        self._mark_depth_handled(stamp_key)
        self.get_logger().info(
            f"Projected {len(metadata)} {self.projection_mode} objects, {arrays['xyz'].shape[0]} points, dt={time_diff:.3f}s"
        )
    @property
    def projection_mode(self) -> str:
        return "bbox"

    @property
    def requires_rgb(self) -> bool:
        return False

    def _prepare_projection_frame(self, msg, payload, depth_m, source_size):
        if not self._color_buffer:
            self.get_logger().warn(
                "No RGB frame available; using class colors",
                throttle_duration_sec=5.0,
            )
            return {"rgb_image": None, "color_time_diff_s": None}
        depth_stamp = self._stamp_seconds(msg.header.stamp)
        color_msg = min(
            self._color_buffer,
            key=lambda item: abs(self._stamp_seconds(item.header.stamp) - depth_stamp),
        )
        time_diff = abs(self._stamp_seconds(color_msg.header.stamp) - depth_stamp)
        if time_diff > self._max_time_diff_s:
            self.get_logger().warn(
                f"No synchronized RGB frame; dt={time_diff:.3f}s, using class colors",
                throttle_duration_sec=5.0,
            )
            return {"rgb_image": None, "color_time_diff_s": time_diff}
        try:
            rgb_image = self._bridge.imgmsg_to_cv2(
                color_msg, desired_encoding="rgb8"
            )
        except Exception as exc:
            self.get_logger().warn(f"RGB conversion failed: {exc}")
            return {"rgb_image": None, "color_time_diff_s": time_diff}
        return {"rgb_image": rgb_image, "color_time_diff_s": time_diff}

    def _project_detection(self, depth_m, detection, detection_id, source_size, context):
        '''
            调用 project_bbox_depth（来自 bbox_projection 模块），
            获得 ProjectedBox（包含相机系点云、深度图坐标、图像坐标等）。
            返回 ProjectedBox 对象和投影元数据（如深度图中的实际边界框 depth_xyxy）。

        '''
        projected = project_bbox_depth(
            depth_m, detection.get("xyxy", []), source_size, self._intrinsics,
            pixel_stride=self._pixel_stride, min_depth_m=self._min_depth_m,
            max_depth_m=self._max_depth_m,
        )
        if projected is None:
            return None
        return projected, {"depth_xyxy": list(projected.depth_bbox)}


    def _mark_depth_handled(self, stamp_key: tuple[int, int]) -> None:
        if len(self._handled_depth_stamps) == self._handled_depth_stamps.maxlen:
            oldest = self._handled_depth_stamps.popleft()
            self._handled_depth_stamp_set.discard(oldest)
        self._handled_depth_stamps.append(stamp_key)
        self._handled_depth_stamp_set.add(stamp_key)

    def _nearest_payload(self, stamp: float) -> tuple[dict | None, float]:
        '''
            利用二分查找（bisect_left）在排序后的 JSONL 记录中定位最接近的时间戳；
            若无 JSONL，则在线性缓冲中搜索最小时间差的检测。返回最匹配的 payload 及时间差。
        '''
        if self._jsonl_records:
            index = bisect_left(self._jsonl_stamps, stamp)
            candidates = []
            if index < len(self._jsonl_records):
                candidates.append(self._jsonl_records[index])
            if index > 0:
                candidates.append(self._jsonl_records[index - 1])
            best_stamp, best_payload = min(
                candidates, key=lambda item: abs(item[0] - stamp)
            )
            return best_payload, abs(best_stamp - stamp)

        records = list(self._detections)
        if not records:
            return None, float("inf")
        best_stamp, best_payload = min(records, key=lambda item: abs(item[0] - stamp))
        return best_payload, abs(best_stamp - stamp)

    def _is_excluded_detection(self, detection: dict) -> bool:
        return detection_is_excluded(
            detection, self._excluded_labels, self._excluded_class_ids
        )

    def _make_cloud(self, stamp, arrays: dict[str, np.ndarray]) -> PointCloud2:
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
            PointField(name="class_id", offset=16, datatype=PointField.INT32, count=1),
            PointField(name="detection_id", offset=20, datatype=PointField.UINT32, count=1),
            PointField(name="confidence", offset=24, datatype=PointField.FLOAT32, count=1),
            PointField(name="image_u", offset=28, datatype=PointField.FLOAT32, count=1),
            PointField(name="image_v", offset=32, datatype=PointField.FLOAT32, count=1),
        ]
        xyz = arrays["xyz"]
        uv = arrays["image_uv"]
        points = zip(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], arrays["rgb"], arrays["class_id"],
            arrays["detection_id"], arrays["confidence"], uv[:, 0], uv[:, 1],
        )
        return point_cloud2.create_cloud(
            Header(stamp=stamp, frame_id=self._target_frame), fields, points
        )

    def _save_frame(self, stamp, arrays: dict[str, np.ndarray], metadata: dict) -> None:
        if not self._save_directory:
            return
        stem = f"{stamp.sec}_{stamp.nanosec:09d}"
        output = Path(self._save_directory).expanduser()
        np.savez_compressed(output / f"{stem}.npz", **arrays)
        (output / f"{stem}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _payload_stamp(payload: dict) -> float | None:
        sec = payload.get("stamp_sec")
        nanosec = payload.get("stamp_nanosec")
        if sec is None or nanosec is None:
            return None
        return float(sec) + float(nanosec) * 1e-9

    @staticmethod
    def _stamp_seconds(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def _class_color(class_id: int) -> int:
        value = (int(class_id) + 1) * 2654435761 & 0xFFFFFFFF
        red = 64 + (value & 0xBF)
        green = 64 + ((value >> 8) & 0xBF)
        blue = 64 + ((value >> 16) & 0xBF)
        return (red << 16) | (green << 8) | blue


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = OfflineProjectorNode()
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
