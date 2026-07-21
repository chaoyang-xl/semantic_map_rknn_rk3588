"""ROS 2 RKNN MobileSAM mask-to-map projection node."""

from __future__ import annotations

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from .mask_projection import ProjectedMask, project_mask_depth
from .mobilesam_rknn import MobileSamRknn
from .offline_projector_node import OfflineProjectorNode


class SamRknnProjectorNode(OfflineProjectorNode):
    """Use one RKNN encoder pass and one decoder pass per YOLO box."""

    def __init__(self) -> None:
        super().__init__(node_name="semantic_map_rknn_projector")
        for name, default in (
            ("sam_encoder", ""), ("sam_decoder", ""),
            ("rknn_backend", "auto"), ("rknn_target", "rk3588"),
            ("sam_encoder_core", "0_1_2"), ("sam_decoder_core", "0_1_2"),
            ("mask_threshold", 0.0), ("mask_erode_px", 2),
            ("publish_debug_image", True),
            ("debug_image_topic", "/semantic_rknn/sam_debug_image"),
            ("debug_mask_alpha", 0.45),
        ):
            self.declare_parameter(name, default)
        encoder = str(self.get_parameter("sam_encoder").value).strip()
        decoder = str(self.get_parameter("sam_decoder").value).strip()
        if not encoder or not decoder:
            raise ValueError("sam_encoder and sam_decoder are required")
        self._sam = MobileSamRknn(
            encoder, decoder,
            backend=str(self.get_parameter("rknn_backend").value),
            target=str(self.get_parameter("rknn_target").value),
            encoder_core=str(self.get_parameter("sam_encoder_core").value),
            decoder_core=str(self.get_parameter("sam_decoder_core").value),
            mask_threshold=float(self.get_parameter("mask_threshold").value),
            mask_erode_px=int(self.get_parameter("mask_erode_px").value),
        )
        self._debug_alpha = float(np.clip(
            self.get_parameter("debug_mask_alpha").value, 0.0, 1.0
        ))
        self._debug_pub = None
        if bool(self.get_parameter("publish_debug_image").value):
            qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
            self._debug_pub = self.create_publisher(
                Image, str(self.get_parameter("debug_image_topic").value), qos
            )
        self.get_logger().info(
            f"RKNN MobileSAM projector ready: backend={self._sam.encoder.backend}, "
            f"image_size={self._sam.image_size}"
        )

    @property
    def projection_mode(self) -> str:
        return "rknn_mobilesam"

    @property
    def requires_rgb(self) -> bool:
        return True

    def _prepare_projection_frame(self, msg, payload, depth_m, source_size):
        context = super()._prepare_projection_frame(msg, payload, depth_m, source_size)
        rgb = context["rgb_image"]
        if rgb is None:
            raise RuntimeError("No RGB frame available for RKNN MobileSAM")
        detections = payload.get("detections", [])
        if not detections:
            return {**context, "masks": [], "scores": []}
        boxes = np.asarray([item.get("xyxy", []) for item in detections], dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise RuntimeError("RKNN MobileSAM requires one xyxy box per detection")
        source_width, source_height = source_size
        boxes[:, (0, 2)] *= rgb.shape[1] / source_width
        boxes[:, (1, 3)] *= rgb.shape[0] / source_height
        self._sam.set_image(rgb)
        results = self._sam.predict_boxes(boxes)
        display_masks = np.stack([item.mask for item in results])
        scores = np.asarray([item.score for item in results], dtype=np.float32)
        masks = display_masks
        if masks.shape[1:] != depth_m.shape:
            masks = np.stack([
                cv2.resize(
                    mask.astype(np.uint8),
                    (depth_m.shape[1], depth_m.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                for mask in masks
            ])
        self._publish_debug(msg, rgb, payload, display_masks, scores)
        return {**context, "masks": masks, "scores": scores}

    def _publish_debug(self, msg, rgb, payload, masks, scores) -> None:
        if self._debug_pub is None:
            return
        overlay = rgb.astype(np.float32)
        for index, (detection, mask) in enumerate(zip(payload.get("detections", []), masks)):
            if float(detection.get("confidence", 0.0)) < self._min_confidence:
                continue
            if self._is_excluded_detection(detection):
                continue
            packed = self._class_color(int(detection.get("class_id", -1)))
            color = np.asarray(
                [(packed >> 16) & 255, (packed >> 8) & 255, packed & 255],
                dtype=np.float32,
            )
            overlay[mask] = (
                (1.0 - self._debug_alpha) * overlay[mask]
                + self._debug_alpha * color
            )
            x1, y1, x2, y2 = (int(round(v)) for v in detection["xyxy"])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color.tolist(), 2)
            cv2.putText(
                overlay,
                f"{detection.get('class_name', 'unknown')} sam={scores[index]:.2f}",
                (max(0, x1), max(16, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color.tolist(), 1, cv2.LINE_AA,
            )
        debug = self._bridge.cv2_to_imgmsg(
            np.clip(overlay, 0, 255).astype(np.uint8), encoding="rgb8"
        )
        debug.header = msg.header
        self._debug_pub.publish(debug)

    def _project_detection(self, depth_m, detection, detection_id, source_size, context):
        if detection_id >= len(context["masks"]):
            return None
        mask = context["masks"][detection_id]
        projected = project_mask_depth(
            depth_m, mask, self._intrinsics,
            pixel_stride=self._pixel_stride,
            min_depth_m=self._min_depth_m,
            max_depth_m=self._max_depth_m,
        )
        if projected is None:
            return None
        source_width, source_height = source_size
        uv = projected.image_uv.copy()
        uv[:, 0] = (uv[:, 0] + 0.5) * source_width / depth_m.shape[1] - 0.5
        uv[:, 1] = (uv[:, 1] + 0.5) * source_height / depth_m.shape[0] - 0.5
        return ProjectedMask(projected.points_camera, uv), {
            "sam_score": float(context["scores"][detection_id]),
            "mask_area_pixels": int(np.count_nonzero(mask)),
        }

    def destroy_node(self):
        self._sam.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SamRknnProjectorNode()
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
