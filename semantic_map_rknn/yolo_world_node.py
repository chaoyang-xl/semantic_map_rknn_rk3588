"""ROS 2 YOLO-World RKNN node with the existing JSON schema."""

from __future__ import annotations

from datetime import datetime
import json
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .yolo_world_rknn import YoloWorldRknn, load_class_names


class YoloWorldRknnNode(Node):
    """Publish the same timestamped detection JSON as the upstream recorder."""

    def __init__(self) -> None:
        super().__init__("yolo_world_rknn")
        defaults = {
            "image_topic": "/camera/color/image_raw",
            "result_topic": "/yolo/results_json",
            "debug_image_topic": "/yolo/debug_image",
            "publish_debug_image": True,
            "model_path": "",
            "text_model_path": "",
            "text_embeddings_path": "",
            "classes_path": "",
            "tokenizer_path": "openai/clip-vit-base-patch32",
            "confidence": 0.50,
            "nms_threshold": 0.45,
            "frame_skip": 0,
            "rknn_backend": "auto",
            "rknn_target": "rk3588",
            "npu_core": "0_1_2",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        model_path = str(self.get_parameter("model_path").value).strip()
        classes_path = str(self.get_parameter("classes_path").value).strip()
        if not model_path or not classes_path:
            raise ValueError("model_path and classes_path are required")
        text_model = str(self.get_parameter("text_model_path").value).strip() or None
        embeddings = str(self.get_parameter("text_embeddings_path").value).strip() or None
        self._detector = YoloWorldRknn(
            model_path,
            load_class_names(classes_path),
            text_model_path=text_model,
            text_embeddings_path=embeddings,
            tokenizer_path=str(self.get_parameter("tokenizer_path").value),
            confidence=float(self.get_parameter("confidence").value),
            nms_threshold=float(self.get_parameter("nms_threshold").value),
            backend=str(self.get_parameter("rknn_backend").value),
            target=str(self.get_parameter("rknn_target").value),
            core_mask=str(self.get_parameter("npu_core").value),
        )
        self._bridge = CvBridge()
        self._frame_skip = max(0, int(self.get_parameter("frame_skip").value))
        self._frame_count = 0
        self._result_pub = self.create_publisher(
            String, str(self.get_parameter("result_topic").value), 10
        )
        self._debug_pub = None
        if bool(self.get_parameter("publish_debug_image").value):
            self._debug_pub = self.create_publisher(
                Image, str(self.get_parameter("debug_image_topic").value), 2
            )
        self.create_subscription(
            Image,
            str(self.get_parameter("image_topic").value),
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"YOLO-World RKNN ready: classes={len(self._detector.class_names)}, "
            f"backend={self._detector.model.backend}"
        )

    def _image_callback(self, msg: Image) -> None:
        self._frame_count += 1
        if self._frame_skip and (self._frame_count - 1) % (self._frame_skip + 1):
            return
        bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        started = time.perf_counter()
        detections = [item.as_dict() for item in self._detector.predict(bgr)]
        payload = {
            "stamp_sec": int(msg.header.stamp.sec),
            "stamp_nanosec": int(msg.header.stamp.nanosec),
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "image_topic": str(self.get_parameter("image_topic").value),
            "image_shape": list(bgr.shape),
            "detect_model": str(self._detector.model.model_path),
            "pose_model": None,
            "detections": detections,
            "poses": [],
            "posture_summary": {},
            "fall_suspected": False,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "infer_count": self._frame_count,
        }
        self._result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        if self._debug_pub is not None:
            debug = bgr.copy()
            for item in detections:
                x1, y1, x2, y2 = (int(round(value)) for value in item["xyxy"])
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 220, 80), 2)
                cv2.putText(
                    debug,
                    f"{item['class_name']} {item['confidence']:.2f}",
                    (max(0, x1), max(16, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 220, 80),
                    1,
                    cv2.LINE_AA,
                )
            debug_msg = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            debug_msg.header = msg.header
            self._debug_pub.publish(debug_msg)

    def destroy_node(self):
        self._detector.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloWorldRknnNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
