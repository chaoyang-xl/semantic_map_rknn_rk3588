#!/usr/bin/env python3
"""Start optional YOLO-World RKNN, MobileSAM projection, and object fusion."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("semantic_map_rknn"))
    defaults = {
        "run_yolo": "false",
        "use_sim_time": "false",
        "image_topic": "/camera/color/image_raw",
        "depth_topic": "/camera/depth/image_raw",
        "detections_topic": "/yolo/results_json",
        "target_frame": "map",
        "camera_frame": "camera_color_optical_frame",
        "camera_fx": "365.1741638183594",
        "camera_fy": "365.42144775390625",
        "camera_cx": "318.27630615234375",
        "camera_cy": "243.80377197265625",
        "depth_scale": "0.001",
        "min_depth": "0.3",
        "max_depth": "5.0",
        "sam_encoder": "/home/orangepi/models/mobile_sam/mobilesam_encoder_tiny.rknn",
        "sam_decoder": "/home/orangepi/models/mobile_sam/mobilesam_decoder.rknn",
        "yolo_model": "/home/orangepi/models/yolo_world_rknn/yolo_world_v2s_i8.rknn",
        "clip_text_model": "/home/orangepi/models/yolo_world_rknn/clip_text_fp16.rknn",
        "text_embeddings": "",
        "classes_path": str(share / "config" / "indoor_classes_80.txt"),
        "rknn_backend": "lite",
        "npu_core": "0_1_2",
        "confidence": "0.50",
        "frame_skip": "0",
        "output_directory": "/home/orangepi/semantic_map_output",
        "snapshot_path": "/home/orangepi/semantic_map_output/semantic_objects.json",
        "fusion_workers": "1",
        "publish_debug_image": "true",
        "publish_markers": "true",
    }
    arguments = [
        DeclareLaunchArgument(name, default_value=value)
        for name, value in defaults.items()
    ]
    yolo = Node(
        package="semantic_map_rknn",
        executable="yolo_world_rknn_node",
        name="yolo_world_rknn",
        output="screen",
        condition=IfCondition(LaunchConfiguration("run_yolo")),
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "image_topic": LaunchConfiguration("image_topic"),
            "result_topic": LaunchConfiguration("detections_topic"),
            "model_path": LaunchConfiguration("yolo_model"),
            "text_model_path": LaunchConfiguration("clip_text_model"),
            "text_embeddings_path": LaunchConfiguration("text_embeddings"),
            "classes_path": LaunchConfiguration("classes_path"),
            "confidence": LaunchConfiguration("confidence"),
            "frame_skip": LaunchConfiguration("frame_skip"),
            "rknn_backend": LaunchConfiguration("rknn_backend"),
            "npu_core": LaunchConfiguration("npu_core"),
            "publish_debug_image": LaunchConfiguration("publish_debug_image"),
        }],
    )
    projector = Node(
        package="semantic_map_rknn",
        executable="sam_rknn_projector_node",
        name="sam_rknn_projector",
        output="screen",
        parameters=[
            str(share / "config" / "semantic_mapping.yaml"),
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "input_topic": LaunchConfiguration("detections_topic"),
                "color_topic": LaunchConfiguration("image_topic"),
                "depth_topic": LaunchConfiguration("depth_topic"),
                "target_frame": LaunchConfiguration("target_frame"),
                "camera_frame": LaunchConfiguration("camera_frame"),
                "camera_fx": LaunchConfiguration("camera_fx"),
                "camera_fy": LaunchConfiguration("camera_fy"),
                "camera_cx": LaunchConfiguration("camera_cx"),
                "camera_cy": LaunchConfiguration("camera_cy"),
                "depth_scale": LaunchConfiguration("depth_scale"),
                "min_depth_m": LaunchConfiguration("min_depth"),
                "max_depth_m": LaunchConfiguration("max_depth"),
                "min_confidence": LaunchConfiguration("confidence"),
                "sam_encoder": LaunchConfiguration("sam_encoder"),
                "sam_decoder": LaunchConfiguration("sam_decoder"),
                "rknn_backend": LaunchConfiguration("rknn_backend"),
                "sam_encoder_core": LaunchConfiguration("npu_core"),
                "sam_decoder_core": LaunchConfiguration("npu_core"),
                "publish_debug_image": LaunchConfiguration("publish_debug_image"),
            },
        ],
    )
    fusion = Node(
        package="semantic_map_rknn",
        executable="object_fusion_node",
        name="semantic_rknn_fusion",
        output="screen",
        parameters=[
            str(share / "config" / "semantic_mapping.yaml"),
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "output_directory": LaunchConfiguration("output_directory"),
                "snapshot_path": LaunchConfiguration("snapshot_path"),
                "publish_markers": LaunchConfiguration("publish_markers"),
                "fusion_workers": LaunchConfiguration("fusion_workers"),
            },
        ],
    )
    return LaunchDescription([*arguments, yolo, projector, fusion])
