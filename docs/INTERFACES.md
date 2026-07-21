# Interfaces

## Detection JSON Input

Topic: `/yolo/results_json` (`std_msgs/msg/String`)

Required fields:

```json
{
  "stamp_sec": 1710000000,
  "stamp_nanosec": 123000000,
  "image_shape": [480, 640, 3],
  "detections": [
    {
      "class_id": 0,
      "class_name": "chair",
      "confidence": 0.83,
      "xyxy": [120.0, 80.0, 320.0, 420.0]
    }
  ]
}
```

This is compatible with `opi_yolo_rknn_recorder`.

## ROS Outputs

| Topic | Type | Content |
| --- | --- | --- |
| `/semantic_rknn/points` | `PointCloud2` | Per-frame segmented points in `map` |
| `/semantic_rknn/detections` | `String` | Per-frame projection metadata |
| `/semantic_rknn/fused_points` | `PointCloud2` | Confirmed fused object clouds |
| `/semantic_objects` | `String` | Navigation-compatible confirmed objects |
| `/semantic_rknn/object_markers` | `MarkerArray` | Object bounds and labels |
| `/semantic_rknn/sam_debug_image` | `Image` | RGB mask/box overlay |

## Files

```text
semantic_map_output/
  semantic_objects.json
  fused_objects.npz
  objects/
    object_0001_chair.ply
    object_0001_chair.npz
```

Only confirmed objects enter the navigation JSON and object files.
