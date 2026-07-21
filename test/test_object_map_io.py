"""Tests for navigation-compatible semantic object JSON serialization."""

import json

import numpy as np

from semantic_map_rknn.object_map_io import object_record, write_semantic_object_map


def test_object_record_contains_navigation_and_full_3d_fields(tmp_path):
    points = np.array([[1.0, 2.0, 0.2], [1.8, 2.6, 1.0]], dtype=np.float32)
    record = object_record(
        track_id=7,
        class_id=56,
        class_name="chair",
        confidence=0.91,
        observation_count=12,
        first_seen=3.0,
        last_seen=20.0,
        points=points,
        ply_path="objects/object_0007_chair.ply",
        npz_path="objects/object_0007_chair.npz",
        source="bbox_object_tracking",
        semantic_scores={56: 4.2, 57: 0.3},
    )
    assert record["id"] == "chair_7"
    assert record["state"] == "confirmed"
    assert record["x"] == np.mean(points[:, 0])
    assert np.isclose(record["size_x"], 0.8)
    assert record["times_seen"] == 12
    assert record["bounds_min"] == [1.0, 2.0, 0.20000000298023224]
    assert record["point_cloud_ply"].endswith(".ply")
    assert record["semantic_scores"] == {"56": 4.2, "57": 0.3}


def test_write_semantic_object_map_is_navigation_shaped(tmp_path):
    output = tmp_path / "semantic_objects.json"
    payload = write_semantic_object_map(
        output,
        [],
        frame_id="map",
        source="test",
        metadata={"min_observations": 5},
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == payload
    assert loaded["frame_id"] == "map"
    assert loaded["count"] == 0
    assert loaded["objects"] == []
