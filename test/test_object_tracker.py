"""Tests for object association, lifecycle, and point-cloud fusion."""

import numpy as np

from semantic_map_rknn.object_tracker import (
    ObjectObservation,
    ObjectTracker,
    merge_voxel_clouds,
    nearest_neighbor_overlap,
    voxel_downsample_with_colors,
)


def observation(
    detection_id,
    center,
    *,
    class_id=56,
    name="chair",
    confidence=0.9,
    stamp=1.0,
):
    offsets = np.asarray(
        [
            [-0.04, -0.04, 0.0],
            [0.00, -0.04, 0.0],
            [0.04, -0.04, 0.0],
            [-0.04, 0.00, 0.0],
            [0.00, 0.00, 0.0],
            [0.04, 0.00, 0.0],
            [-0.04, 0.04, 0.0],
            [0.00, 0.04, 0.0],
            [0.04, 0.04, 0.0],
        ],
        dtype=np.float32,
    )
    return ObjectObservation(
        detection_id,
        class_id,
        name,
        confidence,
        stamp,
        offsets + np.asarray(center, dtype=np.float32),
    )


def tracker(**kwargs):
    defaults = {
        "voxel_size": 0.01,
        "overlap_radius": 0.06,
        "observation_cluster_eps": 0.10,
        "observation_cluster_min_points": 3,
        "min_confirmed_observations": 2,
    }
    defaults.update(kwargs)
    return ObjectTracker(**defaults)


def test_nearest_overlap_accepts_shifted_surfaces():
    first = observation(0, [0, 0, 1]).points
    shifted = observation(0, [0.02, 0, 1]).points
    assert nearest_neighbor_overlap(first, shifted, 0.03) > 0.8


def test_nearest_overlap_rejects_empty_cloud():
    empty = np.empty((0, 3), dtype=np.float32)
    points = observation(0, [0, 0, 1]).points
    assert nearest_neighbor_overlap(empty, points, 0.03) == 0.0


def test_same_object_is_associated_and_confirmed():
    mapping = tracker()
    first = mapping.update([observation(0, [0, 0, 1], stamp=1.0)])[0]
    second = mapping.update([observation(0, [0.01, 0, 1], stamp=2.0)])[0]
    assert first.track_id == second.track_id
    assert mapping.tracks[first.track_id].status == "confirmed"


def test_track_geometry_cache_invalidates_after_merge():
    mapping = tracker()
    track_id = mapping.update(
        [observation(0, [0, 0, 1], stamp=1.0)]
    )[0].track_id
    track = mapping.tracks[track_id]
    initial_geometry = track.geometry

    mapping.update([observation(0, [0.01, 0, 1], stamp=2.0)])

    assert track.geometry is not initial_geometry
    np.testing.assert_allclose(track.geometry.centroid, np.mean(track.points, axis=0))


def test_global_assignment_prevents_double_use_of_one_track():
    mapping = tracker(min_confirmed_observations=1)
    mapping.update([observation(0, [0, 0, 1])])
    matches = mapping.update(
        [
            observation(0, [0.01, 0, 1], stamp=2.0),
            observation(1, [-0.01, 0, 1], stamp=2.0),
        ]
    )
    assert len({item.track_id for item in matches}) == 2


def test_candidate_expires_but_confirmed_track_remains():
    mapping = tracker(candidate_max_missed_frames=1)
    candidate_id = mapping.update([observation(0, [0, 0, 1])])[0].track_id
    mapping.update([])
    mapping.update([])
    assert candidate_id not in mapping.tracks

    confirmed_id = mapping.update([observation(0, [1, 0, 1], stamp=3.0)])[0].track_id
    mapping.update([observation(0, [1.01, 0, 1], stamp=4.0)])
    mapping.update([])
    mapping.update([])
    assert confirmed_id in mapping.tracks


def test_semantic_history_updates_dominant_class():
    mapping = tracker(
        geometry_weight=1.0,
        semantic_weight=0.0,
        association_threshold=0.1,
    )
    track_id = mapping.update(
        [observation(0, [0, 0, 1], class_id=56, name="chair", confidence=0.2)]
    )[0].track_id
    mapping.update(
        [
            observation(
                0,
                [0, 0, 1],
                class_id=57,
                name="couch",
                confidence=0.9,
                stamp=2.0,
            )
        ]
    )
    assert mapping.tracks[track_id].class_name == "couch"


def test_periodic_map_merge_removes_duplicate_tracks():
    mapping = tracker(
        min_confirmed_observations=1,
        map_merge_interval=1,
        map_merge_overlap=0.8,
    )
    mapping.update([observation(0, [0, 0, 1])])
    duplicate = mapping._create_track(observation(1, [0.005, 0, 1], stamp=2.0))
    assert duplicate.track_id in mapping.tracks
    mapping.update([])
    assert len(mapping.tracks) == 1


def test_voxel_downsample_preserves_rgb_alignment():
    points = np.asarray([[0.01, 0, 0], [0.02, 0, 0]], dtype=np.float32)
    colors = np.asarray([[100, 20, 0], [200, 40, 10]], dtype=np.uint8)
    sampled_points, sampled_colors = voxel_downsample_with_colors(
        points, colors, 0.1
    )
    assert sampled_points.shape == sampled_colors.shape == (1, 3)
    np.testing.assert_array_equal(sampled_colors[0], [150, 30, 5])


def test_fast_voxel_merge_matches_full_downsample():
    rng = np.random.default_rng(42)
    voxel_size = 0.05
    first_points, first_colors = voxel_downsample_with_colors(
        rng.uniform(-2.0, 2.0, size=(500, 3)).astype(np.float32),
        rng.integers(0, 256, size=(500, 3), dtype=np.uint8),
        voxel_size,
    )
    second_points, second_colors = voxel_downsample_with_colors(
        rng.uniform(-2.0, 2.0, size=(500, 3)).astype(np.float32),
        rng.integers(0, 256, size=(500, 3), dtype=np.uint8),
        voxel_size,
    )

    expected_points, expected_colors = voxel_downsample_with_colors(
        np.concatenate((first_points, second_points)),
        np.concatenate((first_colors, second_colors)),
        voxel_size,
    )
    actual_points, actual_colors = merge_voxel_clouds(
        first_points,
        first_colors,
        second_points,
        second_colors,
        voxel_size,
    )

    np.testing.assert_array_equal(actual_points, expected_points)
    np.testing.assert_array_equal(actual_colors, expected_colors)
