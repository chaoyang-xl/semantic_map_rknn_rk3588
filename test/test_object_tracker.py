"""Tests for object association, lifecycle, and point-cloud fusion."""

import numpy as np

from semantic_map_rknn.object_tracker import (
    ObjectObservation,
    ObjectTracker,
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


def test_same_object_is_associated_and_confirmed():
    mapping = tracker()
    first = mapping.update([observation(0, [0, 0, 1], stamp=1.0)])[0]
    second = mapping.update([observation(0, [0.01, 0, 1], stamp=2.0)])[0]
    assert first.track_id == second.track_id
    assert mapping.tracks[first.track_id].status == "confirmed"


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


def test_parallel_tracker_matches_serial_results():
    serial = tracker(worker_count=1, denoise_interval=1)
    parallel = tracker(worker_count=4, denoise_interval=1)
    frames = [
        [
            observation(0, [0, 0, 1], stamp=1.0),
            observation(1, [1, 0, 1], stamp=1.0),
        ],
        [
            observation(0, [0.01, 0, 1], stamp=2.0),
            observation(1, [1.01, 0, 1], stamp=2.0),
        ],
        [
            observation(0, [0.02, 0, 1], stamp=3.0),
            observation(1, [1.02, 0, 1], stamp=3.0),
        ],
    ]
    try:
        for items in frames:
            serial_matches = serial.update(items)
            parallel_matches = parallel.update(items)
            assert [
                (item.observation_index, item.track_id, item.is_new)
                for item in parallel_matches
            ] == [
                (item.observation_index, item.track_id, item.is_new)
                for item in serial_matches
            ]
        assert parallel.worker_count == 4
        assert sorted(parallel.tracks) == sorted(serial.tracks)
        for track_id in serial.tracks:
            serial_track = serial.tracks[track_id]
            parallel_track = parallel.tracks[track_id]
            assert parallel_track.observation_count == serial_track.observation_count
            np.testing.assert_allclose(parallel_track.points, serial_track.points)
    finally:
        serial.close()
        parallel.close()


def test_voxel_downsample_preserves_rgb_alignment():
    points = np.asarray([[0.01, 0, 0], [0.02, 0, 0]], dtype=np.float32)
    colors = np.asarray([[100, 20, 0], [200, 40, 10]], dtype=np.uint8)
    sampled_points, sampled_colors = voxel_downsample_with_colors(
        points, colors, 0.1
    )
    assert sampled_points.shape == sampled_colors.shape == (1, 3)
    np.testing.assert_array_equal(sampled_colors[0], [150, 30, 5])
