from concurrent.futures import Future

import numpy as np

from semantic_map_rknn.dataset_pipeline import (
    _ALL_TIMING_STAGES,
    _record_stage,
    _segment_prepared_frame,
    _timing_report,
)


def test_timing_report_contains_per_call_and_per_frame_metrics(monkeypatch):
    samples = iter((10.125,))
    monkeypatch.setattr(
        "semantic_map_rknn.dataset_pipeline.time.perf_counter",
        lambda: next(samples),
    )
    totals = {name: 0.0 for name in _ALL_TIMING_STAGES}
    counts = {name: 0 for name in _ALL_TIMING_STAGES}

    elapsed = _record_stage(
        totals, counts, "sam_decoder", 10.0, count=4
    )
    report = _timing_report(
        totals, counts, elapsed=1.0, frame_count=2
    )

    assert elapsed == 0.125
    assert report["accounted_seconds"] == 0.125
    assert report["unaccounted_seconds"] == 0.875
    assert report["overlapped_seconds"] == 0.0
    assert report["stages"]["sam_decoder"] == {
        "total_seconds": 0.125,
        "share_percent": 12.5,
        "calls": 4,
        "avg_ms_per_call": 31.25,
        "avg_ms_per_frame": 62.5,
    }


def test_timing_report_exposes_pipeline_overlap():
    totals = {name: 0.0 for name in _ALL_TIMING_STAGES}
    counts = {name: 0 for name in _ALL_TIMING_STAGES}
    totals["detection"] = 0.8
    totals["sam_encoder"] = 0.7
    counts["detection"] = counts["sam_encoder"] = 1

    report = _timing_report(totals, counts, elapsed=1.0, frame_count=1)

    assert report["accounted_seconds"] == 1.5
    assert report["unaccounted_seconds"] == 0.0
    assert report["overlapped_seconds"] == 0.5


def test_segment_stage_resolves_future_without_running_sam_for_empty_frame():
    prepared = Future()
    prepared.set_result((
        np.zeros((4, 6, 3), dtype=np.uint8),
        np.zeros((4, 6), dtype=np.uint16),
        [],
        0.01,
        0.02,
    ))

    detections, observations, records, timings, counts = (
        _segment_prepared_frame(
            prepared,
            7,
            segmenter=None,
            camera_scale=1000.0,
            intrinsics=None,
            camera_to_map=np.eye(4),
            pixel_stride=2,
            min_depth_m=0.3,
            max_depth_m=5.0,
        )
    )

    assert detections == observations == records == []
    assert timings == {
        "io": 0.01,
        "detection": 0.02,
        "sam_encoder": 0.0,
        "sam_decoder": 0.0,
        "projection": 0.0,
    }
    assert counts == {
        "io": 1,
        "detection": 1,
        "sam_encoder": 0,
        "sam_decoder": 0,
        "projection": 0,
    }
