from semantic_map_rknn.dataset_pipeline import (
    _ALL_TIMING_STAGES,
    _record_stage,
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
    assert report["stages"]["sam_decoder"] == {
        "total_seconds": 0.125,
        "share_percent": 12.5,
        "calls": 4,
        "avg_ms_per_call": 31.25,
        "avg_ms_per_frame": 62.5,
    }
