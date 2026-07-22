"""Offline RKNN MobileSAM projection and semantic object tracking."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .mask_projection import CameraIntrinsics, project_mask_depth
from .mobilesam_rknn import MobileSamRknn
from .object_map_io import object_record, write_semantic_object_map
from .object_tracker import ObjectObservation, ObjectTracker
from .point_cloud_io import save_object_ply
from .yolo_world_rknn import YoloWorldRknn, load_class_names


_LOOP_TIMING_STAGES = (
    "io",
    "detection",
    "sam_encoder",
    "sam_decoder",
    "projection",
    "fusion",
)
_ALL_TIMING_STAGES = _LOOP_TIMING_STAGES + ("finalize", "save")


def _accumulate_stage(
    totals: dict[str, float],
    counts: dict[str, int],
    name: str,
    elapsed: float,
    *,
    count: int = 1,
) -> float:
    """Accumulate work measured inside an optional pipeline stage."""
    totals[name] += elapsed
    counts[name] += int(count)
    return elapsed


def _record_stage(
    totals: dict[str, float],
    counts: dict[str, int],
    name: str,
    started: float,
    *,
    count: int = 1,
) -> float:
    """Accumulate one measured stage and return its elapsed seconds."""
    elapsed = time.perf_counter() - started
    totals[name] += elapsed
    counts[name] += int(count)
    return elapsed


def _timing_report(
    totals: dict[str, float],
    counts: dict[str, int],
    *,
    elapsed: float,
    frame_count: int,
) -> dict:
    """Build a compact, machine-readable profiling report."""
    accounted = sum(totals.values())
    stages = {}
    for name in _ALL_TIMING_STAGES:
        total = totals[name]
        calls = counts[name]
        stages[name] = {
            "total_seconds": round(total, 6),
            "share_percent": round(100.0 * total / elapsed, 2) if elapsed else 0.0,
            "calls": calls,
            "avg_ms_per_call": round(1000.0 * total / calls, 3) if calls else 0.0,
            "avg_ms_per_frame": round(1000.0 * total / frame_count, 3)
            if frame_count else 0.0,
        }
    return {
        "scope": "dataset loop and final output; model initialization excluded",
        "elapsed_seconds": round(elapsed, 6),
        "accounted_seconds": round(accounted, 6),
        "unaccounted_seconds": round(max(0.0, elapsed - accounted), 6),
        "overlapped_seconds": round(max(0.0, accounted - elapsed), 6),
        "stages": stages,
    }


def _load_detections(path: Path, confidence: float) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for item in payload.get("detections", []):
        class_name = str(item.get("class_name", "unknown"))
        score = float(item.get("confidence", 0.0))
        xyxy = item.get("xyxy", [])
        if score < confidence or class_name.casefold() == "person" or len(xyxy) != 4:
            continue
        output.append({
            "class_id": int(item.get("class_id", -1)),
            "class_name": class_name,
            "confidence": score,
            "xyxy": [float(value) for value in xyxy],
        })
    return output


def _transform(points: np.ndarray, camera_to_map: np.ndarray) -> np.ndarray:
    return (
        points.astype(np.float64) @ camera_to_map[:3, :3].T
        + camera_to_map[:3, 3]
    ).astype(np.float32)


def _prepare_frame(
    data_root: Path,
    frame_id: int,
    detection_path: Path,
    *,
    use_exported: bool,
    detector,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray, list[dict], float, float]:
    """Load RGB-D and run YOLO, suitable for one-frame NPU prefetch."""
    started = time.perf_counter()
    image = cv2.imread(str(data_root / "results" / f"frame{frame_id:06d}.jpg"))
    depth = cv2.imread(
        str(data_root / "results" / f"depth{frame_id:06d}.png"),
        cv2.IMREAD_UNCHANGED,
    )
    io_elapsed = time.perf_counter() - started
    if image is None or depth is None:
        raise FileNotFoundError(f"Unable to load frame {frame_id}")

    started = time.perf_counter()
    if use_exported:
        detections = _load_detections(detection_path, confidence)
    else:
        detections = [item.as_dict() for item in detector.predict(image)]
        detections = [
            item
            for item in detections
            if item["class_name"].casefold() != "person"
        ]
    detection_elapsed = time.perf_counter() - started
    return image, depth, detections, io_elapsed, detection_elapsed


def save_tracks(
    output: Path, tracker: ObjectTracker, *, confirmed_only: bool = False
) -> list[dict]:
    objects_directory = output / "objects"
    objects_directory.mkdir(parents=True, exist_ok=True)
    records = []
    tracks = sorted(tracker.tracks.values(), key=lambda item: item.track_id)
    if confirmed_only:
        tracks = [track for track in tracks if track.status == "confirmed"]
    for track in tracks:
        safe_name = "".join(
            char if char.isalnum() or char in ("-", "_") else "_"
            for char in track.class_name
        )
        stem = f"object_{track.track_id:04d}_{safe_name}"
        arrays = {
            "points_map": track.points,
            "track_id": np.asarray(track.track_id, dtype=np.int32),
            "class_id": np.asarray(track.class_id, dtype=np.int32),
            "class_name": np.asarray(track.class_name),
            "confidence": np.asarray(track.confidence, dtype=np.float32),
            "observation_count": np.asarray(track.observation_count, dtype=np.int32),
            "first_seen": np.asarray(track.first_seen, dtype=np.float64),
            "last_seen": np.asarray(track.last_seen, dtype=np.float64),
            "status": np.asarray(track.status),
            "semantic_scores": np.asarray(json.dumps(track.semantic_scores)),
        }
        if track.colors is not None:
            arrays["rgb"] = track.colors
        np.savez_compressed(objects_directory / f"{stem}.npz", **arrays)
        save_object_ply(objects_directory / f"{stem}.ply", track.points, track.colors)
        records.append(object_record(
            track_id=track.track_id,
            class_id=track.class_id,
            class_name=track.class_name,
            confidence=track.confidence,
            observation_count=track.observation_count,
            first_seen=track.first_seen,
            last_seen=track.last_seen,
            points=track.points,
            ply_path=f"objects/{stem}.ply",
            npz_path=f"objects/{stem}.npz",
            source="rknn_mobilesam_object_tracking",
            semantic_scores=track.semantic_scores,
            status=track.status,
            missed_frames=track.missed_frames,
        ))
    return records


def run_dataset(args) -> dict:
    data_root = Path(args.data_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    camera = json.loads((data_root / "cam_params.json").read_text(encoding="utf-8"))["camera"]
    poses = np.loadtxt(data_root / "traj.txt", dtype=np.float64).reshape(-1, 4, 4)
    intrinsics = CameraIntrinsics(
        float(camera["fx"]), float(camera["fy"]),
        float(camera["cx"]), float(camera["cy"]),
    )
    frame_step = int(getattr(args, "frame_step", 1))
    if frame_step < 1:
        raise ValueError("frame_step must be at least 1")
    frame_ids = []
    for index in range(args.start, len(poses), frame_step):
        if not (data_root / "results" / f"frame{index:06d}.jpg").is_file():
            break
        if not (data_root / "results" / f"depth{index:06d}.png").is_file():
            break
        frame_ids.append(index)
        if args.frames and len(frame_ids) >= args.frames:
            break
    if not frame_ids:
        raise FileNotFoundError("No contiguous RGB-D frames found")

    detection_paths = [data_root / "detections" / f"{index:06d}.json" for index in frame_ids]
    use_exported = all(path.is_file() for path in detection_paths)
    detector = None
    if not use_exported:
        required = (args.yolo_model, args.classes_path)
        if any(value is None for value in required):
            raise ValueError("Missing detections require --yolo-model and --classes-path")
        detector = YoloWorldRknn(
            args.yolo_model,
            load_class_names(args.classes_path),
            text_model_path=args.clip_text_model,
            text_embeddings_path=args.text_embeddings,
            tokenizer_path=args.tokenizer_path,
            confidence=args.confidence,
            nms_threshold=args.nms_threshold,
            backend=args.rknn_backend,
            target=args.rknn_target,
            core_mask=args.yolo_core,
        )
    segmenter = MobileSamRknn(
        args.sam_encoder,
        args.sam_decoder,
        backend=args.rknn_backend,
        target=args.rknn_target,
        encoder_core=args.sam_encoder_core,
        decoder_core=args.sam_decoder_core,
        mask_threshold=args.mask_threshold,
        mask_erode_px=args.mask_erode_px,
    )
    tracker = ObjectTracker(
        voxel_size=args.voxel_size,
        overlap_radius=args.overlap_radius,
        max_centroid_distance_m=args.max_centroid_distance_m,
        min_geometric_overlap=args.min_geometric_overlap,
        association_threshold=args.association_threshold,
        geometry_weight=args.geometry_weight,
        semantic_weight=args.semantic_weight,
        observation_cluster_eps=args.observation_cluster_eps,
        observation_cluster_min_points=args.observation_cluster_min_points,
        max_extent_growth=args.max_extent_growth,
        denoise_interval=args.denoise_interval,
        map_merge_interval=args.map_merge_interval,
        min_confirmed_observations=args.min_confirmed_observations,
        candidate_max_missed_frames=args.candidate_max_missed_frames,
    )
    association_records = []
    input_detections = projected_detections = 0
    stage_totals = {name: 0.0 for name in _ALL_TIMING_STAGES}
    stage_counts = {name: 0 for name in _ALL_TIMING_STAGES}
    pipeline_prefetch = bool(getattr(args, "pipeline_prefetch", False))
    executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="rknn-yolo-prefetch")
        if pipeline_prefetch
        else None
    )

    def submit_frame(position: int):
        frame_id = frame_ids[position]
        arguments = (data_root, frame_id, detection_paths[position])
        keywords = {
            "use_exported": use_exported,
            "detector": detector,
            "confidence": args.confidence,
        }
        if executor is None:
            return _prepare_frame(*arguments, **keywords)
        return executor.submit(_prepare_frame, *arguments, **keywords)

    started = time.perf_counter()
    pending = submit_frame(0) if executor is not None else None
    try:
        for processed, frame_id in enumerate(frame_ids, start=1):
            frame_started = time.perf_counter()
            frame_stages = {name: 0.0 for name in _LOOP_TIMING_STAGES}
            if pending is None:
                prepared = submit_frame(processed - 1)
            else:
                prepared = pending.result()
                pending = (
                    submit_frame(processed)
                    if processed < len(frame_ids)
                    else None
                )
            image, depth, detections, io_elapsed, detection_elapsed = prepared
            frame_stages["io"] = _accumulate_stage(
                stage_totals, stage_counts, "io", io_elapsed
            )
            frame_stages["detection"] = _accumulate_stage(
                stage_totals, stage_counts, "detection", detection_elapsed
            )
            input_detections += len(detections)
            observations, frame_records = [], []
            if detections:
                stage_started = time.perf_counter()
                segmenter.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                frame_stages["sam_encoder"] = _record_stage(
                    stage_totals, stage_counts, "sam_encoder", stage_started
                )
                boxes = np.asarray([item["xyxy"] for item in detections], dtype=np.float32)
                stage_started = time.perf_counter()
                mask_results = segmenter.predict_boxes(boxes)
                frame_stages["sam_decoder"] = _record_stage(
                    stage_totals,
                    stage_counts,
                    "sam_decoder",
                    stage_started,
                    count=len(boxes),
                )
                stage_started = time.perf_counter()
                depth_m = depth.astype(np.float32) / float(camera.get("scale", 1000.0))
                for detection_id, (detection, sam_result) in enumerate(zip(detections, mask_results)):
                    mask = sam_result.mask
                    if mask.shape != depth_m.shape:
                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (depth_m.shape[1], depth_m.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    projection = project_mask_depth(
                        depth_m,
                        mask,
                        intrinsics,
                        pixel_stride=args.pixel_stride,
                        min_depth_m=args.min_depth,
                        max_depth_m=args.max_depth,
                    )
                    if projection is None:
                        continue
                    points_map = _transform(projection.points_camera, poses[frame_id])
                    uv = projection.image_uv
                    rgb_x = np.clip(
                        np.rint((uv[:, 0] + 0.5) * image.shape[1] / depth.shape[1] - 0.5),
                        0, image.shape[1] - 1,
                    ).astype(np.int64)
                    rgb_y = np.clip(
                        np.rint((uv[:, 1] + 0.5) * image.shape[0] / depth.shape[0] - 0.5),
                        0, image.shape[0] - 1,
                    ).astype(np.int64)
                    colors = image[rgb_y, rgb_x, ::-1].copy()
                    observations.append(ObjectObservation(
                        detection_id=detection_id,
                        class_id=int(detection["class_id"]),
                        class_name=str(detection["class_name"]),
                        confidence=float(detection["confidence"]),
                        stamp=float(frame_id),
                        points=points_map,
                        colors=colors,
                    ))
                    frame_records.append({
                        "frame_id": frame_id,
                        "detection_id": detection_id,
                        **detection,
                        "sam_score": sam_result.score,
                        "mask_area_pixels": int(np.count_nonzero(mask)),
                        "forward_point_count": int(points_map.shape[0]),
                    })
                frame_stages["projection"] = _record_stage(
                    stage_totals,
                    stage_counts,
                    "projection",
                    stage_started,
                    count=len(mask_results),
                )
            stage_started = time.perf_counter()
            assignments = tracker.update(observations)
            for assignment in assignments:
                record = frame_records[assignment.observation_index]
                record.update({
                    "track_id": assignment.track_id,
                    "association_score": assignment.score,
                    "geometric_overlap": assignment.geometric_overlap,
                    "semantic_similarity": assignment.semantic_similarity,
                    "is_new_track": assignment.is_new,
                })
            frame_stages["fusion"] = _record_stage(
                stage_totals, stage_counts, "fusion", stage_started
            )
            projected_detections += len(frame_records)
            association_records.extend(frame_records)
            if processed == 1 or processed == len(frame_ids) or (
                args.progress_every and processed % args.progress_every == 0
            ):
                elapsed = time.perf_counter() - started
                eta = elapsed / processed * (len(frame_ids) - processed)
                stage_text = " ".join(
                    f"{name}={frame_stages[name] * 1000.0:.1f}"
                    for name in _LOOP_TIMING_STAGES
                )
                print(
                    f"[{processed}/{len(frame_ids)}] frame={frame_id} "
                    f"det={len(detections)} projected={len(frame_records)} "
                    f"tracks={len(tracker.tracks)} ETA={time.strftime('%H:%M:%S', time.gmtime(eta))} "
                    f"frame_s={time.perf_counter() - frame_started:.3f} "
                    f"stage_ms[{stage_text}]",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        segmenter.close()
        if detector is not None:
            detector.close()

    stage_started = time.perf_counter()
    tracker.finalize()
    _record_stage(stage_totals, stage_counts, "finalize", stage_started)

    stage_started = time.perf_counter()
    object_records = save_tracks(output, tracker)
    write_semantic_object_map(
        output / "semantic_objects.json",
        object_records,
        frame_id="map",
        source="semantic_map_rknn",
        metadata={
            "frame_start": args.start,
            "frame_count": len(frame_ids),
            "frame_step": frame_step,
            "frame_last": frame_ids[-1],
            "pipeline_prefetch": pipeline_prefetch,
            "npu_core_masks": {
                "yolo": args.yolo_core,
                "sam_encoder": args.sam_encoder_core,
                "sam_decoder": args.sam_decoder_core,
            },
        },
    )
    (output / "associations.json").write_text(
        json.dumps(association_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _record_stage(stage_totals, stage_counts, "save", stage_started)
    elapsed = time.perf_counter() - started
    timing = _timing_report(
        stage_totals,
        stage_counts,
        elapsed=elapsed,
        frame_count=len(frame_ids),
    )
    summary = {
        "frame_start": args.start,
        "frame_count": len(frame_ids),
        "frame_step": frame_step,
        "frame_last": frame_ids[-1],
        "pipeline_prefetch": pipeline_prefetch,
        "npu_core_masks": {
            "yolo": args.yolo_core,
            "sam_encoder": args.sam_encoder_core,
            "sam_decoder": args.sam_decoder_core,
        },
        "detection_source": "exported_json" if use_exported else "rknn_yolo_world",
        "input_detections": input_detections,
        "projected_detections": projected_detections,
        "objects": len(object_records),
        "projection_mode": "rknn_mobilesam",
        "depth_range_m": [args.min_depth, args.max_depth],
        "elapsed_seconds": round(elapsed, 3),
        "timing": timing,
    }
    (output / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary
