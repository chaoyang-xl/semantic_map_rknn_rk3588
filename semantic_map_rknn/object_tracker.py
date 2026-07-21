"""Semantic object tracking and point-cloud fusion in the map frame."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .spatial_filter import largest_spatial_cluster_indices


@dataclass(frozen=True)
class ObjectObservation:
    """One projected detection in the common map frame."""

    detection_id: int
    class_id: int
    class_name: str
    confidence: float
    stamp: float
    points: np.ndarray
    fuse_geometry: bool = True
    colors: np.ndarray | None = None

    @property
    def centroid(self) -> np.ndarray:
        return np.mean(self.points, axis=0)


@dataclass
class TrackedObject:
    """Persistent object model built from multiple frame observations."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    first_seen: float
    last_seen: float
    observation_count: int
    points: np.ndarray
    colors: np.ndarray | None = None
    semantic_scores: dict[int, float] = field(default_factory=dict)
    class_names: dict[int, str] = field(default_factory=dict)
    missed_frames: int = 0
    status: str = "candidate"

    @property
    def centroid(self) -> np.ndarray:
        return np.mean(self.points, axis=0)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.min(self.points, axis=0), np.max(self.points, axis=0)


@dataclass(frozen=True)
class Association:
    """Result of assigning one frame observation to one map object."""

    observation_index: int
    track_id: int
    score: float
    geometric_overlap: float
    semantic_similarity: float
    bbox_overlap: float
    is_new: bool = False


def voxel_downsample_with_colors(
    points: np.ndarray,
    colors: np.ndarray | None,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Average XYZ and aligned RGB values independently in each occupied voxel."""
    point_array = np.asarray(points, dtype=np.float32)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    if point_array.shape[0] == 0:
        empty_colors = None if colors is None else np.empty((0, 3), dtype=np.uint8)
        return point_array.copy(), empty_colors

    color_array = None if colors is None else np.asarray(colors, dtype=np.uint8)
    if color_array is not None and color_array.shape != point_array.shape:
        raise ValueError("colors must have the same (N, 3) shape as points")

    keys = np.floor(point_array / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).reshape(-1, 1)
    point_sums = np.zeros((counts.shape[0], 3), dtype=np.float64)
    np.add.at(point_sums, inverse, point_array)
    sampled_points = (point_sums / counts).astype(np.float32)
    if color_array is None:
        return sampled_points, None
    color_sums = np.zeros((counts.shape[0], 3), dtype=np.float64)
    np.add.at(color_sums, inverse, color_array)
    sampled_colors = np.rint(color_sums / counts).clip(0, 255).astype(np.uint8)
    return sampled_points, sampled_colors


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel-downsample XYZ points without colors."""
    return voxel_downsample_with_colors(points, None, voxel_size)[0]


def nearest_neighbor_overlap(
    points_a: np.ndarray, points_b: np.ndarray, radius: float
) -> float:
    """Return the larger bidirectional fraction with a neighbour inside radius."""
    if points_a.size == 0 or points_b.size == 0 or radius <= 0.0:
        return 0.0
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        keys_a = {
            tuple(key)
            for key in np.floor(np.asarray(points_a) / radius).astype(np.int64)
        }
        keys_b = {
            tuple(key)
            for key in np.floor(np.asarray(points_b) / radius).astype(np.int64)
        }
        intersection = len(keys_a & keys_b)
        return max(intersection / len(keys_a), intersection / len(keys_b))

    a = np.asarray(points_a, dtype=np.float32)
    b = np.asarray(points_b, dtype=np.float32)
    distance_a, _ = cKDTree(b).query(a, k=1)
    distance_b, _ = cKDTree(a).query(b, k=1)
    return float(max(np.mean(distance_a <= radius), np.mean(distance_b <= radius)))


def aabb_overlap_ratio(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Return intersection volume divided by the smaller AABB volume."""
    low_a, high_a = np.min(points_a, axis=0), np.max(points_a, axis=0)
    low_b, high_b = np.min(points_b, axis=0), np.max(points_b, axis=0)
    intersection = float(
        np.prod(np.maximum(0.0, np.minimum(high_a, high_b) - np.maximum(low_a, low_b)))
    )
    extent_a = np.maximum(high_a - low_a, 0.02)
    extent_b = np.maximum(high_b - low_b, 0.02)
    denominator = min(float(np.prod(extent_a)), float(np.prod(extent_b)))
    return intersection / denominator if denominator > 0.0 else 0.0


def semantic_similarity(
    observation: ObjectObservation, track: TrackedObject
) -> float:
    """Compare a detection class with the track's confidence-weighted class history."""
    if observation.class_id < 0:
        return 1.0 if observation.class_name == track.class_name else 0.0
    total = sum(track.semantic_scores.values())
    if total <= 0.0:
        return 1.0 if observation.class_id == track.class_id else 0.0
    return float(track.semantic_scores.get(observation.class_id, 0.0) / total)


class ObjectTracker:
    """Associate observations with persistent semantic objects in the map frame."""

    def __init__(
        self,
        voxel_size: float = 0.02,
        overlap_radius: float = 0.04,
        max_centroid_distance_m: float = 1.0,
        min_geometric_overlap: float = 0.05,
        min_bbox_overlap: float = 0.0,
        association_threshold: float = 0.45,
        geometry_weight: float = 0.7,
        semantic_weight: float = 0.3,
        observation_cluster_eps: float = 0.10,
        observation_cluster_min_points: int = 10,
        max_extent_growth: float = 2.0,
        denoise_interval: int = 20,
        map_merge_interval: int = 20,
        map_merge_overlap: float = 0.80,
        min_confirmed_observations: int = 3,
        candidate_max_missed_frames: int = 30,
        stale_after_s: float = 0.0,
    ) -> None:
        if voxel_size <= 0.0 or overlap_radius <= 0.0:
            raise ValueError("voxel_size and overlap_radius must be positive")
        weight_sum = geometry_weight + semantic_weight
        if geometry_weight < 0.0 or semantic_weight < 0.0 or weight_sum <= 0.0:
            raise ValueError("similarity weights must have a positive sum")
        self.voxel_size = float(voxel_size)
        self.overlap_radius = float(overlap_radius)
        self.max_centroid_distance_m = float(max_centroid_distance_m)
        self.min_geometric_overlap = float(min_geometric_overlap)
        self.min_bbox_overlap = float(min_bbox_overlap)
        self.association_threshold = float(association_threshold)
        self.geometry_weight = float(geometry_weight) / weight_sum
        self.semantic_weight = float(semantic_weight) / weight_sum
        self.observation_cluster_eps = max(0.0, float(observation_cluster_eps))
        self.observation_cluster_min_points = max(
            1, int(observation_cluster_min_points)
        )
        self.max_extent_growth = max(1.0, float(max_extent_growth))
        self.denoise_interval = max(0, int(denoise_interval))
        self.map_merge_interval = max(0, int(map_merge_interval))
        self.map_merge_overlap = float(map_merge_overlap)
        self.min_confirmed_observations = max(1, int(min_confirmed_observations))
        self.candidate_max_missed_frames = max(0, int(candidate_max_missed_frames))
        self.stale_after_s = max(0.0, float(stale_after_s))
        self.tracks: dict[int, TrackedObject] = {}
        self._next_track_id = 1
        self._frame_count = 0

    @property
    def confirmed_tracks(self) -> dict[int, TrackedObject]:
        return {
            track_id: track
            for track_id, track in self.tracks.items()
            if track.status == "confirmed"
        }

    def update(self, observations: list[ObjectObservation]) -> list[Association]:
        """Clean, globally assign, fuse, and maintain one frame of observations."""
        self._frame_count += 1
        current_stamp = max((obs.stamp for obs in observations), default=0.0)
        for track in self.tracks.values():
            track.missed_frames += 1

        cleaned = []
        source_indices = []
        for source_index, observation in enumerate(observations):
            item = self._clean_observation(observation)
            if item is not None:
                cleaned.append(item)
                source_indices.append(source_index)
        assignments = self._assign(cleaned)
        matched_observations = {item.observation_index for item in assignments}

        for association in assignments:
            track = self.tracks[association.track_id]
            self._merge_observation(track, cleaned[association.observation_index])

        results = list(assignments)
        for index, observation in enumerate(cleaned):
            if index in matched_observations:
                continue
            track = self._create_track(observation)
            results.append(
                Association(index, track.track_id, 1.0, 1.0, 1.0, 1.0, True)
            )

        self._remove_stale(current_stamp)
        remap = {}
        if self.map_merge_interval and self._frame_count % self.map_merge_interval == 0:
            remap = self._merge_duplicate_tracks()
        if self.denoise_interval and self._frame_count % self.denoise_interval == 0:
            self._denoise_tracks()
        if remap:
            results = [
                replace(item, track_id=remap.get(item.track_id, item.track_id))
                for item in results
            ]
        return sorted(
            (
                replace(
                    item,
                    observation_index=source_indices[item.observation_index],
                )
                for item in results
            ),
            key=lambda item: item.observation_index,
        )

    def finalize(self) -> None:
        """Denoise final models and discard tracks that never became confirmed."""
        self._denoise_tracks()
        self._merge_duplicate_tracks()
        self.tracks = dict(self.confirmed_tracks)

    def _clean_observation(
        self, observation: ObjectObservation
    ) -> ObjectObservation | None:
        if observation.points.size == 0:
            return None
        points, colors = voxel_downsample_with_colors(
            observation.points, observation.colors, self.voxel_size
        )
        keep = largest_spatial_cluster_indices(
            points,
            self.observation_cluster_eps,
            self.observation_cluster_min_points,
        )
        points = points[keep]
        colors = None if colors is None else colors[keep]
        if points.shape[0] < 3:
            return None
        return replace(observation, points=points, colors=colors)

    def _assign(self, observations: list[ObjectObservation]) -> list[Association]:
        if not observations or not self.tracks:
            return []
        track_ids = sorted(self.tracks)
        scores = np.full((len(observations), len(track_ids)), -np.inf, dtype=np.float64)
        details: dict[tuple[int, int], tuple[float, float, float]] = {}
        for obs_index, observation in enumerate(observations):
            for column, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                if (
                    np.linalg.norm(observation.centroid - track.centroid)
                    > self.max_centroid_distance_m
                ):
                    continue
                bbox = aabb_overlap_ratio(observation.points, track.points)
                if bbox < self.min_bbox_overlap:
                    continue
                overlap = nearest_neighbor_overlap(
                    observation.points, track.points, self.overlap_radius
                )
                if overlap < self.min_geometric_overlap:
                    continue
                semantic = semantic_similarity(observation, track)
                score = (
                    self.geometry_weight * overlap
                    + self.semantic_weight * semantic
                )
                if score < self.association_threshold:
                    continue
                if not self._extent_growth_is_valid(track.points, observation.points):
                    continue
                scores[obs_index, column] = score
                details[(obs_index, column)] = (overlap, semantic, bbox)

        finite = np.isfinite(scores)
        if not np.any(finite):
            return []
        try:
            from scipy.optimize import linear_sum_assignment

            cost = np.where(finite, -scores, 1e6)
            row_indices, columns = linear_sum_assignment(cost)
            pairs = [
                (int(row), int(column))
                for row, column in zip(row_indices, columns)
                if finite[row, column]
            ]
        except ImportError:
            candidates = np.argwhere(finite)
            pairs = []
            used_rows: set[int] = set()
            used_columns: set[int] = set()
            for row, column in sorted(
                candidates, key=lambda item: scores[tuple(item)], reverse=True
            ):
                if int(row) in used_rows or int(column) in used_columns:
                    continue
                pairs.append((int(row), int(column)))
                used_rows.add(int(row))
                used_columns.add(int(column))

        return [
            Association(
                row,
                track_ids[column],
                float(scores[row, column]),
                details[(row, column)][0],
                details[(row, column)][1],
                details[(row, column)][2],
            )
            for row, column in pairs
        ]

    def _create_track(self, observation: ObjectObservation) -> TrackedObject:
        score = max(0.0, float(observation.confidence))
        track = TrackedObject(
            track_id=self._next_track_id,
            class_id=observation.class_id,
            class_name=observation.class_name,
            confidence=observation.confidence,
            first_seen=observation.stamp,
            last_seen=observation.stamp,
            observation_count=1,
            points=observation.points.copy(),
            colors=None if observation.colors is None else observation.colors.copy(),
            semantic_scores={observation.class_id: score},
            class_names={observation.class_id: observation.class_name},
        )
        self._update_status(track)
        self.tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    def _merge_observation(
        self, track: TrackedObject, observation: ObjectObservation
    ) -> None:
        old_count = track.observation_count
        track.confidence = (
            track.confidence * old_count + observation.confidence
        ) / (old_count + 1)
        track.observation_count += 1
        track.last_seen = max(track.last_seen, observation.stamp)
        track.missed_frames = 0
        track.semantic_scores[observation.class_id] = (
            track.semantic_scores.get(observation.class_id, 0.0)
            + max(0.0, observation.confidence)
        )
        track.class_names[observation.class_id] = observation.class_name
        self._refresh_class(track)
        if observation.fuse_geometry:
            points = np.concatenate((track.points, observation.points))
            colors = (
                np.concatenate((track.colors, observation.colors))
                if track.colors is not None and observation.colors is not None
                else None
            )
        else:
            points = observation.points
            colors = observation.colors
        track.points, track.colors = voxel_downsample_with_colors(
            points, colors, self.voxel_size
        )
        self._update_status(track)

    def _merge_duplicate_tracks(self) -> dict[int, int]:
        track_ids = sorted(self.tracks)
        parent = {track_id: track_id for track_id in track_ids}

        def find(item: int) -> int:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        def union(first: int, second: int) -> None:
            root_a, root_b = find(first), find(second)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)

        for index, first_id in enumerate(track_ids):
            first = self.tracks[first_id]
            for second_id in track_ids[index + 1 :]:
                second = self.tracks[second_id]
                if first.class_id != second.class_id:
                    continue
                if (
                    np.linalg.norm(first.centroid - second.centroid)
                    > self.max_centroid_distance_m
                ):
                    continue
                overlap = nearest_neighbor_overlap(
                    first.points, second.points, self.overlap_radius
                )
                if overlap >= self.map_merge_overlap:
                    union(first_id, second_id)

        groups: dict[int, list[int]] = {}
        for track_id in track_ids:
            groups.setdefault(find(track_id), []).append(track_id)
        remap: dict[int, int] = {}
        for survivor_id, members in groups.items():
            survivor = self.tracks[survivor_id]
            for merged_id in members:
                remap[merged_id] = survivor_id
                if merged_id == survivor_id:
                    continue
                self._merge_track(survivor, self.tracks.pop(merged_id))
        return remap

    def _merge_track(self, target: TrackedObject, source: TrackedObject) -> None:
        total = target.observation_count + source.observation_count
        target.confidence = (
            target.confidence * target.observation_count
            + source.confidence * source.observation_count
        ) / total
        target.observation_count = total
        target.first_seen = min(target.first_seen, source.first_seen)
        target.last_seen = max(target.last_seen, source.last_seen)
        target.missed_frames = min(target.missed_frames, source.missed_frames)
        for class_id, score in source.semantic_scores.items():
            target.semantic_scores[class_id] = (
                target.semantic_scores.get(class_id, 0.0) + score
            )
        target.class_names.update(source.class_names)
        points = np.concatenate((target.points, source.points))
        colors = (
            np.concatenate((target.colors, source.colors))
            if target.colors is not None and source.colors is not None
            else None
        )
        target.points, target.colors = voxel_downsample_with_colors(
            points, colors, self.voxel_size
        )
        self._refresh_class(target)
        self._update_status(target)

    def _denoise_tracks(self) -> None:
        for track in self.tracks.values():
            keep = largest_spatial_cluster_indices(
                track.points,
                self.observation_cluster_eps,
                self.observation_cluster_min_points,
            )
            track.points = track.points[keep]
            if track.colors is not None:
                track.colors = track.colors[keep]

    def _remove_stale(self, current_stamp: float) -> None:
        remove = []
        for track_id, track in self.tracks.items():
            candidate_expired = (
                track.status == "candidate"
                and track.missed_frames > self.candidate_max_missed_frames
            )
            time_expired = (
                self.stale_after_s > 0.0
                and current_stamp > 0.0
                and current_stamp - track.last_seen > self.stale_after_s
            )
            if candidate_expired or time_expired:
                remove.append(track_id)
        for track_id in remove:
            del self.tracks[track_id]

    def _extent_growth_is_valid(
        self, existing: np.ndarray, observation: np.ndarray
    ) -> bool:
        old_extent = np.maximum(np.ptp(existing, axis=0), self.voxel_size)
        merged_extent = np.ptp(np.concatenate((existing, observation)), axis=0)
        return bool(
            np.all(
                merged_extent
                <= old_extent * self.max_extent_growth + self.overlap_radius
            )
        )

    def _refresh_class(self, track: TrackedObject) -> None:
        track.class_id = max(track.semantic_scores, key=track.semantic_scores.get)
        track.class_name = track.class_names.get(track.class_id, track.class_name)

    def _update_status(self, track: TrackedObject) -> None:
        track.status = (
            "confirmed"
            if track.observation_count >= self.min_confirmed_observations
            else "candidate"
        )
