"""Spatial filtering for one RGB-D object observation.

The functions in this module operate only on one projected detection. They do
not perform tracking or modify object identity, so projection nodes and fusion
algorithms can use them independently.
"""

from __future__ import annotations

import numpy as np


def largest_spatial_cluster_indices(
    points: np.ndarray, eps: float, min_points: int
) -> np.ndarray:
    """Return indices of the largest DBSCAN-style spatial cluster.

    Noise and disconnected background surfaces are removed. If clustering is
    disabled, SciPy is unavailable, or no valid cluster is found, all points
    are returned so an optional denoiser cannot erase a valid observation.
    """
    point_array = np.asarray(points, dtype=np.float32)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    count = point_array.shape[0]
    minimum = max(1, int(min_points))
    if count < minimum or eps <= 0.0:
        return np.arange(count, dtype=np.int64)

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return np.arange(count, dtype=np.int64)

    neighbours = cKDTree(point_array).query_ball_point(point_array, float(eps))
    core = np.asarray([len(items) >= minimum for items in neighbours], dtype=bool)
    visited = np.zeros(count, dtype=bool)
    clusters: list[np.ndarray] = []

    for seed in np.flatnonzero(core):
        if visited[seed]:
            continue
        stack = [int(seed)]
        visited[seed] = True
        members: set[int] = set()
        while stack:
            index = stack.pop()
            members.update(neighbours[index])
            for neighbour in neighbours[index]:
                if core[neighbour] and not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(int(neighbour))
        clusters.append(np.asarray(sorted(members), dtype=np.int64))

    if not clusters:
        return np.arange(count, dtype=np.int64)
    largest = max(clusters, key=lambda indices: indices.size)
    return largest if largest.size >= 5 else np.arange(count, dtype=np.int64)
