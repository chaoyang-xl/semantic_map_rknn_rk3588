"""Spatial filtering for one RGB-D object observation.

The functions in this module operate only on one projected detection. They do
not perform tracking or modify object identity, so projection nodes and fusion
algorithms can use them independently.
"""

from __future__ import annotations

import numpy as np


def largest_spatial_cluster_indices(
    points: np.ndarray,
    eps: float,
    min_points: int,
    max_neighbors: int = 32,
) -> np.ndarray:
    """Return indices of the largest bounded-kNN DBSCAN-style cluster.

    Noise and disconnected background surfaces are removed. If clustering is
    disabled, SciPy is unavailable, or no valid cluster is found, all points
    are returned so an optional denoiser cannot erase a valid observation.

    A full radius-neighbour list becomes very expensive for dense RGB-D
    surfaces because every point may materialize hundreds of Python objects.
    The bounded kNN graph retains local connectivity while keeping memory and
    runtime proportional to ``N * max_neighbors``.
    """
    point_array = np.asarray(points, dtype=np.float32)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    count = point_array.shape[0]
    minimum = max(1, int(min_points))
    if count < minimum or eps <= 0.0:
        return np.arange(count, dtype=np.int64)

    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree
    except ImportError:
        return np.arange(count, dtype=np.int64)

    neighbor_count = min(
        count,
        max(minimum, max(1, int(max_neighbors))),
    )
    _, neighbours = cKDTree(point_array).query(
        point_array,
        k=neighbor_count,
        distance_upper_bound=float(eps),
    )
    if neighbor_count == 1:
        neighbours = neighbours.reshape(-1, 1)
    valid = neighbours < count
    core = np.count_nonzero(valid, axis=1) >= minimum
    core_indices = np.flatnonzero(core)
    if core_indices.size == 0:
        return np.arange(count, dtype=np.int64)

    core_lookup = np.full(count + 1, -1, dtype=np.int64)
    core_lookup[core_indices] = np.arange(core_indices.size, dtype=np.int64)
    core_neighbours = core_lookup[neighbours]
    edge_mask = core_neighbours[core_indices] >= 0
    rows = np.repeat(
        np.arange(core_indices.size, dtype=np.int64),
        np.count_nonzero(edge_mask, axis=1),
    )
    columns = core_neighbours[core_indices][edge_mask]
    graph = csr_matrix(
        (np.ones(rows.size, dtype=np.uint8), (rows, columns)),
        shape=(core_indices.size, core_indices.size),
    )
    _, core_labels = connected_components(
        graph, directed=False, return_labels=True
    )

    point_labels = np.full(count, -1, dtype=np.int64)
    point_labels[core_indices] = core_labels
    has_core_neighbour = np.any(core_neighbours >= 0, axis=1)
    border_indices = np.flatnonzero(~core & has_core_neighbour)
    if border_indices.size:
        first_core_column = np.argmax(
            core_neighbours[border_indices] >= 0,
            axis=1,
        )
        nearest_core = core_neighbours[
            border_indices,
            first_core_column,
        ]
        point_labels[border_indices] = core_labels[nearest_core]

    labelled = point_labels >= 0
    if not np.any(labelled):
        return np.arange(count, dtype=np.int64)
    largest_label = int(np.argmax(np.bincount(point_labels[labelled])))
    largest = np.flatnonzero(point_labels == largest_label)
    return largest if largest.size >= 5 else np.arange(count, dtype=np.int64)
