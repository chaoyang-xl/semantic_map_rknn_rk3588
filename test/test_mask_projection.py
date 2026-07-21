"""Tests for instance-mask depth projection."""

import numpy as np
import pytest

from semantic_map_rknn.bbox_projection import CameraIntrinsics
from semantic_map_rknn.mask_projection import project_mask_depth


def test_only_masked_valid_pixels_are_projected():
    depth = np.full((4, 5), 2.0, dtype=np.float32)
    depth[1, 2] = 0.0
    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 1:4] = True
    result = project_mask_depth(
        depth, mask, CameraIntrinsics(100.0, 100.0, 2.0, 1.0), pixel_stride=1
    )
    assert result is not None
    assert result.points_camera.shape == (2, 3)
    assert np.allclose(result.image_uv, [[1, 1], [3, 1]])
    assert np.allclose(result.points_camera[:, 2], 2.0)


def test_stride_downsamples_mask_pixels():
    result = project_mask_depth(
        np.ones((6, 6), dtype=np.float32),
        np.ones((6, 6), dtype=bool),
        CameraIntrinsics(10.0, 10.0, 3.0, 3.0),
        pixel_stride=2,
    )
    assert result is not None
    assert result.points_camera.shape[0] == 9


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        project_mask_depth(
            np.ones((4, 4)), np.ones((3, 4), dtype=bool),
            CameraIntrinsics(10.0, 10.0, 2.0, 2.0),
        )
