import numpy as np
import pytest

from semantic_map_rknn.top_down_projection import (
    make_top_down_layout,
    xy_bounds,
)


def test_world_to_pixel_preserves_scale_and_uses_upward_map_y():
    points = np.asarray([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    layout = make_top_down_layout(
        [points], pixels_per_meter=100.0, margin_m=0.0,
        min_canvas_px=1, max_canvas_px=1000,
    )
    pixels = layout.world_to_pixel(points[:, :2])

    assert pixels[1, 0] - pixels[0, 0] == 200
    assert pixels[0, 1] - pixels[1, 1] == 100


def test_layout_reduces_scale_to_respect_max_canvas():
    points = np.asarray([[0.0, 0.0], [100.0, 20.0]])
    layout = make_top_down_layout(
        [points], pixels_per_meter=100.0, margin_m=0.0,
        min_canvas_px=100, max_canvas_px=1000,
    )

    assert layout.width == 1000
    assert layout.height >= 100
    assert layout.pixels_per_meter == pytest.approx(9.99)


def test_xy_bounds_ignores_nonfinite_rows():
    bounds = xy_bounds(
        np.asarray([[1.0, 2.0, 5.0], [3.0, -1.0, 7.0], [np.nan, 4.0, 1.0]])
    )

    np.testing.assert_allclose(bounds.minimum, [1.0, -1.0])
    np.testing.assert_allclose(bounds.maximum, [3.0, 2.0])
    np.testing.assert_allclose(bounds.center, [2.0, 0.5])
    np.testing.assert_allclose(bounds.size, [2.0, 3.0])


def test_layout_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        make_top_down_layout([])
