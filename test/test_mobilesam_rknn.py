import numpy as np

from semantic_map_rknn.mobilesam_rknn import MobileSamRknn


class FakeSession:
    instances = []

    def __init__(self, model_path, **kwargs):
        self.model_path = str(model_path)
        self.backend = "fake"
        self.calls = []
        FakeSession.instances.append(self)

    def inference(self, inputs, data_format=None):
        self.calls.append((inputs, data_format))
        if "encoder" in self.model_path:
            return [np.zeros((1, 256, 28, 28), dtype=np.float32)]
        scores = np.asarray([[0.1, 0.9, 0.2, 0.3]], dtype=np.float32)
        masks = np.full((1, 4, 112, 112), -1.0, dtype=np.float32)
        masks[:, 1, 20:90, 30:80] = 1.0
        return [scores, masks]

    def close(self):
        pass


def test_preprocess_and_official_decoder_postprocess():
    FakeSession.instances.clear()
    model = MobileSamRknn(
        "encoder.rknn",
        "decoder.rknn",
        mask_erode_px=0,
        session_factory=FakeSession,
    )
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    model.set_image(rgb)
    result = model.predict_boxes(np.asarray([[100, 120, 500, 400]], dtype=np.float32))[0]

    encoder_input = FakeSession.instances[0].calls[0][0][0]
    assert encoder_input.shape == (1, 448, 448, 3)
    assert model._resized_shape == (336, 448)
    assert result.mask.shape == (480, 640)
    assert result.mask.dtype == bool
    assert result.mask.any()
    assert result.score == np.float32(0.9)

    decoder_inputs, data_format = FakeSession.instances[1].calls[0]
    np.testing.assert_allclose(
        decoder_inputs[1],
        [[[70.0, 84.0], [350.0, 280.0]]],
    )
    np.testing.assert_array_equal(decoder_inputs[2], [[2.0, 3.0]])
    assert decoder_inputs[0].shape == (1, 28, 28, 256)
    assert decoder_inputs[0].flags.c_contiguous
    assert decoder_inputs[3].shape == (1, 112, 112, 1)
    assert data_format == ["nhwc", "nchw", "nchw", "nhwc", "nchw"]


def test_nhwc_decoder_masks_are_accepted():
    masks = np.zeros((1, 112, 112, 4), dtype=np.float32)
    result = MobileSamRknn._masks_to_nchw(masks, 4)
    assert result.shape == (1, 4, 112, 112)
