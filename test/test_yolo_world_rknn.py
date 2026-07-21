from pathlib import Path

import numpy as np
import pytest

from semantic_map_rknn.yolo_world_rknn import YoloWorldRknn, load_class_names


class FakeYoloSession:
    def __init__(self, model_path, **kwargs):
        self.model_path = Path(model_path)
        self.backend = "fake"

    def inference(self, inputs, data_format=None):
        outputs = []
        for _ in range(3):
            probabilities = np.zeros((1, 80, 1, 1), dtype=np.float32)
            probabilities[0, 0, 0, 0] = 0.9
            positions = np.full((1, 4, 1, 1), 0.25, dtype=np.float32)
            outputs.extend([probabilities, positions, np.zeros((1, 1, 1, 1))])
        return outputs

    def close(self):
        pass


def test_yolo_letterbox_postprocess_returns_original_coordinates(tmp_path):
    embeddings = tmp_path / "text.npy"
    np.save(embeddings, np.ones((1, 80, 512), dtype=np.float32))
    detector = YoloWorldRknn(
        "yolo.rknn",
        ["chair"],
        text_embeddings_path=embeddings,
        confidence=0.5,
        session_factory=FakeYoloSession,
    )
    detections = detector.predict(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(detections) == 1
    assert detections[0].class_name == "chair"
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[0].xyxy == pytest.approx((160.0, 80.0, 480.0, 400.0))


def test_class_file_enforces_fixed_80_prompt_limit(tmp_path):
    class_file = tmp_path / "classes.txt"
    class_file.write_text("\n".join(f"class_{index}" for index in range(81)))
    with pytest.raises(ValueError, match="at most 80"):
        load_class_names(class_file)
