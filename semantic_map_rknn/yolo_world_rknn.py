"""YOLO-World RKNN inference compatible with Rockchip model-zoo models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import itertools

import cv2
import numpy as np

from .rknn_runtime import RknnSession


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    def as_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "xyxy": list(self.xyxy),
        }


def load_class_names(path: str | Path) -> list[str]:
    names = [
        line.strip()
        for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not names:
        raise ValueError(f"Class list is empty: {path}")
    if len(names) > 80:
        raise ValueError("The converted YOLO-World RKNN model supports at most 80 prompts")
    return names


class YoloWorldRknn:
    """Run fixed-80-prompt YOLO-World v2s with reusable text embeddings."""

    image_size = 640
    prompt_slots = 80
    embedding_size = 512
    sequence_length = 20
    pad_token_id = 49407

    def __init__(
        self,
        model_path: str | Path,
        class_names: list[str],
        *,
        text_model_path: str | Path | None = None,
        text_embeddings_path: str | Path | None = None,
        tokenizer_path: str = "openai/clip-vit-base-patch32",
        confidence: float = 0.25,
        nms_threshold: float = 0.45,
        backend: str = "auto",
        target: str = "rk3588",
        core_mask: str = "0_1_2",
        session_factory=RknnSession,
    ) -> None:
        if not class_names or len(class_names) > self.prompt_slots:
            raise ValueError("class_names must contain between 1 and 80 prompts")
        self.class_names = list(class_names)
        self.confidence = float(confidence)
        self.nms_threshold = float(nms_threshold)
        self.model = session_factory(
            model_path,
            backend=backend,
            target=target,
            core_mask=core_mask,
        )
        if text_embeddings_path:
            embeddings = np.load(Path(text_embeddings_path).expanduser())
            self.text_embeddings = self._normalize_embeddings(embeddings)
        else:
            if text_model_path is None:
                raise ValueError("text_model_path or text_embeddings_path is required")
            self.text_embeddings = self._build_text_embeddings(
                text_model_path,
                tokenizer_path,
                backend=backend,
                target=target,
                core_mask=core_mask,
                session_factory=session_factory,
            )

    def _build_text_embeddings(
        self,
        model_path: str | Path,
        tokenizer_path: str,
        **session_options,
    ) -> np.ndarray:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Generating YOLO-World text embeddings requires transformers. "
                "Install it or provide a precomputed --text-embeddings file."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        tokens = tokenizer(
            text=list(itertools.chain(self.class_names)),
            return_tensors="np",
            padding=True,
        )["input_ids"]
        text_session = session_options.pop("session_factory")(
            model_path, **session_options
        )
        try:
            outputs = []
            for row in tokens:
                input_ids = np.full(
                    (1, self.sequence_length), self.pad_token_id, dtype=np.float32
                )
                count = min(self.sequence_length, row.size)
                input_ids[0, :count] = row[:count]
                outputs.append(np.asarray(text_session.inference([input_ids])[0]))
        finally:
            text_session.close()
        return self._normalize_embeddings(np.concatenate(outputs, axis=0))

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        array = array.reshape(array.shape[0], -1)
        if array.shape[0] < len(self.class_names) or array.shape[1] != self.embedding_size:
            raise ValueError(
                f"Text embeddings must include at least {len(self.class_names)}x"
                f"{self.embedding_size} values, got {array.shape}"
            )
        padded = np.zeros(
            (1, self.prompt_slots, self.embedding_size), dtype=np.float32
        )
        padded[0, : len(self.class_names)] = array[: len(self.class_names)]
        return padded

    @classmethod
    def letterbox(cls, bgr: np.ndarray):
        height, width = bgr.shape[:2]
        ratio = min(cls.image_size / height, cls.image_size / width)
        new_width = int(round(width * ratio))
        new_height = int(round(height * ratio))
        resized = cv2.resize(bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (cls.image_size - new_width) / 2.0
        pad_y = (cls.image_size - new_height) / 2.0
        left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
        top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        return rgb[None].astype(np.float32), ratio, (pad_x, pad_y)

    def predict(self, bgr: np.ndarray) -> list[Detection]:
        image = np.asarray(bgr)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("YOLO-World input must have shape (H, W, 3)")
        model_input, ratio, padding = self.letterbox(image)
        outputs = self.model.inference([model_input, self.text_embeddings])
        boxes, classes, scores = self._postprocess(outputs)
        if boxes.size == 0:
            return []
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - padding[0]) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - padding[1]) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image.shape[1] - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image.shape[0] - 1)
        detections = []
        for box, class_id, score in zip(boxes, classes, scores):
            class_id = int(class_id)
            if class_id >= len(self.class_names):
                continue
            detections.append(Detection(
                class_id=class_id,
                class_name=self.class_names[class_id],
                confidence=float(score),
                xyxy=tuple(float(value) for value in box),
            ))
        return detections

    def _postprocess(self, outputs: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(outputs) < 6 or len(outputs) % 3 != 0:
            raise RuntimeError(f"Unexpected YOLO-World output count: {len(outputs)}")
        boxes_all, probabilities_all = [], []
        pair_per_branch = len(outputs) // 3
        for branch in range(3):
            probabilities = self._as_nchw(
                outputs[pair_per_branch * branch], expected_channels=self.prompt_slots
            )
            positions = self._as_nchw(
                outputs[pair_per_branch * branch + 1], expected_channels=4
            )
            boxes_all.append(self._decode_boxes(positions))
            probabilities_all.append(self._flatten(probabilities))
        boxes = np.concatenate(boxes_all)
        probabilities = np.concatenate(probabilities_all)
        classes = np.argmax(probabilities, axis=1)
        scores = np.max(probabilities, axis=1)
        valid = (scores >= self.confidence) & (classes < len(self.class_names))
        boxes, classes, scores = boxes[valid], classes[valid], scores[valid]
        kept = []
        for class_id in np.unique(classes):
            indices = np.flatnonzero(classes == class_id)
            local = self._nms(boxes[indices], scores[indices])
            kept.extend(indices[local].tolist())
        if not kept:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )
        kept = np.asarray(kept, dtype=np.int64)
        order = np.argsort(scores[kept])[::-1]
        kept = kept[order]
        return boxes[kept], classes[kept], scores[kept]

    @staticmethod
    def _as_nchw(value: np.ndarray, expected_channels: int) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 4:
            raise RuntimeError(f"Expected a 4D YOLO output, got {array.shape}")
        if array.shape[1] == expected_channels:
            return array
        if array.shape[-1] == expected_channels:
            return array.transpose(0, 3, 1, 2)
        raise RuntimeError(
            f"Cannot find {expected_channels} channels in YOLO output {array.shape}"
        )

    @classmethod
    def _decode_boxes(cls, position: np.ndarray) -> np.ndarray:
        grid_height, grid_width = position.shape[2:4]
        columns, rows = np.meshgrid(np.arange(grid_width), np.arange(grid_height))
        grid = np.stack((columns, rows), axis=0)[None].astype(np.float32)
        stride = np.asarray(
            [cls.image_size // grid_width, cls.image_size // grid_height],
            dtype=np.float32,
        ).reshape(1, 2, 1, 1)
        top_left = (grid + 0.5 - position[:, :2]) * stride
        bottom_right = (grid + 0.5 + position[:, 2:4]) * stride
        return cls._flatten(np.concatenate((top_left, bottom_right), axis=1))

    @staticmethod
    def _flatten(value: np.ndarray) -> np.ndarray:
        return value.transpose(0, 2, 3, 1).reshape(-1, value.shape[1])

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        if boxes.size == 0:
            return np.empty((0,), dtype=np.int64)
        x1, y1, x2, y2 = boxes.T
        areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size:
            index = int(order[0])
            keep.append(index)
            xx1 = np.maximum(x1[index], x1[order[1:]])
            yy1 = np.maximum(y1[index], y1[order[1:]])
            xx2 = np.minimum(x2[index], x2[order[1:]])
            yy2 = np.minimum(y2[index], y2[order[1:]])
            intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            union = areas[index] + areas[order[1:]] - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            order = order[np.flatnonzero(iou <= self.nms_threshold) + 1]
        return np.asarray(keep, dtype=np.int64)

    def close(self) -> None:
        self.model.close()

    def __enter__(self) -> "YoloWorldRknn":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
