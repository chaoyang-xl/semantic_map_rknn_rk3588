"""Rockchip model-zoo MobileSAM encoder/decoder inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .rknn_runtime import RknnSession


@dataclass(frozen=True)
class SamMaskResult:
    mask: np.ndarray
    score: float


class MobileSamRknn:
    """Run the official 448-pixel RKNN MobileSAM models with box prompts."""

    image_size = 448
    low_res_size = 112

    def __init__(
        self,
        encoder_model: str | Path,
        decoder_model: str | Path,
        *,
        backend: str = "auto",
        target: str = "rk3588",
        encoder_core: str = "0_1_2",
        decoder_core: str = "0_1_2",
        mask_threshold: float = 0.0,
        mask_erode_px: int = 2,
        session_factory=RknnSession,
    ) -> None:
        options = {"backend": backend, "target": target}
        self.encoder = session_factory(encoder_model, core_mask=encoder_core, **options)
        self.decoder = session_factory(decoder_model, core_mask=decoder_core, **options)
        self.mask_threshold = float(mask_threshold)
        self.mask_erode_px = max(0, int(mask_erode_px))
        self._embedding: np.ndarray | None = None
        self._original_shape: tuple[int, int] | None = None
        self._resized_shape: tuple[int, int] | None = None

    @classmethod
    def resized_shape(cls, height: int, width: int) -> tuple[int, int]:
        scale = cls.image_size / float(max(height, width))
        return int(height * scale + 0.5), int(width * scale + 0.5)

    @classmethod
    def preprocess_image(cls, rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("MobileSAM input must have shape (H, W, 3)")
        height, width = image.shape[:2]
        new_height, new_width = cls.resized_shape(height, width)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(
            resized,
            0,
            cls.image_size - new_height,
            0,
            cls.image_size - new_width,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        # Mean/std normalization is embedded by the official encoder converter.
        return padded[None].astype(np.float32), (new_height, new_width)

    def set_image(self, rgb: np.ndarray) -> None:
        encoder_input, resized_shape = self.preprocess_image(rgb)
        outputs = self.encoder.inference([encoder_input])
        if not outputs:
            raise RuntimeError("MobileSAM encoder returned no output")
        self._embedding = np.asarray(outputs[0])
        self._original_shape = tuple(int(value) for value in rgb.shape[:2])
        self._resized_shape = resized_shape

    def predict_boxes(self, boxes_xyxy: np.ndarray) -> list[SamMaskResult]:
        if self._embedding is None or self._original_shape is None:
            raise RuntimeError("set_image() must be called before predict_boxes()")
        boxes = np.asarray(boxes_xyxy, dtype=np.float32)
        if boxes.size == 0:
            return []
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes_xyxy must have shape (N, 4)")
        height, width = self._original_shape
        results = []
        for box in boxes:
            clipped = box.copy()
            clipped[[0, 2]] = np.clip(clipped[[0, 2]], 0, max(0, width - 1))
            clipped[[1, 3]] = np.clip(clipped[[1, 3]], 0, max(0, height - 1))
            coords = np.asarray(
                [[[clipped[0], clipped[1]], [clipped[2], clipped[3]]]],
                dtype=np.float32,
            )
            coords[..., 0] *= self._resized_shape[1] / float(width)
            coords[..., 1] *= self._resized_shape[0] / float(height)
            labels = np.asarray([[2.0, 3.0]], dtype=np.float32)
            mask_input = np.zeros(
                (1, 1, self.low_res_size, self.low_res_size), dtype=np.float32
            )
            has_mask = np.zeros((1,), dtype=np.float32)
            outputs = self.decoder.inference(
                [self._embedding, coords, labels, mask_input, has_mask],
                data_format="NCHW",
            )
            scores, low_res_masks = self._identify_decoder_outputs(outputs)
            mask, score = self._postprocess(scores, low_res_masks)
            results.append(SamMaskResult(mask=mask, score=score))
        return results

    @staticmethod
    def _identify_decoder_outputs(outputs: list) -> tuple[np.ndarray, np.ndarray]:
        if len(outputs) != 2:
            raise RuntimeError(f"MobileSAM decoder expected 2 outputs, received {len(outputs)}")
        arrays = [np.asarray(value) for value in outputs]
        score_index = next((i for i, value in enumerate(arrays) if value.size <= 16), None)
        if score_index is None:
            raise RuntimeError(f"Cannot identify decoder outputs: {[a.shape for a in arrays]}")
        return arrays[score_index], arrays[1 - score_index]

    def _postprocess(
        self, scores: np.ndarray, low_res_masks: np.ndarray
    ) -> tuple[np.ndarray, float]:
        score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
        masks = np.asarray(low_res_masks, dtype=np.float32)
        masks = self._masks_to_nchw(masks, score_values.size)
        index = int(np.argmax(score_values[: masks.shape[1]]))
        logits = masks[0, index]
        full = cv2.resize(
            logits, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        new_height, new_width = self._resized_shape
        cropped = full[:new_height, :new_width]
        height, width = self._original_shape
        restored = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = restored > self.mask_threshold
        if self.mask_erode_px > 0:
            kernel_size = self.mask_erode_px * 2 + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        return mask, float(score_values[index])

    @staticmethod
    def _masks_to_nchw(masks: np.ndarray, score_count: int) -> np.ndarray:
        if masks.ndim == 3:
            masks = masks[None]
        if masks.ndim != 4:
            raise RuntimeError(f"Unexpected low_res_masks shape: {masks.shape}")
        if masks.shape[1] == score_count:
            return masks
        if masks.shape[-1] == score_count:
            return masks.transpose(0, 3, 1, 2)
        raise RuntimeError(
            f"Mask channels do not match scores: masks={masks.shape}, scores={score_count}"
        )

    def close(self) -> None:
        self.encoder.close()
        self.decoder.close()

    def __enter__(self) -> "MobileSamRknn":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
