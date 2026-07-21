#!/usr/bin/env python3
"""Generate and save the fixed YOLO-World text input on an RK3588 board."""

from pathlib import Path
import argparse
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_map_rknn.yolo_world_rknn import YoloWorldRknn, load_class_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-model", type=Path, required=True)
    parser.add_argument("--clip-text-model", type=Path, required=True)
    parser.add_argument("--classes-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rknn-backend", default="lite", choices=("auto", "lite", "toolkit"))
    parser.add_argument("--rknn-target", default="rk3588")
    args = parser.parse_args()
    detector = YoloWorldRknn(
        args.yolo_model,
        load_class_names(args.classes_path),
        text_model_path=args.clip_text_model,
        tokenizer_path=args.tokenizer_path,
        backend=args.rknn_backend,
        target=args.rknn_target,
    )
    try:
        args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output.expanduser(), detector.text_embeddings)
        print(f"Saved {detector.text_embeddings.shape}: {args.output.expanduser()}")
    finally:
        detector.close()


if __name__ == "__main__":
    main()
