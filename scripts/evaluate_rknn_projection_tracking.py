#!/usr/bin/env python3
"""Run the complete RKNN offline semantic-map pipeline."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_map_rknn.dataset_pipeline import run_dataset


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--start", type=int, default=0)
    result.add_argument("--frames", type=int, default=0)
    result.add_argument(
        "--frame-step", type=int, default=1,
        help="Process every Nth source frame; --frames counts processed frames",
    )
    result.add_argument("--sam-encoder", type=Path, required=True)
    result.add_argument("--sam-decoder", type=Path, required=True)
    result.add_argument("--yolo-model", type=Path)
    result.add_argument("--clip-text-model", type=Path)
    result.add_argument("--text-embeddings", type=Path)
    result.add_argument("--classes-path", type=Path)
    result.add_argument("--tokenizer-path", default="openai/clip-vit-base-patch32")
    result.add_argument("--rknn-backend", choices=("auto", "lite", "toolkit"), default="auto")
    result.add_argument("--rknn-target", default="rk3588")
    result.add_argument("--yolo-core", default="0")
    result.add_argument("--sam-encoder-core", default="1")
    result.add_argument("--sam-decoder-core", default="2")
    result.add_argument(
        "--pipeline-prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pipeline RGB-D/YOLO, SAM/projection, and ordered CPU fusion",
    )
    result.add_argument("--confidence", type=float, default=0.50)
    result.add_argument("--nms-threshold", type=float, default=0.45)
    result.add_argument("--mask-threshold", type=float, default=0.0)
    result.add_argument("--mask-erode-px", type=int, default=2)
    result.add_argument("--min-depth", type=float, default=0.3)
    result.add_argument("--max-depth", type=float, default=5.0)
    result.add_argument("--pixel-stride", type=int, default=2)
    result.add_argument("--voxel-size", type=float, default=0.02)
    result.add_argument("--overlap-radius", type=float, default=0.04)
    result.add_argument("--max-centroid-distance-m", type=float, default=0.75)
    result.add_argument("--min-geometric-overlap", type=float, default=0.08)
    result.add_argument("--association-threshold", type=float, default=0.50)
    result.add_argument("--geometry-weight", type=float, default=0.70)
    result.add_argument("--semantic-weight", type=float, default=0.30)
    result.add_argument("--observation-cluster-eps", type=float, default=0.10)
    result.add_argument("--observation-cluster-min-points", type=int, default=10)
    result.add_argument("--max-extent-growth", type=float, default=1.50)
    result.add_argument("--association-max-points", type=int, default=4096)
    result.add_argument("--denoise-interval", type=int, default=0)
    result.add_argument("--map-merge-interval", type=int, default=0)
    result.add_argument("--min-confirmed-observations", type=int, default=8)
    result.add_argument("--candidate-max-missed-frames", type=int, default=30)
    result.add_argument("--progress-every", type=int, default=10)
    return result


def main() -> None:
    run_dataset(parser().parse_args())


if __name__ == "__main__":
    main()
