#!/usr/bin/env python3
"""Check OrangePi RKNN runtime dependencies and optionally load all models."""

from pathlib import Path
import argparse
import importlib
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-encoder", type=Path)
    parser.add_argument("--sam-decoder", type=Path)
    parser.add_argument("--yolo-model", type=Path)
    parser.add_argument("--clip-text-model", type=Path)
    parser.add_argument("--load-models", action="store_true")
    args = parser.parse_args()
    failed = False
    print(f"architecture: {platform.machine()}")
    print(f"python: {platform.python_version()}")
    for module in ("numpy", "cv2", "scipy", "rclpy", "rknnlite.api"):
        try:
            loaded = importlib.import_module(module)
            version = getattr(loaded, "__version__", "available")
            print(f"[OK] {module}: {version}")
        except Exception as exc:
            failed = True
            print(f"[FAIL] {module}: {exc!r}")
    paths = [args.sam_encoder, args.sam_decoder, args.yolo_model, args.clip_text_model]
    for path in (item for item in paths if item is not None):
        exists = path.expanduser().is_file()
        print(f"[{'OK' if exists else 'FAIL'}] model: {path.expanduser()}")
        failed |= not exists
    if args.load_models and not failed:
        from semantic_map_rknn.rknn_runtime import RknnSession

        for path in (item for item in paths if item is not None):
            print(f"loading: {path}")
            session = RknnSession(path, backend="lite", core_mask="0_1_2")
            session.close()
            print(f"[OK] runtime: {path.name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
