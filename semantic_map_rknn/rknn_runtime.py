"""Small compatibility layer for RKNN Toolkit and RKNN Lite runtimes."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any


class RknnSession:
    """Own one loaded RKNN model and serialize access to its runtime context."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        backend: str = "auto",
        target: str = "rk3588",
        core_mask: str = "auto",
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"RKNN model not found: {self.model_path}")
        self.backend = backend
        self.target = target
        self.core_mask = core_mask
        self._runtime: Any = None
        self._is_lite = False
        self._lock = Lock()
        self._open()

    def _open(self) -> None:
        errors: list[str] = []
        if self.backend in ("auto", "lite"):
            try:
                from rknnlite.api import RKNNLite

                runtime = RKNNLite(verbose=False)
                ret = runtime.load_rknn(str(self.model_path))
                if ret != 0:
                    raise RuntimeError(f"load_rknn returned {ret}")
                kwargs = {}
                mask = self._lite_core_mask(RKNNLite, self.core_mask)
                if mask is not None:
                    kwargs["core_mask"] = mask
                ret = runtime.init_runtime(**kwargs)
                if ret != 0:
                    raise RuntimeError(f"init_runtime returned {ret}")
                self._runtime = runtime
                self._is_lite = True
                self.backend = "lite"
                return
            except Exception as exc:  # pragma: no cover - depends on board runtime
                errors.append(f"RKNNLite: {exc!r}")
                if self.backend == "lite":
                    raise RuntimeError("; ".join(errors)) from exc

        if self.backend in ("auto", "toolkit"):
            try:
                from rknn.api import RKNN

                runtime = RKNN(verbose=False)
                ret = runtime.load_rknn(str(self.model_path))
                if ret != 0:
                    raise RuntimeError(f"load_rknn returned {ret}")
                ret = runtime.init_runtime(target=self.target)
                if ret != 0:
                    raise RuntimeError(f"init_runtime returned {ret}")
                self._runtime = runtime
                self.backend = "toolkit"
                return
            except Exception as exc:  # pragma: no cover - requires toolkit/device
                errors.append(f"RKNN Toolkit: {exc!r}")
                raise RuntimeError(
                    f"Unable to initialize {self.model_path.name}: " + "; ".join(errors)
                ) from exc
        raise ValueError("backend must be auto, lite, or toolkit")

    @staticmethod
    def _lite_core_mask(api, value: str):
        names = {
            "auto": None,
            "0": "NPU_CORE_0",
            "1": "NPU_CORE_1",
            "2": "NPU_CORE_2",
            "0_1": "NPU_CORE_0_1",
            "0_1_2": "NPU_CORE_0_1_2",
            "all": "NPU_CORE_0_1_2",
        }
        key = value.strip().lower()
        if key not in names:
            raise ValueError(f"Unsupported RKNN core mask: {value}")
        attribute = names[key]
        return None if attribute is None else getattr(api, attribute)

    def inference(self, inputs: list, *, data_format=None) -> list:
        if self._runtime is None:
            raise RuntimeError("RKNN session is closed")
        kwargs = {"inputs": inputs}
        if data_format is not None:
            if self._is_lite and isinstance(data_format, str):
                # RKNN Toolkit accepts one string; RKNNLite2 expects one entry
                # per input for multi-input models such as MobileSAM decoder.
                kwargs["data_format"] = [data_format.lower()] * len(inputs)
            else:
                kwargs["data_format"] = data_format
        with self._lock:
            outputs = self._runtime.inference(**kwargs)
        if outputs is None:
            raise RuntimeError(f"RKNN inference failed: {self.model_path.name}")
        return list(outputs)

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.release()
            self._runtime = None

    def __enter__(self) -> "RknnSession":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort process cleanup
        try:
            self.close()
        except Exception:
            pass
