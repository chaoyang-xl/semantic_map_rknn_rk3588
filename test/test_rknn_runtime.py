from pathlib import Path
from threading import Lock

from semantic_map_rknn.rknn_runtime import RknnSession


class RuntimeRecorder:
    def __init__(self):
        self.kwargs = None

    def inference(self, **kwargs):
        self.kwargs = kwargs
        return [1]


def test_lite_multi_input_data_format_is_expanded():
    recorder = RuntimeRecorder()
    session = RknnSession.__new__(RknnSession)
    session.model_path = Path("decoder.rknn")
    session._runtime = recorder
    session._is_lite = True
    session._lock = Lock()
    assert session.inference([1, 2, 3], data_format="NCHW") == [1]
    assert recorder.kwargs["data_format"] == ["nchw", "nchw", "nchw"]


def test_toolkit_keeps_official_data_format_string():
    recorder = RuntimeRecorder()
    session = RknnSession.__new__(RknnSession)
    session.model_path = Path("decoder.rknn")
    session._runtime = recorder
    session._is_lite = False
    session._lock = Lock()
    session.inference([1, 2], data_format="NCHW")
    assert recorder.kwargs["data_format"] == "NCHW"
