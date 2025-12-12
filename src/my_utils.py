import importlib
from typing import Any

import ffmpeg
import numpy as np


def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


def optional_import(module_name: str, feature: str, install_hint: str, *, raise_error: bool = True, default: Any = None):
    """Attempt to import a module and provide a helpful hint when missing.

    Args:
        module_name: Module to import via :func:`importlib.import_module`.
        feature: Human-friendly description of the feature that needs the module.
        install_hint: Guidance on how to install the missing dependency.
        raise_error: Whether to raise an :class:`ImportError` when the import fails.
        default: Value to return when the import is optional and the module is missing.

    Returns:
        The imported module or ``default`` when ``raise_error`` is ``False``.
    """

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        if raise_error:
            raise ImportError(
                f"{feature} requires `{module_name}`. {install_hint}"
            ) from exc
        return default
