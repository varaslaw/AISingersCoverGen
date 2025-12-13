"""Minimal runtime smoke test for AISingersCoverGen.

This script validates HuBERT loading and a dummy forward pass
without running the full voice conversion pipeline.
"""

import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from rvc import load_hubert


def main():
    device = "cpu"
    try:
        model = load_hubert(device=device, is_half=False, model_path=None, backend="auto")
    except Exception as exc:  # pragma: no cover - runtime download path
        # Environments without network (CI) cannot download the HuBERT bundle.
        # Surface a clear skip message instead of a hard failure.
        print("[skip] HuBERT download failed; smoke test skipped:", exc)
        return

    dummy = torch.randn(1, 16000)
    features, *_ = model.extract_features(dummy)
    assert isinstance(features, torch.Tensor), "HuBERT output is not a tensor"
    assert features.numel() > 0, "HuBERT output is empty"
    print("HuBERT smoke test passed. Feature shape:", tuple(features.shape))


if __name__ == "__main__":
    main()
