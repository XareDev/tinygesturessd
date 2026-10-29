"""Fast tests that do not download HaGRID or run MediaPipe.

Run from the repository root:
    pytest
    python -m pytest tests/test_smoke.py
"""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tinygesturessd import config as cfg
from tinygesturessd.anchors import center_to_corner, generate_anchors
from tinygesturessd.model import SSDLiteGestureDetector

ROOT = Path(__file__).resolve().parents[1]


def test_parameter_counts_match_paper():
    assert SSDLiteGestureDetector(width_mult=0.25).param_summary(verbose=False) == 46_182
    assert SSDLiteGestureDetector(width_mult=0.30).param_summary(verbose=False) == 86_870
    assert SSDLiteGestureDetector(width_mult=0.50).param_summary(verbose=False) == 153_526
    assert SSDLiteGestureDetector(width_mult=1.00).param_summary(verbose=False) == 546_774


def test_forward_shapes():
    model = SSDLiteGestureDetector(width_mult=0.30).eval()
    with torch.no_grad():
        cls, loc = model(torch.randn(2, 3, cfg.IMG_SIZE, cfg.IMG_SIZE))
    assert cls.shape == (2, cfg.NUM_TOTAL_ANCHORS, cfg.NUM_CLASSES + 1)
    assert loc.shape == (2, cfg.NUM_TOTAL_ANCHORS, 4)


def test_anchor_geometry_round_trip():
    anchors = generate_anchors(cfg.IMG_SIZE, cfg.STRIDE, cfg.ANCHOR_BASE_PX, cfg.ASPECT_RATIOS)
    assert anchors.shape == (cfg.NUM_TOTAL_ANCHORS, 4)
    assert torch.all(anchors[:, :2] >= 0) and torch.all(anchors[:, :2] <= 1)
    corners = center_to_corner(anchors)
    assert corners.shape == anchors.shape
    assert torch.allclose(corners[:, 2:] - corners[:, :2], anchors[:, 2:], atol=1e-6)


def test_released_kd_checkpoint_matches_its_onnx_export():
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    ckpt = ROOT / "models" / "kd" / "best_model.pt"
    onnx_path = ROOT / "models" / "kd" / "gesture_ssdlite.onnx"
    if not ckpt.exists() or not onnx_path.exists():
        pytest.skip("released KD checkpoint/ONNX not present")

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    model = SSDLiteGestureDetector(width_mult=0.30).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    dummy = torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE)
    with torch.no_grad():
        torch_cls, torch_loc = model(dummy)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_cls, ort_loc = sess.run(None, {"input": dummy.numpy()})
    assert np.allclose(ort_cls, torch_cls.numpy(), atol=1e-4)
    assert np.allclose(ort_loc, torch_loc.numpy(), atol=1e-4)
