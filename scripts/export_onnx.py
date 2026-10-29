"""
Exports a TinyGestureSSD checkpoint to a static-shape ONNX graph for ESP-PPQ / ESP-DL.

Run from the project root (after scripts/train.py has produced best_model.pt):
    python scripts/export_onnx.py
    python scripts/export_onnx.py --width-mult 0.30 \
        --checkpoint models/kd/best_model.pt --out models/kd/gesture_ssdlite.onnx

The width multiplier must match the checkpoint. The released KD student is
width 0.30; the preserved baseline checkpoint is models/baseline_w0p50 (0.50).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from tinygesturessd import config as cfg
from tinygesturessd.model import SSDLiteGestureDetector
from tinygesturessd.export import export_to_onnx, verify_onnx


def parse_args():
    parser = argparse.ArgumentParser(description="Export TinyGestureSSD to ONNX.")
    parser.add_argument("--width-mult", type=float, default=cfg.WIDTH_MULT,
                        help="Must match the checkpoint (baseline 0.50, KD student 0.30).")
    parser.add_argument("--checkpoint", default=os.path.join(cfg.MODELS_DIR, "best_model.pt"))
    parser.add_argument("--out", default=os.path.join(cfg.MODELS_DIR, "gesture_ssdlite.onnx"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")   # export always happens on CPU
    model = SSDLiteGestureDetector(width_mult=args.width_mult).to(device)
    model.param_summary()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"{args.checkpoint} not found -- run scripts/train.py first")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    onnx_path, export_model, dummy_input = export_to_onnx(model, onnx_path=args.out)
    verify_onnx(onnx_path, export_model, dummy_input)
    print(f"ONNX export complete: {onnx_path}")


if __name__ == "__main__":
    main()
