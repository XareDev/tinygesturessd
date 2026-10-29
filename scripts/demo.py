"""
Runs a trained TinyGestureSSD model on validation images and saves a figure
comparing predicted vs. ground-truth boxes.

Run from the project root (after scripts/train.py has produced a checkpoint):
    python scripts/demo.py
    python scripts/demo.py --width-mult 0.30 \
        --checkpoint models/kd/best_model.pt --out models/kd/demo_predictions.png

The width multiplier must match the checkpoint.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from tinygesturessd import config as cfg
from tinygesturessd.anchors import generate_anchors
from tinygesturessd.model import SSDLiteGestureDetector
from tinygesturessd.dataset import prepare_dataset
from tinygesturessd.inference import detect_single_hand


def parse_args():
    parser = argparse.ArgumentParser(description="Render TinyGestureSSD predictions.")
    parser.add_argument("--width-mult", type=float, default=cfg.WIDTH_MULT,
                        help="Must match the checkpoint (baseline 0.50, KD student 0.30).")
    parser.add_argument("--checkpoint", default=os.path.join(cfg.MODELS_DIR, "best_model.pt"))
    parser.add_argument("--out", default=os.path.join(cfg.MODELS_DIR, "demo_predictions.png"))
    parser.add_argument("--n", type=int, default=8, help="Number of validation images to draw.")
    parser.add_argument("--conf-thresh", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = cfg.get_device()

    anchors_center = generate_anchors(
        cfg.IMG_SIZE, cfg.STRIDE, cfg.ANCHOR_BASE_PX, cfg.ASPECT_RATIOS
    ).to(device)

    model = SSDLiteGestureDetector(width_mult=args.width_mult).to(device)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"{args.checkpoint} not found -- run scripts/train.py first")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    _, val_loader, _ = prepare_dataset()
    val_ds = val_loader.dataset

    n = min(args.n, len(val_ds))
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)

    n_hits = 0
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        img, gt_box, gt_label = val_ds[i]
        det = detect_single_hand(model, img.unsqueeze(0).to(device), anchors_center,
                                 conf_thresh=args.conf_thresh)

        ax.imshow(img.permute(1, 2, 0).numpy())
        gx1, gy1, gx2, gy2 = (gt_box * cfg.IMG_SIZE).tolist()
        ax.add_patch(patches.Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1, fill=False,
                                       edgecolor="lime", linewidth=1.5, linestyle="--"))
        gt_name = cfg.CLASS_NAMES[int(gt_label)]
        if det is not None:
            n_hits += 1
            px1, py1, px2, py2 = det["box_px"].tolist()
            ax.add_patch(patches.Rectangle((px1, py1), px2 - px1, py2 - py1, fill=False,
                                           edgecolor="red", linewidth=2))
            ax.set_title(f"gt={gt_name}  pred={det['label_name']} {det['score']:.2f}", fontsize=8)
        else:
            ax.set_title(f"gt={gt_name}  no detection", fontsize=8)
        ax.axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"detections above threshold: {n_hits}/{n}  "
          f"(dashed green = ground truth, solid red = prediction)")
    print(f"saved figure to {args.out}")


if __name__ == "__main__":
    main()
