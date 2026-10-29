"""
Entry point: trains TinyGestureSSD end-to-end on real HaGRID + MediaPipe pseudo-labels,
and saves the best checkpoint, a loss/metric plot, and a small JSON summary into models/.

Run from the project root:
    python scripts/train.py
    python scripts/train.py --width-mult 0.30 --epochs 60

The deployed knowledge-distilled student in the paper is width 0.30 (86,870
trainable parameters). The default here is the width-0.50 baseline row; the KD
recipe itself lives in ``notebooks/02_knowledge_distillation.ipynb``.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib.pyplot as plt
import torch

from tinygesturessd import config as cfg
from tinygesturessd.anchors import generate_anchors, center_to_corner, verify_anchors
from tinygesturessd.model import SSDLiteGestureDetector
from tinygesturessd.dataset import prepare_dataset
from tinygesturessd.training import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyGestureSSD.")
    parser.add_argument("--width-mult", type=float, default=cfg.WIDTH_MULT,
                        help="MobileNetV2 width multiplier (paper sweep: 0.25-1.0).")
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--max-images-per-class", type=int, default=cfg.MAX_IMAGES_PER_CLASS,
                        help="Per-class cap before MediaPipe box extraction.")
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--lr", type=float, default=cfg.LR)
    parser.add_argument("--weight-decay", type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument("--selection-metric", default=cfg.SELECTION_METRIC,
                        choices=["accuracy", "precision", "recall", "f1_macro"])
    parser.add_argument("--out-dir", default=cfg.MODELS_DIR,
                        help="Directory for best_model.pt, the plot, and the summary.")
    parser.add_argument("--data-cache-dir", default=None,
                        help="Where to cache the MediaPipe-pseudo-labelled dataset.")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Ignore an existing pseudo-label cache and recompute it.")
    return parser.parse_args()


def main():
    args = parse_args()

    # The dataset/training modules read these globals at call time, so CLI
    # overrides can be applied without mutating function defaults.
    cfg.SEED = args.seed
    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.MAX_IMAGES_PER_CLASS = args.max_images_per_class
    cfg.SELECTION_METRIC = args.selection_metric
    cfg.MODELS_DIR = os.path.abspath(args.out_dir)
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    cfg.MEDIAPIPE_MODEL_PATH = os.path.join(cfg.MODELS_DIR, "hand_landmarker.task")
    cfg.DATASET_CACHE_DIR = (
        os.path.abspath(args.data_cache_dir)
        if args.data_cache_dir
        else os.path.join(cfg.MODELS_DIR, "hagrid_pseudo_labeled")
    )

    cfg.set_seed(args.seed)
    device = cfg.get_device()
    print(f"grid: {cfg.GRID_SIZE}x{cfg.GRID_SIZE}  anchors/cell: {cfg.NUM_ANCHORS_PER_CELL}  "
          f"total anchors: {cfg.NUM_TOTAL_ANCHORS}")
    print(f"classes: {cfg.CLASS_NAMES}  (+ background)")
    print(f"config: width_mult={args.width_mult}  epochs={args.epochs}  "
          f"batch_size={args.batch_size}  max/class={args.max_images_per_class}\n")

    # ---- anchors --------------------------------------------------------
    anchors_center = generate_anchors(cfg.IMG_SIZE, cfg.STRIDE, cfg.ANCHOR_BASE_PX, cfg.ASPECT_RATIOS)
    verify_anchors(anchors_center, cfg.IMG_SIZE, cfg.STRIDE, cfg.ANCHOR_BASE_PX, cfg.ASPECT_RATIOS)
    anchors_corner = center_to_corner(anchors_center)
    anchors_center_d = anchors_center.to(device)
    anchors_corner_d = anchors_corner.to(device)
    print(f"anchors: {tuple(anchors_center.shape)}\n")

    # ---- model ------------------------------------------------------------
    model = SSDLiteGestureDetector(width_mult=args.width_mult).to(device)
    model.param_summary()
    print(f"\nbackbone output: {model.in_channels} channels @ {model.grid_size}x{model.grid_size}")

    model.eval()
    with torch.no_grad():
        _cls, _loc = model(torch.randn(4, 3, cfg.IMG_SIZE, cfg.IMG_SIZE, device=device))
    print(f"forward OK -> cls {tuple(_cls.shape)}   loc {tuple(_loc.shape)}\n")

    # ---- dataset ------------------------------------------------------------
    train_loader, val_loader, _ = prepare_dataset(force_rebuild=args.force_rebuild)

    # ---- train ------------------------------------------------------------
    history, best_epoch, best_metric_value = train_model(
        model, train_loader, val_loader, anchors_center_d, anchors_corner_d, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        selection_metric=args.selection_metric,
    )

    # ---- plot ------------------------------------------------------------
    epochs_range = range(1, len(history["total"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(epochs_range, history["total"], marker="o", label="total")
    axes[0].plot(epochs_range, history["loc"], marker="o", label="loc")
    axes[0].plot(epochs_range, history["cls"], marker="o", label="cls")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for key, label in [("val_accuracy", "accuracy"), ("val_precision", "precision"),
                       ("val_recall", "recall"), ("val_f1_macro", "F1 (macro)")]:
        axes[1].plot(epochs_range, history[key], marker="o", label=label)
    axes[1].axvline(best_epoch, color="gray", linestyle="--", alpha=0.6, label=f"best epoch ({best_epoch})")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(cfg.MODELS_DIR, "training_curves.png")
    fig.savefig(plot_path, dpi=150)
    print(f"\nsaved training curves to {plot_path}")

    best = best_epoch - 1
    summary = {
        "width_mult": args.width_mult,
        "trainable_parameters": int(model.param_summary(verbose=False)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_images_per_class": args.max_images_per_class,
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "selection_metric": args.selection_metric,
        "best_epoch": int(best_epoch),
        "best_selection_value": float(best_metric_value),
        "val_accuracy": float(history["val_accuracy"][best]),
        "val_precision": float(history["val_precision"][best]),
        "val_recall": float(history["val_recall"][best]),
        "val_f1_macro": float(history["val_f1_macro"][best]),
    }
    summary_path = os.path.join(cfg.MODELS_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"saved training summary to {summary_path}")

    print(f"best epoch: {best_epoch}   selection metric: {cfg.SELECTION_METRIC} = {best_metric_value:.4f}")
    print(f"at best epoch -> accuracy={summary['val_accuracy']:.4f}  "
          f"precision={summary['val_precision']:.4f}  "
          f"recall={summary['val_recall']:.4f}  "
          f"f1={summary['val_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
