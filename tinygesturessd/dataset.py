"""
Real Dataset: HaGRID (images) + MediaPipe Hand Landmarker (boxes).

The `cj-mills/hagrid-sample-30k-384p` variant used here has no `bboxes` column (image +
label only), so ground-truth boxes are pseudo-labeled by running MediaPipe's Hand
Landmarker -- a real trained detector, not a color heuristic -- on each kept image and
taking the largest detected hand's bounding box (padded 10%). The gesture *class* label
still comes from HaGRID itself; MediaPipe only supplies the box.

Local addition vs. the original notebook: the MediaPipe pass takes a long time (tens of
minutes for a few thousand images), so its output is cached to disk
(config.DATASET_CACHE_DIR) and reused on subsequent runs instead of being recomputed
every time.
"""
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm

from datasets import load_dataset, load_from_disk
from huggingface_hub import login

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from . import config as cfg


# ---------------------------------------------------------------------------
# HaGRID download + per-class capping
# ---------------------------------------------------------------------------

def _load_raw_hagrid():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("logged in to Hugging Face Hub via HF_TOKEN env var")

    print(f"downloading / loading cached copy of {cfg.HF_DATASET_ID} ...")
    raw_ds = load_dataset(cfg.HF_DATASET_ID)

    hagrid_label_names = [name.replace("train_val_", "")
                           for name in raw_ds["train"].features["label"].names]
    print("HaGRID classes in this sample:", hagrid_label_names)
    return raw_ds, hagrid_label_names


def _cap_per_class(raw_ds, hagrid_label_names):
    all_labels = raw_ds["train"]["label"]
    print(f"read {len(all_labels)} labels")

    rng = random.Random(cfg.SEED)
    by_class_idx = {c: [] for c in cfg.WANTED_HAGRID_CLASSES}
    order = list(range(len(all_labels)))
    rng.shuffle(order)

    for i in order:
        cname = hagrid_label_names[all_labels[i]]
        if cname in cfg.WANTED_HAGRID_CLASSES and len(by_class_idx[cname]) < cfg.MAX_IMAGES_PER_CLASS:
            by_class_idx[cname].append(i)
        if all(len(v) >= cfg.MAX_IMAGES_PER_CLASS for v in by_class_idx.values()):
            break

    keep_indices = sorted(idx for idxs in by_class_idx.values() for idx in idxs)
    capped = raw_ds["train"].select(keep_indices)

    print("materializing the capped subset into its own compact file...")
    capped = capped.flatten_indices()

    print("per-class counts after capping:", {c: len(v) for c, v in by_class_idx.items()})
    print(f"total images going into box extraction: {len(capped)}")
    return capped


# ---------------------------------------------------------------------------
# MediaPipe Hand Landmarker pseudo-labeling
# ---------------------------------------------------------------------------

def _detect_hand_bbox(landmarker, pil_image, pad_frac=0.10):
    """
    Runs the real Hand Landmarker (not a color heuristic) on one PIL image.
    Returns (x1, y1, x2, y2) normalized to [0,1], or None if no hand found.

    If multiple hands are detected, picks the one with the larger bounding box -- the
    gesture-performing hand is usually the more prominent one. MediaPipe has no way to
    know which hand matches this row's *label*, only that a hand is present, so this is
    still a heuristic pick, just a much more reliable one than color thresholding.
    """
    rgb = np.array(pil_image.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    candidates = []
    for hand in result.hand_landmarks:
        xs = [lm.x for lm in hand]
        ys = [lm.y for lm in hand]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        candidates.append((x1, y1, x2, y2, (x2 - x1) * (y2 - y1)))

    x1, y1, x2, y2, _ = max(candidates, key=lambda c: c[4])   # largest box wins

    pw, ph = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    x1, x2 = max(0.0, x1 - pw), min(1.0, x2 + pw)
    y1, y2 = max(0.0, y1 - ph), min(1.0, y2 + ph)
    return (x1, y1, x2, y2)


def _extract_boxes_with_mediapipe(capped, hagrid_label_names):
    if not os.path.exists(cfg.MEDIAPIPE_MODEL_PATH):
        import urllib.request
        print(f"downloading hand landmarker model to {cfg.MEDIAPIPE_MODEL_PATH} ...")
        urllib.request.urlretrieve(cfg.MEDIAPIPE_MODEL_URL, cfg.MEDIAPIPE_MODEL_PATH)
        print(f"  done, {os.path.getsize(cfg.MEDIAPIPE_MODEL_PATH) / 1e6:.1f} MB")
    else:
        print(f"model already present at {cfg.MEDIAPIPE_MODEL_PATH}")

    base_options = mp_python.BaseOptions(model_asset_path=cfg.MEDIAPIPE_MODEL_PATH)
    landmarker_options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(landmarker_options)

    print(f"loading all {len(capped)} images into memory...")
    t0 = time.time()
    images_in_memory = capped["image"]
    print(f"  loaded in {time.time() - t0:.1f}s")

    print(f"running hand detection on {len(images_in_memory)} images...")
    t0 = time.time()
    final_boxes, keep_mask = [], []
    for img in tqdm(images_in_memory):
        box = _detect_hand_bbox(landmarker, img)
        keep_mask.append(box is not None)
        final_boxes.append(box)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s  ({elapsed / max(len(images_in_memory), 1) * 1000:.0f} ms/image)")

    n_found = sum(keep_mask)
    print(f"hand detected in {n_found}/{len(images_in_memory)} images ({n_found / len(images_in_memory):.1%})")

    detected_indices = [i for i, keep in enumerate(keep_mask) if keep]
    final_ds = capped.select(detected_indices)
    final_ds = final_ds.add_column("bbox", [final_boxes[i] for i in detected_indices])

    class_counts = {c: 0 for c in cfg.CLASS_NAMES}
    for lbl in final_ds["label"]:
        class_counts[cfg.HAGRID_TO_PROJECT[hagrid_label_names[lbl]]] += 1
    print("final class balance:", class_counts)
    return final_ds


# ---------------------------------------------------------------------------
# PyTorch Dataset / DataLoaders
# ---------------------------------------------------------------------------

class HaGRIDGestureDataset(Dataset):
    def __init__(self, hf_split, hagrid_label_names, img_size: int = cfg.IMG_SIZE):
        self.ds = hf_split
        self.hagrid_label_names = hagrid_label_names
        self.img_size = img_size

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = row["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        x1, y1, x2, y2 = row["bbox"]
        box_t = torch.tensor([x1, y1, x2, y2], dtype=torch.float32).clamp(0.0, 1.0)

        label_name = cfg.HAGRID_TO_PROJECT[self.hagrid_label_names[row["label"]]]
        label_t = torch.tensor(cfg.CLASS_NAMES.index(label_name), dtype=torch.long)

        image_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return image_t, box_t, label_t


def prepare_dataset(force_rebuild: bool = False):
    """
    Returns (train_loader, val_loader, hagrid_label_names).

    Caches the MediaPipe-pseudo-labeled dataset to `config.DATASET_CACHE_DIR` so the
    (slow) MediaPipe pass only ever runs once per MAX_IMAGES_PER_CLASS/SEED combination.
    Pass force_rebuild=True to ignore an existing cache and recompute it.
    """
    raw_ds, hagrid_label_names = _load_raw_hagrid()   # cheap; needed either way for label names

    if os.path.isdir(cfg.DATASET_CACHE_DIR) and not force_rebuild:
        print(f"found cached pseudo-labeled dataset at {cfg.DATASET_CACHE_DIR}, skipping MediaPipe pass")
        final_ds = load_from_disk(cfg.DATASET_CACHE_DIR)
    else:
        capped = _cap_per_class(raw_ds, hagrid_label_names)
        final_ds = _extract_boxes_with_mediapipe(capped, hagrid_label_names)
        print(f"caching pseudo-labeled dataset to {cfg.DATASET_CACHE_DIR} for future runs...")
        final_ds.save_to_disk(cfg.DATASET_CACHE_DIR)

    final_ds = final_ds.shuffle(seed=cfg.SEED)
    n_val = max(1, int(0.1 * len(final_ds)))
    hf_val = final_ds.select(range(n_val))
    hf_train = final_ds.select(range(n_val, len(final_ds)))
    print(f"train: {len(hf_train)}   val: {len(hf_val)}")

    train_ds = HaGRIDGestureDataset(hf_train, hagrid_label_names)
    val_ds = HaGRIDGestureDataset(hf_val, hagrid_label_names)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=torch.cuda.is_available())

    imgs, boxes, labels = next(iter(train_loader))
    print("batch:", tuple(imgs.shape), tuple(boxes.shape), tuple(labels.shape))

    return train_loader, val_loader, hagrid_label_names
