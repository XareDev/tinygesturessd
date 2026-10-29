"""
All hyperparameters, class names, and paths in one place.

This mirrors Section 1 ("Imports & Configuration") of the original notebook, with
the Colab-only setup (Drive mount, Colab Secrets, `!nvidia-smi`) replaced by plain
local paths and a straightforward `torch.cuda.is_available()` check.
"""
import os
import random

import numpy as np
import torch

# ---- reproducibility --------------------------------------------------------
SEED = 0


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---- geometry -----------------------------------------------------------
IMG_SIZE = 128
STRIDE = 16                                # single tap point: 128 / 16 = 8x8 feature map
GRID_SIZE = IMG_SIZE // STRIDE             # 8
ASPECT_RATIOS = (1.0, 1.8)                 # 1:1 (fist/palm) + elongated (pointing)
NUM_ANCHORS_PER_CELL = len(ASPECT_RATIOS)  # 2
ANCHOR_BASE_PX = 48                        # base anchor size in pixels on the 128px frame
NUM_TOTAL_ANCHORS = GRID_SIZE * GRID_SIZE * NUM_ANCHORS_PER_CELL  # 8*8*2 = 128

# ---- backbone -------------------------------------------------------------
CUT_INDEX = 14                             # features[:14] -> output stride 16, 8x8

# Baseline width used by the from-scratch training path. The deployed
# knowledge-distilled student in the paper is width 0.30 (86,870 trainable
# parameters); width 0.50 is the baseline row of the width sweep. Override
# either value from the CLI, e.g. `python scripts/train.py --width-mult 0.30`.
WIDTH_MULT = 0.5

# ---- classes ----------------------------------------------------------------
CLASS_NAMES = ["fist", "palm", "like", "ok", "peace", "point"]
NUM_CLASSES = len(CLASS_NAMES)             # background is implicit class 0 in the head output

# ---- box encoding / matching -----------------------------------------------
VARIANCES = (0.1, 0.1, 0.2, 0.2)           # standard SSD-style scale factors for regression targets
POS_IOU_THRESH = 0.5
NEG_IOU_THRESH = 0.4
NEG_POS_RATIO = 3                          # hard-negative-mining ratio

# ---- dataset ----------------------------------------------------------------
HF_DATASET_ID = "cj-mills/hagrid-sample-30k-384p"
HAGRID_TO_PROJECT = {"fist": "fist", "palm": "palm", "like": "like",
                      "ok": "ok", "peace": "peace", "one": "point"}
WANTED_HAGRID_CLASSES = set(HAGRID_TO_PROJECT.keys())
MAX_IMAGES_PER_CLASS = 800                 # per-class cap before MediaPipe box extraction

# ---- training ----------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 12
LR = 2e-3
WEIGHT_DECAY = 1e-4
SELECTION_METRIC = "f1_macro"              # which val metric selects the saved checkpoint

# ---- mediapipe (pseudo-label source for ground-truth boxes) -----------------
MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# ---- paths ----------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MEDIAPIPE_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
DATASET_CACHE_DIR = os.path.join(MODELS_DIR, "hagrid_pseudo_labeled")


# ---- device ----------------------------------------------------------------
def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # fixed 128x128 input shape -> let cuDNN pick fast kernels
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: no GPU detected -- running on CPU. Training will be much slower.")
    return device
