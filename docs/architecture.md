# Architecture and training details

This note documents the exact tensor contract and the non-obvious design choices
in the code. It matches the released checkpoints and the paper's description.

## 1. Input contract

| Property | Value |
|---|---|
| Input | RGB image, 128 × 128 |
| Tensor layout | `float32`, NCHW, `[B, 3, 128, 128]` |
| Value range | `[0, 1]` (no ImageNet mean/std normalization) |
| Hand assumption | at most one hand per frame |

The `[0,1]` range is deliberate: the INT8 deployment path multiplies by
`2^7 = 128` and rounds/clips to `int8`, so the host-side float model and the
device-side quantized model see the same normalized input. The dataset code
resizes with bilinear interpolation and divides by 255; it does not apply
ImageNet normalization.

## 2. Backbone

`torchvision.models.mobilenet_v2(weights=None, width_mult=W)` followed by
`features[:14]`. The cut is after the stride-16 block, so a 128 × 128 input
produces an 8 × 8 feature map. The channel count depends on `W`:

| `width_mult` | Output channels | Grid | Trainable parameters |
|---:|---:|---:|---:|
| 0.25 | 24 | 8 × 8 | 46,182 |
| 0.30 | 32 | 8 × 8 | 86,870 |
| 0.40 | 40 | 8 × 8 | 109,382 |
| 0.50 | 48 | 8 × 8 | 153,526 |
| 0.75 | 72 | 8 × 8 | 326,214 |
| 1.00 | 96 | 8 × 8 | 546,774 |

The head's input channel count is inferred from a dummy forward pass, never
hard-coded, so `width_mult` is the single capacity knob. `W=0.30` is the
deployed knowledge-distilled student; `W=0.50` is the package default baseline;
`W=1.00` is the ImageNet-pretrained teacher.

## 3. Detection heads

Both heads are SSDLite-style depthwise-separable blocks:

```text
input (B, C, 8, 8)
  └─ depthwise 3×3 conv, groups=C, padding=1, bias=False
  └─ BatchNorm2d(C)
  └─ ReLU6
  └─ pointwise 1×1 conv, bias=True
       ├─ classification: 2 × (6+1) = 14 channels
       └─ localization:   2 × 4     =  8 channels
```

The classification output is reshaped from `(B, 14, 8, 8)` to
`(B, 128, 7)`, and the localization output from `(B, 8, 8, 8)` to
`(B, 128, 4)`. The reshape is:

```text
(B, A_cell*K, H, W) -> view(B, A_cell, K, H, W)
                    -> permute(0, 3, 4, 1, 2)   # (B, H, W, A_cell, K)
                    -> view(B, H*W*A_cell, K)
```

so the anchor order is **row-major cells, then aspect ratio within each cell**.
The anchor generator emits the same order.

The background class is channel 0 of the 7-way classification output; the six
gesture classes are channels 1–6 in `CLASS_NAMES` order:
`fist`, `palm`, `like`, `ok`, `peace`, `point`.

## 4. Anchors

For an 8 × 8 grid, two aspect ratios `r ∈ {1.0, 1.8}`, and base size
`48/128 = 0.375`:

```text
cx = (col + 0.5) * 16 / 128
cy = (row + 0.5) * 16 / 128
w  = 0.375 * sqrt(r)
h  = 0.375 / sqrt(r)
```

There are `8*8*2 = 128` anchors. The `1.0` ratio covers fists/palms; the `1.8`
ratio covers elongated pointing/peace gestures. `anchors.py` also contains a
slow loop reference and `verify_anchors()` compares the vectorized output to it
exactly.

## 5. Target assignment and box encoding

Each image has exactly one ground-truth hand box (normalized corner form). The
IoU between that box and all 128 anchors is computed in one broadcast:

- `IoU >= 0.5`: positive, assigned the foreground class.
- `IoU < 0.4`: negative (background).
- `0.4 <= IoU < 0.5`: ignored, excluded from both losses.

The single best-IoU anchor per image is forced positive even if it does not
clear 0.5, so every image contributes at least one localization target.

Positives are encoded relative to their anchor with the standard SSD
parameterization and variances `(0.1, 0.1, 0.2, 0.2)`:

```text
t_cx = (g_cx - a_cx) / a_w / 0.1
t_cy = (g_cy - a_cy) / a_h / 0.1
t_w  = log(g_w / a_w) / 0.2
t_h  = log(g_h / a_h) / 0.2
```

`decode_boxes()` is the exact inverse and is used at inference. Boxes are
clamped to `[0, 1]` after decoding.

## 6. Loss

```text
loc_loss = SmoothL1(pred_loc[pos], target_loc[pos], reduction="sum") / num_pos
cls_loss = CE(pred_cls[selected], target_cls[selected], reduction="sum") / num_pos
total    = loc_loss + cls_loss
```

Classification uses 3:1 hard-negative mining: for each image, rank the
background anchors by their per-anchor cross-entropy, keep the hardest
`min(3 * num_pos, num_neg_available)`, and ignore the rest. Positives and
ignored anchors are masked out of the ranking with `-inf`. The double-argsort
implementation avoids a Python loop over the batch.

## 7. Inference

There is no NMS. The deployment target assumes one hand, so:

1. `softmax` over the 7 class channels.
2. Drop background; take the best foreground score per anchor.
3. Threshold at 0.5 (configurable).
4. Take one global argmax over all 128 anchors.
5. Decode that anchor's box and return the class, score, and box.

A missed detection (no anchor above threshold) is counted as a wrong prediction
in the metrics, which makes the reported accuracy/F1 include detection recall.
Bounding-box localization is not scored in the paper.

## 8. ONNX / ESP-DL contract

The exported graph is static and batch-1:

| Tensor | Shape | Meaning |
|---|---|---|
| `input` | `[1, 3, 128, 128]` | `float32` for ONNX, `int8` after quantization |
| `cls_logits` | `[1, 128, 7]` | background + six gestures |
| `loc_preds` | `[1, 128, 4]` | encoded `(t_cx, t_cy, t_w, t_h)` |

`torch.onnx.export` is called with `opset_version=13`, `dynamic_axes=None`,
`do_constant_folding=True`, and `dynamo=False`. ESP-PPQ and ESP-DL expect fixed
shapes; do not add dynamic axes. The ONNX and PyTorch outputs are checked with
`atol=1e-4` in `export.py` and in the test suite.

## 9. Reproducibility notes

- The split is seed-0, stratified 90/10, with 3,906 train / 433 validation
  images. The validation label MD5 is
  `d210029e2814357ca28251390535186e` (`results/metrics/split_fingerprint.json`).
- The width sweep is single-seed and unaugmented; the width-0.30 row memorized
  the training set. The three-seed augmented family is the fair comparison.
- The checkpoint selection metric is validation macro-F1 by default; it can be
  changed with `--selection-metric`.
- The evaluation set is reused for checkpoint selection, PTQ calibration, and
  QAT epoch selection. Absolute scores are therefore mildly optimistic; the
  paper's bootstrap intervals quantify evaluation-set noise only.
