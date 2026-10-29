# Released models

Every checkpoint, ONNX graph and ESP-DL `.espdl` file that fits comfortably in
the repository is here. The full SHA-256 manifest is
[`SHA256SUMS`](SHA256SUMS); file paths in it are relative to the repository root.

**Read this first:** the `width_mult` must match the checkpoint. Loading a
width-0.30 checkpoint into a width-0.50 model raises a size-mismatch error, and
loading it with `strict=False` would silently leave random weights. The width is
in each folder/file name where it matters.

## Recommended files

| Goal | File | Width | Size | F1 |
|---|---|---:|---:|---:|
| Deploy the paper's student | `kd/gesture_ssdlite_qat.espdl` | 0.30 | 183.31 KB | 69.20 |
| Deploy PTQ without test tensors | `kd/gesture_ssdlite_no_test.espdl` | 0.30 | 183.33 KB | 67.88 |
| Reproduce the student in PyTorch | `kd/best_model.pt` | 0.30 | 447,672 B | 69.78 |
| Reproduce the teacher | `kd/teacher_best.pt` | 1.00 | 2,339,892 B | 86.92 |
| Deploy the 1.0 teacher | `pretrained/gesture_ssdlite_qat.espdl` | 1.00 | 643.03 KB | 83.94 |
| Matched 0.50 checkpoint + export | `width_sweep/model_w0p5_best.pt` + `model_w0p5.onnx` | 0.50 | 723,768 B / 628,988 B | 25.87 (fp32, unaug) |
| Matched 0.75 KD checkpoint + export | `width_sweep/student_w0p75_best.pt` + `student_w0p75.onnx` | 0.75 | 1,432,552 B / 1,311,120 B | 81.33 (fp32) |

## Folder inventory

### `kd/` — released knowledge-distilled student and teacher

| File | What it is | Size |
|---|---|---:|
| `best_model.pt` | Width-0.30 KD student, seed 0 (the released student) | 447,672 B |
| `teacher_best.pt` | Width-1.0 pretrained teacher used by the KD notebook | 2,339,892 B |
| `gesture_ssdlite.onnx` | Static ONNX export of `best_model.pt` | 366,694 B |
| `gesture_ssdlite.espdl` | PTQ-INT8 with embedded test tensors (paper table) | 238,576 B |
| `gesture_ssdlite.json` | ESP-PPQ graph metadata for the above | 250,637 B |
| `gesture_ssdlite_no_test.espdl` | PTQ-INT8 re-export without test tensors | 187,728 B |
| `gesture_ssdlite_no_test.json` | Metadata for the re-export | 250,544 B |
| `gesture_ssdlite_qat.espdl` | QAT-INT8 (best epoch 6) | 187,712 B |
| `gesture_ssdlite_qat.json` | Metadata for the QAT export | 250,600 B |
| `ab_A_unaug_s0.pt` | Ablation A, no augmentation, seed 0 (width 0.30) | 448,434 B |

Metrics: fp32 57.51% accuracy / 69.78% F1; PTQ 56.12 / 67.88; QAT 57.04 /
69.20. The KD family mean over three seeds is 71.00 ± 1.09 F1. The teacher is
86.92 / 79.68.

### `pretrained/` — width-1.0 ImageNet-pretrained fine-tune

| File | What it is | Size |
|---|---|---:|
| `best_model.pt` | Width-1.0 pretrained fine-tune (teacher architecture) | 2,328,952 B |
| `gesture_ssdlite.onnx` | Static ONNX export | 2,185,963 B |
| `gesture_ssdlite.espdl` | PTQ-INT8 re-export without test tensors (paper table) | 658,464 B |
| `gesture_ssdlite.json` | Metadata for the re-export | 250,773 B |
| `gesture_ssdlite_qat.espdl` | QAT-INT8 (best epoch 10) | 658,464 B |
| `gesture_ssdlite_qat.json` | Metadata | 250,805 B |
| `gesture_ssdlite_with_test_tensors.espdl` | Earlier PTQ export with test tensors | 709,328 B |
| `gesture_ssdlite_with_test_tensors.json` | Metadata for the earlier export | 250,755 B |

Metrics: fp32 79.68 / 86.92; PTQ 69.75 / 80.22; QAT 75.06 / 83.94. The earlier
PTQ export with test tensors scores 70.44 / 80.80 and is 692.70 KB; the paper's
final table reports the 643.03 KB re-export.

`teacher_best.pt` in `kd/` is the copy used by the distillation notebook. The
two width-1.0 checkpoints have the same architecture and 563,327 state-dict
elements; they are kept separately because they are separate notebook outputs.
Do not assume they are bit-identical.

### `baseline/` — width-0.30 from-scratch baseline exports

| File | What it is | Size |
|---|---|---:|
| `gesture_ssdlite.onnx` | Width-0.30 baseline ONNX export | 366,694 B |
| `gesture_ssdlite.espdl` | PTQ-INT8 with embedded test tensors (paper table) | 238,624 B |
| `gesture_ssdlite.json` | Metadata | 250,612 B |
| `gesture_ssdlite_no_test.espdl` | PTQ-INT8 re-export without test tensors | 187,728 B |
| `gesture_ssdlite_no_test.json` | Metadata | 250,619 B |
| `gesture_ssdlite_qat.espdl` | QAT-INT8 (best epoch 9) | 187,760 B |
| `gesture_ssdlite_qat.json` | Metadata | 250,594 B |

Metrics: fp32 18.48 / 25.32; PTQ 13.86 / 20.13; QAT 17.32 / 24.22. **There is
no width-0.30 checkpoint in this folder.** The checkpoint from the original
baseline run was not preserved; only its exports survived. If you need a matched
width-0.30 checkpoint/export pair, use `ablations/scratch_0p30_s0.pt` (the
augmented seed-0 model, 60.81 F1) or retrain with `scripts/train.py --width-mult
0.30`. The unaugmented baseline export is not the same model as the augmented
seed checkpoints.

### `baseline_w0p50/` — width-0.50 from-scratch checkpoint

| File | What it is | Size |
|---|---|---:|
| `best_model.pt` | Width-0.50 baseline checkpoint | 723,768 B |

This is the width-0.50 checkpoint from the original baseline notebook. Its
matching ONNX/`.espdl` export was overwritten when the notebook's `WIDTH_MULT`
was changed to 0.30, so the folder contains only the checkpoint. To produce a
matched export:

```bash
python scripts/export_onnx.py --width-mult 0.50 \
  --checkpoint models/baseline_w0p50/best_model.pt \
  --out /tmp/baseline_w0p50.onnx
```

The `width_sweep/model_w0p5_best.pt` + `model_w0p5.onnx` pair is a **different**
width-0.50 run; use that pair when you need a published metric (25.87 F1, fp32,
unaugmented).

### `width_sweep/` — matched width sweep

Scratch models (`model_w*`) and KD students (`student_w*`) at the widths below.
Each row has a `.pt`, `.onnx`, and `.espdl` with the same width.

| Width | Scratch params | Scratch `.pt` / `.onnx` / `.espdl` | KD params | KD `.pt` / `.onnx` / `.espdl` |
|---:|---:|---|---:|---|
| 0.25 | 46,182 | 276,216 / 208,010 / 145,728 B | 46,182 | 278,312 / 208,010 / 147,120 B |
| 0.50 | 153,526 | 723,768 / 628,988 / 257,200 B | 153,526 | 725,674 / 628,988 / 257,824 B |
| 0.75 | 326,214 | 1,430,456 / 1,311,120 / 433,728 B | 326,214 | 1,432,552 / 1,311,120 / 434,368 B |
| 1.00 | 546,774 | 2,328,952 / 2,185,963 / 657,808 B | — | — |

Metrics (fp32 single seed, unaugmented; INT8 from the paper tables):

| Width | Scratch fp32 F1 / acc | Scratch INT8 F1 / acc | KD fp32 F1 / acc | KD INT8 F1 / acc |
|---:|---:|---:|---:|---:|
| 0.25 | 19.04 / 12.70 | 13.58 / 9.01 | 57.43 / 42.96 | 59.27 / 44.80 |
| 0.30 | 25.32 / 18.48 | 20.13 / 13.86 | 69.78 / 57.51 | 67.88 / 56.12 |
| 0.50 | 25.87 / 17.55 | 22.96 / 15.24 | 73.00 / 62.12 | 70.32 / 58.66 |
| 0.75 | 36.48 / 27.94 | 36.35 / 27.48 | 81.33 / 72.98 | 79.95 / 70.90 |
| 1.00 | 44.58 / 33.26 | 42.48 / 31.64 | — | — |

The 0.30 scratch row is the overfit unaugmented run; the width sweep is a shape
reference, not the fair training comparison. The three-seed augmented family is
in `ablations/`.

### `ablations/` — seed and ablation checkpoints

| File | Width | F1 / acc | Note |
|---|---:|---:|---|
| `scratch_0p30_s0.pt` | 0.30 | 60.81 / 48.27 | augmented scratch, seed 0 |
| `scratch_0p30_s1.pt` | 0.30 | 58.52 / 45.03 | augmented scratch, seed 1 |
| `scratch_0p30_s2.pt` | 0.30 | 59.78 / 46.19 | augmented scratch, seed 2 |
| `scratch_0p30_unaug_s1.pt` | 0.30 | 24.26 / 17.55 | no augmentation, seed 1 |
| `scratch_0p30_unaug_s2.pt` | 0.30 | 29.91 / 21.71 | no augmentation, seed 2 |
| `kd_0p30_aug_s1.pt` | 0.30 | 71.36 / 58.66 | KD, seed 1 |
| `kd_0p30_aug_s2.pt` | 0.30 | 71.87 / 59.82 | KD, seed 2 |
| `scratch_1p00_aug_s0.pt` | 1.00 | 78.61 / 69.05 | augmented scratch, seed 0 |

All width-0.30 `.pt` files are 447,672 B except `kd/ab_A_unaug_s0.pt` (448,434 B);
the width-1.0 file is 2,328,952 B. The unaugmented seed-0 scratch model is
`ab_A_unaug_s0.pt` in `kd/` (32.70 / 23.33), because it was produced by the KD
notebook's ablation arm.

### `calibration/` — KD-0.30 QAT calibration sweep

| File | Calibration steps | Size | F1 / acc |
|---|---:|---:|---:|
| `student_calib50.espdl` | 50 | 183.98 KB | 67.95 / 56.35 |
| `student_calib100.espdl` | 100 | 183.98 KB | 68.06 / 55.89 |
| `student_calib433.espdl` | 433 | 183.98 KB | 67.39 / 54.97 |

The 200-step main PTQ export is `kd/gesture_ssdlite.espdl` (232.98 KB, includes
test tensors); its 183.33 KB re-export is `kd/gesture_ssdlite_no_test.espdl`.
F1 varies by <0.7 points from 50 to 433 calibration images, within evaluation
noise.

## Parameter counts vs checkpoint size

`SSDLiteGestureDetector.param_summary()` counts **trainable parameters**:
46,182 / 86,870 / 153,526 / 326,214 / 546,774 for widths 0.25 / 0.30 / 0.50 /
0.75 / 1.00. A saved `state_dict` also contains BatchNorm running means,
variances and `num_batches_tracked`, so the number of tensor elements is larger:

| Width | Trainable params | State-dict tensor elements |
|---:|---:|---:|
| 0.25 | 46,182 | 50,559 |
| 0.30 | 86,870 | 93,199 |
| 0.50 | 153,526 | 162,031 |
| 0.75 | 326,214 | 339,055 |
| 1.00 | 546,774 | 563,327 |

The `.pt` file size also includes pickle metadata. Do not use the file size as a
parameter count.

## Loading and verification

```python
import torch
from tinygesturessd.model import SSDLiteGestureDetector

model = SSDLiteGestureDetector(width_mult=0.30)
state = torch.load("models/kd/best_model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)          # strict=True by default
model.eval()
print(model.param_summary())          # 86,870
```

The ONNX graph for the same checkpoint has input `[1,3,128,128]` and outputs
`cls_logits [1,128,7]`, `loc_preds [1,128,4]`. The test suite checks the
released KD checkpoint against its ONNX export with `atol=1e-4`.

Verify the manifest from the repository root:

```bash
# Linux/macOS
sha256sum -c models/SHA256SUMS

# Windows PowerShell
Get-Content models/SHA256SUMS | ForEach-Object {
  $h, $p = $_ -split '  ', 2
  if ((Get-FileHash $p -Algorithm SHA256).Hash.ToLower() -ne $h) { Write-Error $p }
}
```

## What is not here

- The HaGRID dataset (including the 106.7 MB Arrow shard) is not included.
- The MediaPipe `hand_landmarker.task` is downloaded on first use.
- The ESP-PPQ `.info` and test `.npy` files are not included; the `.json`
  metadata and `.espdl` models are.
- The width-sweep QAT files and per-seed ONNX/`.espdl` exports are not included
  (only the checkpoints and the main KD/baseline/pretrained exports). The
  notebooks can regenerate them.

The original project trees and the private `XareDev/edge-ai-based-cnn`
repository were not modified.
