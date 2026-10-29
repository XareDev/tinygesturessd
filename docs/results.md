# Results

All accuracy/F1 numbers are on the fixed 433-image validation split unless
stated otherwise. F1 is macro-F1 over the six gesture classes. A missed
detection counts as a wrong prediction, so the numbers combine detection recall
with gesture classification; box localization is not scored. `±` is the sample
standard deviation over training seeds. The fp32 scores are recomputed from the
saved checkpoints; INT8 scores use ESP-PPQ's host-side reference executor.

The paper's main comparison is the width-0.30 family (86,870 trainable
parameters), because that is the deployable student. The width-1.0 pretrained
model is the teacher and the accuracy reference.

## 1. Width sweep (fp32, single seed, no augmentation)

This is the shape/capacity baseline, not the fair training comparison: the
width-0.30 run memorized the training set (training loss ≈ 0.02, validation
stuck), which is why its 25.32 F1 is an outlier.

| `width_mult` | Params | F1 macro (%) | Accuracy (%) |
|---:|---:|---:|---:|
| 0.25 | 46,182 | 19.04 | 12.70 |
| 0.30 | 86,870 | 25.32 | 18.48 |
| 0.50 | 153,526 | 25.87 | 17.55 |
| 0.75 | 326,214 | 36.48 | 27.94 |
| 1.00 | 546,774 | 44.58 | 33.26 |

The paired bootstrap says the 0.25→0.30 F1 step is unresolved
(`+6.28 [−0.22, +12.74]`), 0.30→0.50 is flat (`+0.55 [−5.41, +6.43]`), and the
larger steps are resolved (0.50→0.75 `+10.61 [+4.38, +16.76]`, 0.75→1.00
`+8.10 [+1.97, +14.21]`).

## 2. Training-regime comparisons

| Model | Width | Params | F1 macro (%) | Accuracy (%) | Seeds |
|---|---:|---:|---:|---:|---|
| Scratch, no aug, single seed | 0.30 | 86,870 | 25.32 | 18.48 | 1 |
| Scratch, aug | 0.30 | 86,870 | **59.70 ± 1.15** | **46.50 ± 1.64** | 3 (60.81, 58.52, 59.78) |
| Scratch, no aug | 0.30 | 86,870 | 28.96 ± 4.30 | — | 3 (32.70, 24.26, 29.91) |
| KD student | 0.30 | 86,870 | **69.78** | **57.51** | 1 (released) |
| KD student | 0.30 | 86,870 | **71.00 ± 1.09** | — | 3 (69.78, 71.36, 71.87) |
| Scratch, aug | 1.00 | 546,774 | 78.61 | 69.05 | 1 |
| ImageNet-pretrained fine-tune | 1.00 | 546,774 | **86.92** | **79.68** | 1 (teacher) |

The clean training effect is the three-seed width-0.30 comparison:
**59.70 → 71.00 F1 (+11.30 points)** at the same parameter count, which recovers
about 42% of the 27.22-point gap to the 1.0 teacher. Pretraining at width 1.0 is
**+8.31 F1** over the augmented scratch model (78.61 → 86.92).

The paper's revision table also reports a `+44.47 [+38.74, +50.02]` "distillation
effect". That compares the single-seed KD student (69.78) with the single-seed
**unaugmented** scratch model (25.32), not with the augmented 3-seed baseline.
Both numbers are in the repository; use the 59.70 → 71.00 comparison when the
question is "what does distillation add over a properly trained scratch model?"

## 3. PTQ and QAT

| Model | Size (KB) | Accuracy (%) | Precision (%) | Recall (%) | F1 macro (%) |
|---|---:|---:|---:|---:|---:|
| KD student, fp32 | 446.31 | 57.51 | 90.11 | 57.42 | 69.78 |
| KD student, PTQ-INT8 | 232.98 | 56.12 | 87.66 | 56.04 | 67.88 |
| KD student, PTQ-INT8 (no test tensors) | 183.33 | 56.12 | — | — | 67.88 |
| KD student, QAT-INT8 | 183.31 | 57.04 | 88.95 | 57.00 | 69.20 |
| Pretrained 1.0, fp32 | 2,283.50 | 79.68 | 95.73 | 79.82 | 86.92 |
| Pretrained 1.0, PTQ-INT8 | 643.03 | 69.75 | 94.82 | 70.59 | 80.22 |
| Pretrained 1.0, QAT-INT8 | 643.03 | 75.06 | 95.07 | 75.24 | 83.94 |
| Baseline 0.30, fp32 | 446.31 | 18.48 | 42.19 | 18.56 | 25.32 |
| Baseline 0.30, PTQ-INT8 | 233.03 | 13.86 | 36.89 | 13.98 | 20.13 |
| Baseline 0.30, QAT-INT8 | 183.36 | 17.32 | 40.04 | 17.42 | 24.22 |

Paired differences (percentage points, 95% CI):

| Comparison | ΔF1 | Resolved? |
|---|---|---:|
| PTQ − fp32 (KD 0.30) | −1.90 [−4.44, +0.58] | no |
| QAT − PTQ (KD 0.30) | +1.32 [−0.40, +3.08] | no |
| QAT − fp32 (KD 0.30) | −0.59 [−3.13, +1.93] | no |
| PTQ − fp32 (pretrained 1.0) | −6.71 [−9.76, −3.82] | yes |
| QAT − PTQ (pretrained 1.0) | +3.73 [+1.46, +6.10] | yes |
| QAT − fp32 (pretrained 1.0) | −2.98 [−5.66, −0.41] | yes |
| PTQ − fp32 (scratch 0.30) | −5.19 [−9.62, −0.59] | yes |
| QAT − PTQ (scratch 0.30) | +4.09 [+0.37, +7.75] | yes |
| QAT − fp32 (scratch 0.30) | −1.10 [−5.55, +3.43] | no |

At width 0.30, PTQ and QAT are indistinguishable on this validation set; QAT is
not worth the extra engineering for the student. At width 1.0, QAT recovers
about half of the PTQ loss, but still trails fp32 by ~3 F1 points.

## 4. Calibration-set size (KD student, PTQ)

| Calibration steps | Size (KB) | F1 macro (%) | Accuracy (%) |
|---:|---:|---:|---:|
| 50 | 183.98 | 67.95 | 56.35 |
| 100 | 183.98 | 68.06 | 55.89 |
| 200 | 232.98 | 67.88 | 56.12 |
| 433 | 183.98 | 67.39 | 54.97 |

F1 varies by <0.7 points across an 8.7× range. The 200-step row is the main PTQ
student and is larger because that export embeds test tensors. The supported
reading is that 50–100 calibration images already match the full set within
evaluation noise; more calibration data is not the lever here.

## 5. Paired bootstrap intervals

`results/metrics/bootstrap_summary.md` is generated from
`bootstrap_results.json` with 10,000 paired resamples of the 433 images. The
intervals capture evaluation-set sampling noise only, not training-seed variance.
Selected per-model intervals:

| Model | F1 % [95% CI] | Accuracy % [95% CI] |
|---|---|---|
| fp32 scratch 0.30 | 25.32 [20.54, 29.81] | 18.48 [14.78, 22.17] |
| fp32 KD 0.30 | 69.78 [65.34, 73.64] | 57.51 [52.66, 62.12] |
| fp32 pretrained 1.0 | 86.92 [83.99, 89.53] | 79.68 [75.75, 83.37] |
| PTQ KD 0.30 | 67.88 [63.43, 71.93] | 56.12 [51.50, 60.74] |
| QAT KD 0.30 | 69.20 [64.78, 73.09] | 57.04 [52.42, 61.66] |
| PTQ pretrained 1.0 | 80.22 [76.59, 83.40] | 69.75 [65.36, 74.13] |
| QAT pretrained 1.0 | 83.94 [80.76, 86.83] | 75.06 [70.90, 78.98] |

The full table, including every paired comparison, is in
[`../results/metrics/bootstrap_summary.md`](../results/metrics/bootstrap_summary.md).

## 6. On-device latency

ESP32-S3, 8 MB octal PSRAM, 16 MB flash, native USB-Serial-JTAG, 30 repeated
inferences per row on one fixed image:

| Model | Quant. | Inference (ms) | Round trip (ms) | Overhead share |
|---|---|---:|---:|---:|
| Scratch 0.30 | PTQ | 94.05 ± 0.24 | 190.51 ± 0.67 | 50.63% |
| Scratch 0.30 | QAT | 94.05 ± 0.24 | 191.32 ± 1.05 | 50.84% |
| KD student 0.30 | PTQ | 94.11 ± 0.22 | 191.32 ± 0.92 | 50.81% |
| KD student 0.30 | QAT | 94.10 ± 0.22 | 190.95 ± 0.63 | 50.72% |
| Pretrained 1.0 | PTQ | 468.35 ± 0.38 | 564.88 ± 0.73 | 17.09% |
| Pretrained 1.0 | QAT | 468.90 ± 0.37 | 565.95 ± 1.10 | 17.15% |

Inference latency tracks model size (~5× from 0.30 to 1.0), not quantization
scheme: PTQ and QAT produce the same INT8 execution graph. The fixed transfer
and protocol cost is ~96–97 ms in every row, so it consumes about half the round
trip for the small model and 17% for the large one. At width 0.30, optimizing
the transport is as valuable as shrinking the model.

## 7. Reproducing and reading the numbers

- The validation split is seed-0, fixed, and reused for checkpoint selection,
  PTQ calibration, and QAT epoch selection. Absolute scores are mildly
  optimistic; there is no separate test set.
- The 0.5 confidence threshold is a hard boundary. Re-scoring a checkpoint in a
  fresh process can move one image (≈0.2 accuracy points); the width-0.75 and
  1.0 rows differ from their training-time logs by 0.08 and 0.27 F1.
- Most rows are single-seed. The three-seed families are width-0.30 scratch
  (augmented and unaugmented) and width-0.30 KD; only those support seed-variance
  claims.
- The QAT numbers are best-of-10-epochs selected on the validation split, so the
  QAT recovery is optimistic. At width 0.30 the QAT gain is within noise; do not
  claim a recovery ratio there.
- INT8 accuracy is host-side, not read from the device. Only the latency table
  is on-device. The paper's limitations section spells out the consequences.
