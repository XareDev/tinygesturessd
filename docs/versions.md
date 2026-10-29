# Versions, variants and provenance

This project was developed as a sequence of Colab notebooks. The notebooks are
preserved with their outputs; the `tinygesturessd/` package and `scripts/` are a
cleaned local refactor of the same training path. This file maps each notebook
and model folder to the experiment it produced, so the repository does not need
duplicated branches.

## Notebooks

| Notebook | Role | Outputs |
|---|---|---|
| `01_baseline_ssdlite.ipynb` | From-scratch SSDLite baseline; width sweep; PTQ/QAT at width 0.30. The saved notebook currently sets `WIDTH_MULT = 0.3`; the package default is 0.50. | `models/` (0.30 exports), `models_width_sweep/` (0.25/0.50/0.75/1.00), baseline figures/tables |
| `02_knowledge_distillation.ipynb` | ImageNet-pretrained width-1.0 teacher → width-0.30 student; KD with `T=2.0`, `alpha=0.5` (soft classification), `beta=0.5` (soft localization); 120 student epochs. | `models_kd/` (student checkpoint/ONNX/PTQ/QAT, teacher checkpoint), KD figures |
| `03_pretrained_backbone.ipynb` | Width-1.0 ImageNet-pretrained fine-tune (teacher); 60 epochs; PTQ/QAT at width 1.0. | `models_pretrained/`, pretrained figures/tables |
| `04_revision_seed_and_quantization.ipynb` | Revision experiments: three augmented/unaugmented width-0.30 scratch seeds, two extra KD seeds, scratch-1.0 augmented, paired bootstrap, PTQ re-export without test tensors, calibration-size sweep, weight audit. | `models_seeds/`, `models_calib_sweep/`, `revision_results/`, `results/metrics/`, `results/tables/` |

The notebook outputs are not a substitute for a fresh run: several cells use
Colab paths, and the notebook code has more variants than the package scripts.
The package is the tested path for baseline training, export and inference; the
KD and QAT implementations remain in the notebooks.

## Model folders

| Folder | Width | What is there | Checkpoint? | Exports? |
|---|---:|---|---|---|
| `models/kd/` | 0.30 / 1.00 | Released KD student + teacher | `best_model.pt` (student), `teacher_best.pt` (teacher) | student ONNX, PTQ (with/without test tensors), QAT |
| `models/pretrained/` | 1.00 | ImageNet-pretrained fine-tune | `best_model.pt` | ONNX, PTQ re-export (643.03 KB), old PTQ with test tensors (692.70 KB), QAT |
| `models/baseline/` | 0.30 | From-scratch baseline exports | **no** (the 0.30 checkpoint was not preserved) | ONNX, PTQ (with/without test tensors), QAT |
| `models/baseline_w0p50/` | 0.50 | From-scratch baseline checkpoint | `best_model.pt` | **no matching export** (overwritten when the notebook moved to 0.30) |
| `models/width_sweep/` | 0.25 / 0.50 / 0.75 / 1.00 | Matched scratch and KD sweep artifacts | `model_w*_best.pt`, `student_w*_best.pt` | ONNX + `.espdl` |
| `models/ablations/` | 0.30 / 1.00 | Seed/ablation checkpoints | `scratch_0p30_*`, `kd_0p30_*`, `scratch_1p00_aug_s0.pt` | no (checkpoints only) |
| `models/calibration/` | 0.30 | KD QAT calibration sweep | no | `.espdl` for 50 / 100 / 433 calibration steps |

### Why `baseline` and `baseline_w0p50` are separate

The original `models/` folder contained a width-0.50 checkpoint
(`best_model.pt`) alongside width-0.30 ONNX/`.espdl` files. The notebook's
`WIDTH_MULT` was changed from 0.50 to 0.30 between runs, and the export cells
overwrote the ONNX/`.espdl` outputs while the checkpoint stayed at 0.50. Rather
than present a mismatched folder, the two artifacts are split:

- `models/baseline/` is the width-0.30 **export** used in the paper's baseline
  row (fp32 18.48% accuracy / 25.32% F1). Its checkpoint is not preserved.
- `models/baseline_w0p50/` is the width-0.50 **checkpoint** from the earlier
  run. Its matching export is not preserved.
- `models/width_sweep/` contains a *different* width-0.50 run with a matched
  checkpoint and export, which is the row used in the width-sweep table.

Do not pair `baseline_w0p50/best_model.pt` with `baseline/gesture_ssdlite.onnx`;
they are not the same model. Use `width_sweep/model_w0p5_best.pt` +
`width_sweep/model_w0p5.onnx` for a matched 0.50 pair, or re-export the baseline
0.50 checkpoint with `scripts/export_onnx.py --width-mult 0.50`.

## PTQ exports and embedded test tensors

ESP-PPQ can embed test input/output tensors in the `.espdl` for debugging. That
adds about 49.7 KB and does not change the number of integer weights. The paper
reports two kinds of file:

- `models/kd/gesture_ssdlite.espdl` (232.98 KB) and the baseline
  `models/baseline/gesture_ssdlite.espdl` (233.03 KB) include the test tensors,
  matching the paper's PTQ rows.
- `models/kd/gesture_ssdlite_no_test.espdl` (183.33 KB) and
  `models/baseline/gesture_ssdlite_no_test.espdl` (183.33 KB) are the re-exports
  without test tensors; they score the same as their PTQ counterparts.
- The pretrained folder's default `gesture_ssdlite.espdl` is the re-export
  (643.03 KB), matching the paper's final table; the earlier 692.70 KB export is
  kept as `gesture_ssdlite_with_test_tensors.espdl`.

QAT exports omit test tensors, so their sizes are directly comparable to the
`_no_test` PTQ files.

## Firmware and hardware

No TinyGesture firmware application is included in this repository. The paper's
latency measurements (94 ms at width 0.30, 469 ms at width 1.0) were taken on an
ESP32-S3 with a USB-Serial-JTAG harness that uploads a pre-quantized INT8 image
and parses the model output. The deployable artifact is the `.espdl` file; the
integration contract is documented in [`deployment.md`](deployment.md).

The sibling `AgroStake` repository contains a full ESP-IDF reference application
(UART, USB-Serial-JTAG and Wi-Fi variants) for a different four-class plant-disease
model. It is useful as a firmware pattern, but its model, preprocessing and class
order are not interchangeable with TinyGestureSSD.

## Dataset

The HaGRID images are **not** included. The training code downloads
`cj-mills/hagrid-sample-30k-384p` and pseudo-labels the boxes with MediaPipe.
The split is seed-0, 3,906 train / 433 validation; the validation label MD5 is
`d210029e2814357ca28251390535186e`. The dataset card and HaGRID terms apply to
the images; the MIT/CC licenses in this repository cover the code and the
author-created documentation/models, not the HaGRID data.

## What changed in the public copy

- Colab paths and secrets were replaced by local paths and environment checks.
- The package was factored into `config`, `model`, `anchors`, `matching`,
  `losses`, `training`, `dataset`, `inference`, and `export`.
- `scripts/train.py`, `scripts/export_onnx.py` and `scripts/demo.py` gained
  explicit CLI flags (`--width-mult`, `--checkpoint`, `--out`, etc.).
- `scripts/train.py` writes `training_summary.json` so a run is self-describing.
- `model.param_summary(verbose=False)` returns the parameter count for tests.
- The ONNX and PyTorch outputs are checked numerically in
  `tests/test_smoke.py`.
- No model weights, figures or notebook outputs were altered; only metadata,
  paths and the new public documentation were added.

The original private repository (`XareDev/edge-ai-based-cnn`) and the local
project trees were left untouched.
