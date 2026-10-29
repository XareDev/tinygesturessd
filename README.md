# TinyGestureSSD

**Single-scale SSDLite hand-gesture detection for the ESP32-S3, with an end-to-end
PyTorch → ONNX → ESP-PPQ INT8 → ESP-DL pipeline.**

TinyGestureSSD detects one hand and classifies its gesture into six classes
(`fist`, `palm`, `like`, `ok`, `peace`, `point`) on a 128×128 RGB frame. The
backbone is a truncated MobileNetV2 (`features[:14]`, stride 16, 8×8 feature map)
with two depthwise-separable SSDLite heads, 2 anchors per cell (128 anchors total)
and no NMS — the deployment target assumes at most one hand per frame.

The repository contains the training package, the four original experiment
notebooks, released checkpoints and INT8 `.espdl` models, a complete JTAG
reference firmware, the paper source and figures, and the measured results. It
is the artifact companion to:

> **Deploying Lightweight Vision Models on the ESP32-S3: A Case Study in the
> ESP-DL Inference Pipeline**
> Youcef Harmouni, Abdelaziz Bellili, Walid Touzout — IGEE (ex. INELEC), 2026.
> Paper PDF: [`docs/paper/paper.pdf`](docs/paper/paper.pdf)

[![License: MIT](https://img.shields.io/badge/code-MIT-yellow.svg)](LICENSE)
[![Docs: CC BY-NC-SA 4.0](https://img.shields.io/badge/docs%20%26%20data-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE-DOCS)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

---

## Headline results

All numbers are on the 433-image held-out validation split, fp32 unless marked
INT8. F1 is macro-F1 over the six gesture classes; a missed detection counts as a
wrong prediction. `±` is the sample standard deviation over three training seeds.
The paper's main comparison is the width-0.30 family (86,870 trainable
parameters), which is the deployment target.

| Model | Width | Params | F1 macro (%) | Accuracy (%) | Notes |
|---|---:|---:|---:|---:|---|
| Scratch, no augmentation | 0.30 | 86,870 | 25.32 | 18.48 | single seed; overfit, excluded from step comparisons |
| Scratch, augmented, 3 seeds | 0.30 | 86,870 | **59.70 ± 1.15** | **46.50 ± 1.64** | 60.81 / 58.52 / 59.78 F1 |
| Knowledge-distilled student | 0.30 | 86,870 | **69.78** | **57.51** | single seed, the released KD checkpoint |
| Knowledge-distilled, 3 seeds | 0.30 | 86,870 | **71.00 ± 1.09** | — | 69.78 / 71.36 / 71.87 F1 |
| ImageNet-pretrained fine-tune | 1.00 | 546,774 | **86.92** | **79.68** | teacher; single seed |
| Scratch, augmented | 1.00 | 546,774 | 78.61 | 69.05 | single seed |
| KD student, PTQ-INT8 | 0.30 | 86,870 | 67.88 | 56.12 | host-side INT8 reference executor |
| KD student, QAT-INT8 | 0.30 | 86,870 | 69.20 | 57.04 | 183.3 KB `.espdl` |

Paired bootstrap 95% confidence intervals (10,000 resamples of the 433 images,
evaluation-set noise only) put the KD-0.30 F1 at **[65.34, 73.64]**. The
three-seed comparison is the one to quote for the training effect:
**59.70 → 71.00 F1 (+11.3 points)** at the same parameter count. The larger
`+44.47` paired difference reported in the paper's revision table compares the
single-seed KD student (69.78) against the single-seed *unaugmented* scratch
model (25.32), not against the augmented 3-seed baseline.

Pretraining isolates cleanly at width 1.0: the same architecture and data move
from 78.61 to 86.92 F1 (**+8.31 points**) with only the initialization and
schedule changed. Quantization at width 0.30 is close to lossless: PTQ costs
1.90 F1 and QAT is 1.32 F1 above PTQ, both within the paired bootstrap noise. At
width 1.0, QAT recovers about half of the PTQ loss (80.22 → 83.94 F1, still
−2.98 vs fp32).

On-device latency over native USB-Serial-JTAG (paper, single fixed test image,
30 repeats): **94 ms at width 0.30** and **469 ms at width 1.0**. Fixed host
transfer/protocol overhead is about 51% of the round trip for the small model
and 17% for the large one. See [`docs/results.md`](docs/results.md) for the full
tables and [`results/metrics/bootstrap_summary.md`](results/metrics/bootstrap_summary.md)
for every confidence interval.

---

## What is in this repository

```
tinygesturessd/          importable training package (model, anchors, matching, loss, training, export)
scripts/                 train.py, export_onnx.py, demo.py
notebooks/               four original Colab notebooks, with outputs (provenance)
models/                  released checkpoints, ONNX, ESP-DL .espdl, SHA256SUMS
results/                 figures, metrics JSON, bootstrap summary, LaTeX tables
firmware/jtag/           ESP-IDF USB-Serial-JTAG reference app + host benchmark
docs/                    paper source + PDF, architecture/deployment/versions/results/references
tests/                   fast model/ONNX parity tests (no dataset download)
```

**Not included:** the HaGRID images, the MediaPipe hand-landmarker task file, and
the ESP-PPQ/ESP-IDF toolchains. The training code downloads HaGRID and MediaPipe
on first use; the `.espdl` models are included so you do not need ESP-PPQ to
inspect or deploy the released artifacts. The firmware source is included but
is not compiled by the Python test suite.

The four notebooks are preserved as they were run, including their outputs and
some Colab-specific paths. They are useful for provenance and for the KD/QAT
recipes that are not in the package. The package is the cleaned, local,
script-driven path. See [`notebooks/README.md`](notebooks/README.md) for the
differences and caveats.

---

## Quick start

### 1. Environment

Python 3.9+ is recommended. Install the PyTorch build that matches your CUDA
driver first, then the package and its remaining dependencies:

```bash
git clone https://github.com/XareDev/tinygesturessd.git
cd tinygesturessd

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

# Example: CUDA 12.1. Use the index URL for your driver, or the CPU wheel.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[torch,dev]"
```

`requirements.txt` lists the non-torch dependencies if you prefer a plain
`pip install -r requirements.txt` after installing torch yourself.

### 2. Train the from-scratch baseline

The default is width 0.50 and 12 epochs, which is the quick baseline row. The
paper's width-0.30 runs use 120 epochs; the first run also downloads HaGRID and
runs MediaPipe over up to 800 images per class (the slow part; the result is
cached).

```bash
python scripts/train.py
python scripts/train.py --width-mult 0.30 --epochs 120 --batch-size 64 --seed 0
```

Useful flags: `--width-mult`, `--epochs`, `--batch-size`,
`--max-images-per-class`, `--seed`, `--lr`, `--weight-decay`,
`--selection-metric`, `--out-dir`, `--data-cache-dir`, `--force-rebuild`.
The script writes `best_model.pt`, `training_curves.png` and
`training_summary.json` to `models/` (or `--out-dir`).

The released KD recipe is in
[`notebooks/02_knowledge_distillation.ipynb`](notebooks/02_knowledge_distillation.ipynb):
a width-1.0 ImageNet-pretrained teacher is distilled into a width-0.30 student
with temperature 2.0, `alpha=0.5` on the soft-classification KL term and
`beta=0.5` on the soft-localization term, 120 student epochs.

### 3. Export and inspect

```bash
# Default: width 0.50, models/best_model.pt -> models/gesture_ssdlite.onnx
python scripts/export_onnx.py

# The released KD student (width 0.30)
python scripts/export_onnx.py \
  --width-mult 0.30 \
  --checkpoint models/kd/best_model.pt \
  --out models/kd/gesture_ssdlite.onnx

# Render predictions on validation images
python scripts/demo.py \
  --width-mult 0.30 \
  --checkpoint models/kd/best_model.pt \
  --out models/kd/demo_predictions.png --n 8
```

The ONNX graph is static `[1, 3, 128, 128]` with outputs
`cls_logits [1, 128, 7]` and `loc_preds [1, 128, 4]`; it is exported with
opset 13 and the legacy TorchScript exporter (`dynamo=False`) because ESP-PPQ
and ESP-DL expect fixed shapes.

### 4. Tests

```bash
pytest
```

The tests check parameter counts against the paper, forward shapes, anchor
geometry, and — when `onnxruntime` is installed — numerical parity between the
released KD checkpoint and its ONNX export. They do not download HaGRID.

---

## Released models

The model zoo is deliberately small enough to clone (all model files are well
under GitHub's 100 MB limit). Every file's SHA-256 is in
[`models/SHA256SUMS`](models/SHA256SUMS); [`models/README.md`](models/README.md)
has the per-folder provenance and exact file sizes.

| Folder | What it is | Width | Notes |
|---|---|---:|---|
| `models/kd/` | distilled student + teacher | 0.30 / 1.00 | student checkpoint, ONNX, PTQ/QAT `.espdl`, teacher checkpoint |
| `models/pretrained/` | ImageNet-pretrained fine-tune | 1.00 | checkpoint, ONNX, PTQ/QAT `.espdl` |
| `models/baseline/` | from-scratch baseline exports | 0.30 | ONNX/PTQ/QAT `.espdl`; the 0.30 checkpoint was not preserved |
| `models/baseline_w0p50/` | from-scratch baseline checkpoint | 0.50 | matching export was overwritten when the notebook moved to 0.30 |
| `models/width_sweep/` | matched scratch and KD sweeps | 0.25–1.00 | checkpoints + ONNX + `.espdl` for 0.25, 0.50, 0.75, 1.00 scratch and 0.25, 0.50, 0.75 KD |
| `models/ablations/` | seed/ablation checkpoints | 0.30 / 1.00 | augmented/unaugmented scratch and KD seeds |
| `models/calibration/` | KD-0.30 QAT calibration sweep | 0.30 | 50 / 100 / 433 calibration steps |

The `baseline` versus `baseline_w0p50` split is intentional: the original
notebook trained a width-0.50 checkpoint, then its `WIDTH_MULT` was changed to
0.30 and the export cells overwrote the ONNX/`.espdl` files. Both artifacts are
kept, with the width stated in the folder name. Use `models/width_sweep/` when
you need a checkpoint and export that are guaranteed to match.

---

## Training setup

- **Data.** HaGRID subset `cj-mills/hagrid-sample-30k-384p` (image + label only).
  The six classes are mapped from HaGRID's `fist`, `palm`, `like`, `ok`, `peace`,
  `one` (the last becomes `point`). Up to 800 images per class are capped.
  Ground-truth boxes are pseudo-labelled with MediaPipe's Hand Landmarker: the
  largest detected hand's box, padded by 10%. After filtering, the split is
  **3,906 train / 433 validation** (shuffled with seed 0; the validation label
  MD5 is in `results/metrics/split_fingerprint.json`).
- **Input.** 128×128 RGB, scaled to `[0,1]`; no ImageNet mean/std
  normalization, matching the INT8 deployment preprocessing (scale `2^7 = 128`
  on the `[0,1]` image, rounded and clipped to the INT8 range).
- **Anchors.** 8×8 cells, 2 aspect ratios (1.0, 1.8), base size 48 px, 128
  anchors. Centres are generated in row-major cell order with the aspect ratio
  changing fastest; the heads emit channels in the same order.
- **Matching/loss.** IoU ≥ 0.5 is positive, IoU < 0.4 is background, the gap is
  ignored; the single best-IoU anchor per image is always positive. The loss is
  Smooth L1 localization on positives plus cross-entropy with 3:1 hard-negative
  mining. No NMS at inference: threshold then one global argmax over anchors.
- **Optimization.** AdamW, lr 2e-3, weight decay 1e-4, cosine schedule, batch
  64, gradient clipping at 5.0. The paper's runs use 120 epochs (60 for the
  pretrained teacher); checkpoints are selected on validation macro-F1.
- **Augmentation.** Box-aware horizontal flip, scale and colour jitter for the
  three-seed augmented family. The original width-sweep rows are single-seed and
  unaugmented; that is why the 0.30 row (25.32 F1) is an overfit outlier.

The notebooks are the exact KD/QAT/seed-sweep recipes; `docs/versions.md` maps
each notebook and model folder to the variant it produced.

---

## Deployment notes

The released `.espdl` files are ESP-DL 3.3.2 models quantized with ESP-PPQ for
`esp32s3` at 8 bits. They are static-shape, batch-1 models. The complete
USB-Serial-JTAG reference application is in
[`firmware/jtag/`](firmware/jtag/README.md): it embeds the PTQ-INT8 student,
reads a 128×128 RGB frame over native USB, and reports the class plus separate
device-inference and transfer times. It is the source used for the paper's
protocol measurements.

A minimal ESP-DL integration looks like:

```cpp
// main/CMakeLists.txt: target_add_binary_data(... "gesture_ssdlite_no_test.espdl" BINARY)
extern const uint8_t model_data_start[] asm("_binary_gesture_ssdlite_no_test_espdl_start");
dl::Model model(reinterpret_cast<const char *>(model_data_start), fbs::MODEL_LOCATION_IN_FLASH_RODATA);
model.minimize();  // optional: release intermediate runtime buffers

// Input: 128x128 RGB uint8, scaled as [0,1]*128 and clipped to int8.
// Output: cls_logits [1,128,7] and loc_preds [1,128,4].
```

Use the same class order as training:
`fist`, `palm`, `like`, `ok`, `peace`, `point`. The first classification channel
is background. The paper's on-device measurement is 94 ms at width 0.30 and
469 ms at width 1.0, measured with `esp_timer` around `model.run()` and excluding
image transfer. See [`firmware/jtag/README.md`](firmware/jtag/README.md) for the
build, wiring and USB protocol, and [`docs/deployment.md`](docs/deployment.md)
for the full preprocessing/quantization recipe.

---

## Reproducibility and honest caveats

- Most rows are **single-seed**. The three-seed families are the width-0.30
  scratch/KD runs; the paper's bootstrap intervals quantify evaluation-set noise
  only, not training-seed variance.
- The validation set is 433 images, so one image is about 0.23 accuracy points.
  Small differences (especially PTQ vs QAT at width 0.30) should be read through
  the paired CIs in `results/metrics/bootstrap_summary.md`.
- For INT8 rows, accuracy is scored by the host-side reference executor, not by
  reading the device's serial output. fp32 rows are PyTorch evaluations.
  On-device latency is measured separately.
- The width-0.30 unaugmented scratch run memorized the 3,906 training images
  (training loss ≈ 0.02, validation stuck); it is included as the width-sweep
  row but excluded from the paper's step comparisons.
- The notebooks are preserved with their original outputs and some stale
  Colab paths. The package/scripts are the tested path; the notebooks are the
  provenance and the KD/QAT implementations.
- HaGRID is not redistributed. Check the HaGRID terms and the
  `cj-mills/hagrid-sample-30k-384p` dataset card before reusing the images.

---

## Citation

If you use this code, models or results, please cite the paper:

```bibtex
@misc{harmouni2026tinygesturessd,
  title  = {Deploying Lightweight Vision Models on the {ESP32-S3}:
            A Case Study in the {ESP-DL} Inference Pipeline},
  author = {Harmouni, Youcef and Bellili, Abdelaziz and Touzout, Walid},
  year   = {2026},
  note   = {Manuscript, Institute of Electrical \& Electronic Engineering
            (IGEE, ex. INELEC)},
  url    = {https://github.com/XareDev/tinygesturessd}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is included. The paper source
and PDF are in [`docs/paper/`](docs/paper/).

---

## License

- **Code** (`tinygesturessd/`, `scripts/`, `tests/`, `firmware/`): [MIT](LICENSE).
- **Documentation, paper source, figures, results and released model weights/metadata**:
  [CC BY-NC-SA 4.0](LICENSE-DOCS). The weights are shared for research and
  non-commercial use; contact the authors for other terms.
- **HaGRID images** are not included and remain under their own terms.

The MIT license applies to the code; the CC BY-NC-SA 4.0 license applies to the
written documentation, figures, and released model weights. If you need a
different license for a commercial deployment, contact the authors.

---

## Acknowledgements

HaGRID for the gesture data; MediaPipe for the hand-landmarker boxes; PyTorch and
torchvision for the training stack; Espressif's ESP-DL and ESP-PPQ teams for the
on-device runtime and quantization toolchain. The paper's figures and tables were
produced from the notebooks and metrics in this repository.
