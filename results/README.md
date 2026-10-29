# Results and measurements

This folder contains the figures, per-image predictions, metric JSON and LaTeX
tables produced by the notebooks. The narrative version with all caveats is in
[`../docs/results.md`](../docs/results.md); the paper is in
[`../docs/paper/`](../docs/paper/).

## Layout

```text
results/
├── figures/
│   ├── baseline/       notebook 01 figures (fp32 / PTQ / QAT / width sweep)
│   ├── kd/             notebook 02 figures (teacher/student, PTQ/QAT, calib)
│   ├── pretrained/     notebook 03 figures (width 1.0)
│   └── revision/       notebook 04 figures (seed ladder, memorization, weights)
├── metrics/
│   ├── fp32_sweep.json                 fp32 width-sweep scores
│   ├── seed_scratch030_s{0,1,2}.json   augmented 3-seed scratch family
│   ├── abl_*.json                      KD/scratch ablation and seed rows
│   ├── bootstrap_results.json          full paired-bootstrap output
│   ├── bootstrap_summary.md            human-readable bootstrap tables
│   ├── ptq_reexport.json               PTQ re-export sizes/scores
│   ├── split_fingerprint.json          fixed split identity
│   ├── preds/                          per-image predicted labels (.npy)
│   └── qat/                            QAT caches, calibration cache, preds
└── tables/
    ├── baseline/       fp32/PTQ/QAT + INT8 width sweep
    ├── kd/             teacher/student + INT8 width sweep + calibration
    └── pretrained/     fp32/PTQ/QAT at width 1.0
```

## What the numbers mean

- **Accuracy / precision / recall / F1** are computed over the 433-image
  validation split. A missed detection (no anchor above the 0.5 confidence
  threshold) is counted as a wrong prediction, so these are end-to-end
  detection+classification scores, not classification-only scores.
- **F1 is macro-F1** over the six gesture classes. The background class is not
  scored.
- **`preds/*.npy`** contain one predicted class id per validation image
  (`-1` for a miss); **`preds/gts.npy`** is the ground truth. These are the
  arrays behind `bootstrap_results.json`, so the bootstrap can be reproduced
  without re-running a model.
- **`qat/*.npy`** are the QAT prediction arrays; **`qat/*_result_cache.json`**
  records the per-epoch QAT metrics and the selected epoch.
- **`ptq_reexport.json`** compares the original PTQ `.espdl` (with ESP-PPQ test
  tensors) to the re-export without them. The scores are identical; only the
  container size changes (232.98 → 183.33 KB for the KD student, 692.70 →
  643.03 KB for the pretrained model).

## Split identity

`split_fingerprint.json`:

```json
{"val_label_md5": "d210029e2814357ca28251390535186e",
 "n_train": 3906, "n_val": 433, "seed": 0}
```

Any run that does not reproduce this MD5 is using a different split and its
numbers are not comparable to the paper. The validation set is also used for
checkpoint selection, PTQ calibration and QAT epoch selection; there is no
separate test set. Absolute scores are therefore mildly optimistic, and the
bootstrap intervals quantify evaluation-set noise only.

## Figure map

| Figure | Folder | What it shows |
|---|---|---|
| `fig1_metrics_fp32_vs_int8.png` | `baseline/` | fp32 vs PTQ-INT8 metrics |
| `fig2_model_size.png` | `baseline/` | fp32 vs INT8 size |
| `fig3_confusion_fp32.png`, `fig4_confusion_int8.png`, `fig9_confusion_qat.png` | `baseline/` | confusion matrices |
| `fig5_backbone_sqnr.png`, `fig6_backbone_mse.png` | `baseline/` | per-block activation sensitivity |
| `fig7_metrics_fp32_vs_qat.png`, `fig8_model_size_fp32_vs_qat.png` | `baseline/` | QAT pairwise |
| `fig11_metrics_fp32_vs_ptq_vs_qat.png`, `fig12_model_size_fp32_vs_ptq_vs_qat.png` | `baseline/` | three-way comparison |
| `fig13_width_mult_pareto.png` | `baseline/` | F1 vs params / size |
| `fig1_metrics_teacher_vs_student_fp32_vs_int8.png`, `fig2_model_size_teacher_vs_student.png` | `kd/` | teacher vs student |
| `fig3_confusion_teacher_fp32.png`, `fig4_confusion_student_fp32.png`, `fig5_confusion_student_int8.png`, `fig9_confusion_student_qat.png` | `kd/` | KD confusion matrices |
| `fig6_student_backbone_sqnr.png`, `fig6_student_backbone_mse.png` | `kd/` | student activation sensitivity |
| `fig7_metrics_fp32_vs_ptq_vs_qat.png`, `fig8_model_size_fp32_vs_ptq_vs_qat.png` | `kd/` | KD quantization |
| `fig10_width_mult_pareto.png` | `kd/` | KD width sweep |
| `fig12_calib_steps_sensitivity.png` | `kd/` | F1 vs calibration steps |
| `fig1_metrics_fp32_vs_int8.png` … `fig12_model_size_fp32_vs_ptq_vs_qat.png` | `pretrained/` | width-1.0 equivalents |
| `fig_baseline_ladder.png`, `fig_memorization_staircase.png`, `fig_weight_audit.png` | `revision/` | revision diagnostics |

The paper's seven figures are the curated copies in
[`../docs/paper/figs/`](../docs/paper/figs/). The folders here are the raw
notebook outputs, so some figures repeat across experiments with different
file names.

## LaTeX tables

`tables/baseline/table_width_mult_sweep.tex` and the matching KD table report
**INT8** scores; `table_fp32_ptq_qat_comparison.tex` in each folder reports
fp32/PTQ/QAT. `tables/kd/table_calib_steps_sensitivity.tex` reports raw
fractions (0–1), as labelled; multiply by 100 to compare with the README.
Captions and column names were kept close to what the notebooks emitted;
`docs/results.md` is the normalized narrative.

## Reproducing the bootstrap

```python
import json
import numpy as np

gts = np.load("results/metrics/preds/gts.npy")
preds = np.load("results/metrics/preds/fp32_kd_0.30.npy")
# paired resampling over the 433 images, 10,000 draws, percentile 95% CI
```

The exact script is in notebook 04 (the `bootstrap` cells). The stored
`bootstrap_results.json` includes every per-model interval and every pairwise
difference; `bootstrap_summary.md` is a readable rendering of it.

## Integrity

- `preds/*.npy` are small (a few KB each) and are committed intentionally so the
  statistics can be audited.
- `fp32_kd_0.750.y0..npy` was an accidental glob duplicate of
  `fp32_kd_0.75.npy`; it was removed from the public copy. No other
  prediction arrays were altered.
- No raw HaGRID images, MediaPipe task files or ESP-PPQ `.info`/test tensors are
  in this folder.
- Figures are the original PNG outputs; they were not re-rendered for the public
  copy.
