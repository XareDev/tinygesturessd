# Notebooks

These are the four original experiment notebooks, preserved with their outputs.
They are the provenance for the paper and the only place where the full KD, QAT,
seed-sweep and calibration code lives. The importable package in
`../tinygesturessd/` and the scripts in `../scripts/` are a cleaned, tested
refactor of the baseline path — they are the recommended way to run the model,
but they do not implement KD or QAT.

| Notebook | Purpose | Runtime | Outputs in this repo |
|---|---|---|---|
| `01_baseline_ssdlite.ipynb` | From-scratch SSDLite; PTQ/QAT at width 0.30; width sweep. | Long (dataset download + MediaPipe + training + quantization) | `models/`, `models/width_sweep/`, baseline figures/tables |
| `02_knowledge_distillation.ipynb` | Width-1.0 ImageNet teacher → width-0.30 student; KD; PTQ/QAT; KD width sweep; KD ablation; calibration sensitivity. | Very long (teacher + student training + QAT) | `models/kd/`, `models/ablations/`, `models/calibration/`, KD figures/tables |
| `03_pretrained_backbone.ipynb` | Width-1.0 ImageNet-pretrained fine-tune; PTQ/QAT. | Long | `models/pretrained/`, pretrained figures/tables |
| `04_revision_seed_and_quantization.ipynb` | Camera-ready revisions: re-evaluation, paired bootstrap, three-seed families, PTQ re-exports, calibration sweep, weight audit. | Very long; some cells are intentionally skip-if-cached | `results/metrics/`, `results/tables/`, `results/figures/revision/` |

## Running them

The notebooks were developed in Jupyter with a local kernel. They expect the
**project root** as the working directory (the folder that contains `dataset/`,
`models/`, etc.). The original project used sibling folders such as
`models_kd/`, `models_pretrained/`, `models_width_sweep/`, `models_seeds/` and
`revision_results/`; this repository consolidates their released artifacts under
`models/` and `results/`. To re-run a notebook end-to-end, either recreate those
sibling paths or edit the path constants near the top of the notebook.

Install the notebook stack separately from the package:

```bash
pip install jupyterlab ipykernel
pip install -r ../requirements.txt
# torch/torchvision: install the wheel for your CUDA driver first
# quantization cells additionally need ESP-PPQ:
pip install esp-ppq
```

The first baseline/pretrained run downloads
`cj-mills/hagrid-sample-30k-384p` and the MediaPipe `hand_landmarker.task`, then
pseudo-labels boxes. That is the slow part; the result is cached under
`models/hagrid_pseudo_labeled/` (not shipped). The notebooks do not download
anything during import.

## Important caveats

- **Outputs are preserved, paths may be stale.** The notebook outputs include
  absolute paths from the original machine (`c:\Users\...`) and, in a few
  cells, Colab-era wording. The outputs are evidence, not a portable config.
- **Notebook 01 currently sets `WIDTH_MULT = 0.3`.** The package default is
  0.50. The notebook's export cells overwrote the original width-0.50 ONNX /
  `.espdl` files; see `../models/README.md` for the `baseline` vs
  `baseline_w0p50` split.
- **Notebook 02 uses a pre-trained teacher checkpoint.** It loads
  `models_kd/teacher_best.pt`; in this repository that file is
  `../models/kd/teacher_best.pt`. The student checkpoint is
  `../models/kd/best_model.pt` (width 0.30).
- **Notebook 04 is a revision log, not a tidy pipeline.** It contains verbatim
  re-definitions, duplicate helper cells, and skip-if-cached blocks. Do not
  "Run All" from a clean state and expect a single deterministic pass; read the
  markdown around each block. Its purpose is to document exactly how the
  camera-ready numbers were obtained.
- **The width-0.30 unaugmented scratch checkpoint is missing.** Notebook 04
  sets `SCRATCH_030_PT = None` and re-scores the surviving ONNX export. The
  3-seed augmented checkpoints are in `../models/ablations/`.
- **QAT is expensive and nondeterministic across ESP-PPQ versions.** The
  released QAT models were produced with ESP-PPQ/ESP-DL 3.3.2. Re-running QAT on
  a different version can produce different weights and sizes even with the same
  seed. The released `.espdl` files are the reference.
- **No dataset is included.** The notebooks download HaGRID and MediaPipe
  themselves; their licenses are separate from this repository's MIT/CC terms.

## Suggested reading order

1. `01_baseline_ssdlite.ipynb` sections 1–9 for the architecture, anchors,
   matching, loss and inference.
2. `03_pretrained_backbone.ipynb` for the ImageNet initialization experiment.
3. `02_knowledge_distillation.ipynb` for the KD loss, teacher/student training,
   and QAT.
4. `04_revision_seed_and_quantization.ipynb` for the seed families, bootstrap,
   PTQ re-exports and calibration sweep.

The code in the package mirrors sections 1–10 of notebook 01 (plus the
pretrained flag); see `../docs/architecture.md` for the exact tensor contract.
