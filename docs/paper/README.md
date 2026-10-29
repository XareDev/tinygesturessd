# Paper

This folder contains the manuscript and its figures.

| File | Description |
|---|---|
| `paper.pdf` | Compiled camera-ready PDF, included for convenience. |
| `paper.tex` | LaTeX source. Uses `IEEEtran` conference class. |
| `figs/` | The seven figures referenced by `paper.tex`. |

## Rebuilding the PDF

The bibliography is embedded with `thebibliography`, so there is no `.bib` file
and no BibTeX pass is needed. From this directory:

```bash
pdflatex paper.tex
pdflatex paper.tex     # second pass for references / page numbers
```

Requirements: a TeX distribution with `IEEEtran.cls` (e.g. TeX Live
`texlive-publishers` or MiKTeX's `ieeetran` package) and the packages listed in
the preamble (`cite`, `amsmath`, `amssymb`, `amsfonts`, `algorithmic`,
`graphicx`, `textcomp`, `xcolor`, `booktabs`, `multirow`, `url`).

The figure files are the same PNGs used in the version of record:

| Figure | File |
|---|---|
| Width/accuracy Pareto | `figs/fig_scratch_width_pareto.png` |
| KD size comparison | `figs/fig_kd_size_teacher_student.png` |
| KD fp32/PTQ/QAT metrics | `figs/fig_kd_fp32_ptq_qat.png` |
| KD confusion matrices | `figs/fig_confusion_student_qat.png` |
| Calibration sensitivity | `figs/fig_calib_sensitivity.png` |
| Weight audit | `figs/fig_weight_audit.png` |
| Demo predictions | `figs/fig_demo_predictions.png` |

The paper's numbers are generated from the notebooks and the JSON files in
[`../../results/metrics`](../../results/metrics). See
[`../results.md`](../results.md) and
[`../../results/metrics/bootstrap_summary.md`](../../results/metrics/bootstrap_summary.md)
for the machine-readable versions and the caveats.

## Authors

Youcef Harmouni, Abdelaziz Bellili, Walid Touzout — Institute of Electrical &
Electronic Engineering (IGEE, ex. INELEC), 2026.

## License

The paper source, PDF and figures are licensed under CC BY-NC-SA 4.0; see
[`../../LICENSE-DOCS`](../../LICENSE-DOCS). The code is MIT-licensed separately.
