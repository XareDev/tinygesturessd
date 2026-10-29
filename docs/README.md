# Documentation

This folder contains the paper and the technical notes that did not fit in the
top-level README.

| Document | What it covers |
|---|---|
| [`paper/paper.pdf`](paper/paper.pdf) | Camera-ready manuscript (PDF). |
| [`paper/paper.tex`](paper/paper.tex) | LaTeX source with the embedded bibliography; no `.bib` file is required. |
| [`paper/README.md`](paper/README.md) | How to rebuild the PDF and where the figures came from. |
| [`architecture.md`](architecture.md) | Tensor shapes, backbone truncation, anchors, heads, matching, loss, and post-processing. |
| [`deployment.md`](deployment.md) | ONNX → ESP-PPQ INT8 → ESP-DL, preprocessing contract, class order, and integration notes. |
| [`../firmware/jtag/README.md`](../firmware/jtag/README.md) | Build, wiring, USB protocol and host benchmark for the reference firmware. |
| [`versions.md`](versions.md) | Notebook/model/firmware variant map and provenance. |
| [`results.md`](results.md) | Full fp32/PTQ/QAT tables, paired bootstrap intervals, width sweep, calibration sensitivity, and latency. |
| [`references.md`](references.md) | Paper bibliography and links to the external tools/datasets. |

The top-level [`README.md`](../README.md) is the place to start. The
[`models/README.md`](../models/README.md) file is the authoritative per-file model
inventory; [`models/SHA256SUMS`](../models/SHA256SUMS) contains the checksums.
