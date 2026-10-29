# References and external resources

The paper's bibliography is embedded in
[`paper/paper.tex`](paper/paper.tex) (`thebibliography`), so no `.bib` file is
required. This page repeats the links used by the code and documentation.

## Paper bibliography

- Espressif Systems, *ESP32-S3-WROOM-1 and ESP32-S3-WROOM-1U datasheet*, v1.4.
  <https://www.espressif.com/sites/default/files/documentation/esp32-s3-wroom-1_wroom-1u_datasheet_en.pdf>
- R. David et al., “TensorFlow Lite Micro: Embedded machine learning for TinyML
  systems,” MLSys, 2021.
- J. Lin et al., “MCUNet: Tiny deep learning on IoT devices,” NeurIPS, 2020.
- B. Jacob et al., “Quantization and training of neural networks for efficient
  integer-arithmetic-only inference,” CVPR, 2018.
- Espressif Systems, *ESP-DL*. <https://github.com/espressif/esp-dl>
- M. Sandler et al., “MobileNetV2: Inverted residuals and linear bottlenecks,”
  CVPR, 2018.
- W. Liu et al., “SSD: Single shot multibox detector,” ECCV, 2016.
- A. Kapitanov et al., “HaGRID — HAnd Gesture Recognition Image Dataset,” WACV,
  2024. <https://arxiv.org/abs/2206.08219>
- L. Lai, N. Suda, V. Chandra, “CMSIS-NN: Efficient neural network kernels for
  Arm Cortex-M CPUs,” arXiv:1801.06601, 2018.
- C. Banbury et al., “MLPerf Tiny benchmark,” NeurIPS Datasets and Benchmarks,
  2021. <https://arxiv.org/abs/2106.07597>
- Espressif Systems, “How to quantize model,” *ESP-DL User Guide*.
  <https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_quantize_model.html>
- M. Nagel et al., “A white paper on neural network quantization,”
  arXiv:2106.08295, 2021.
- G. Hinton, O. Vinyals, J. Dean, “Distilling the knowledge in a neural
  network,” arXiv:1503.02531, 2015.
- Y. Chen, D. Zhao, Y. S. Koh, T. Xu, “Animal re-identification on
  microcontrollers,” arXiv:2512.08198, 2025.
- A. Karjol, D. M. Hanna, “Edge AI for automotive vulnerable road user safety:
  Deployable detection via knowledge distillation,” arXiv:2604.26857, 2026.
- F. Zhang et al., “MediaPipe Hands: On-device real-time hand tracking,”
  arXiv:2006.10214, 2020.
- K. He, R. Girshick, P. Dollár, “Rethinking ImageNet pre-training,” ICCV, 2019.
- C. Banbury et al., “MicroNets: Neural network architectures for deploying
  TinyML applications on commodity microcontrollers,” arXiv:2010.11267, 2020.

## Code and data used

| Resource | Link | Used for |
|---|---|---|
| HaGRID dataset card (30k, 384p subset) | <https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p> | Training images; not redistributed |
| HaGRID project | <https://github.com/hukenovs/hagrid> | Original dataset and terms |
| MediaPipe Hand Landmarker | <https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker> | Pseudo-label boxes |
| MediaPipe task file | <https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task> | Downloaded by `dataset.py` |
| PyTorch | <https://pytorch.org/> | Training/export |
| torchvision | <https://pytorch.org/vision/stable/index.html> | MobileNetV2 backbone |
| ONNX | <https://onnx.ai/> | Static graph export |
| ONNX Runtime | <https://onnxruntime.ai/> | Export parity check |
| ESP-DL | <https://github.com/espressif/esp-dl> | On-device runtime |
| ESP-PPQ | <https://github.com/espressif/esp-ppq> | INT8 quantization / QAT |
| ESP-IDF | <https://github.com/espressif/esp-idf> | Firmware build system |
| CC BY-NC-SA 4.0 | <https://creativecommons.org/licenses/by-nc-sa/4.0/> | Documentation/data license |
| MIT | <https://opensource.org/license/mit> | Code license |

## Citation

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

See also [`../CITATION.cff`](../CITATION.cff).
