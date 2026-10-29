# Deployment: ONNX → ESP-PPQ → ESP-DL

The released `.espdl` files are already quantized for `esp32s3` at 8 bits. This
note documents which file to use, the exact preprocessing contract, and how the
graph is produced so you can rebuild it.

## 1. Which artifact should I deploy?

| Use case | File | Size | F1 (INT8) | Notes |
|---|---|---:|---:|---|
| Best accuracy at width 0.30 | `models/kd/gesture_ssdlite_qat.espdl` | 183.31 KB | 69.20 | QAT; recommended default |
| PTQ only, no test tensors | `models/kd/gesture_ssdlite_no_test.espdl` | 183.33 KB | 67.88 | same graph as the paper's PTQ row, smaller container |
| Paper-table PTQ file | `models/kd/gesture_ssdlite.espdl` | 232.98 KB | 67.88 | embeds ESP-PPQ test tensors (~49.7 KB); use only if you need the test outputs |
| Width-1.0 pretrained | `models/pretrained/gesture_ssdlite.espdl` | 643.03 KB | 80.22 | re-export without test tensors |
| Width-1.0 pretrained QAT | `models/pretrained/gesture_ssdlite_qat.espdl` | 643.03 KB | 83.94 | best large-model accuracy |
| Width-1.0 old PTQ | `models/pretrained/gesture_ssdlite_with_test_tensors.espdl` | 692.70 KB | 80.80 | earlier export with test tensors; kept for provenance |

The KD student is the paper's deployment target: 86,870 trainable parameters,
94 ms inference on the ESP32-S3, and the smallest useful model. The width-1.0
teacher is 5× slower and is included as the accuracy reference, not as the
recommended deployment.

## 2. Input preprocessing contract

The ESP-PPQ input exponent is `-7` for every variant in this study, i.e. a
quantization scale of `2^7 = 128` on a `[0,1]` RGB image. The host must produce
the exact INT8 input the model was calibrated with:

```python
# img_rgb: uint8 HxWx3, already resized to 128x128 with bilinear interpolation
x = img_rgb.astype(np.float32) / 255.0          # [0, 1]
x = np.rint(x * 128.0)                          # scale 2^7
x = np.clip(x, -128, 127).astype(np.int8)       # symmetric int8
# layout expected by the .espdl model: HWC, 128x128x3, int8
```

No ImageNet mean/std normalization is applied. The model input shape is
`[1, 3, 128, 128]` in ONNX (NCHW); the ESP-DL runtime stores the input as
`128 × 128 × 3` (HWC) but the generated `.espdl` handles the layout. If you
preprocess on the host, send the bytes in the order the firmware expects
(typically HWC) and let ESP-DL's `dl::ImagePreprocessor` or the model's input
tensor handle the rest.

If you use a camera on the device, resize/crop to 128 × 128, convert RGB888 to
the `[0,1] × 128` scale above, and write the result into the input tensor. Do not
feed raw 0–255 pixels: the calibrator never saw that range, and the quantized
graph will saturate.

## 3. Class order

The output has 7 channels: background first, then the six gestures in this
order:

```text
0 background
1 fist
2 palm
3 like
4 ok
5 peace
6 point
```

The firmware must use the same order. A mismatch is silent and produces
plausible but wrong labels.

## 4. Rebuilding the INT8 model

Export the fp32 checkpoint first (see the main README), then run ESP-PPQ. The
notebook uses this call (the API is from ESP-PPQ; check the version you install):

```python
from esp_ppq.api import espdl_quantize_onnx
from torch.utils.data import DataLoader

calib = DataLoader(val_dataset, batch_size=1, shuffle=False)

graph = espdl_quantize_onnx(
    onnx_import_file="models/kd/gesture_ssdlite.onnx",
    espdl_export_file="models/kd/gesture_ssdlite.espdl",
    calib_dataloader=calib,
    calib_steps=200,                 # 50-433 all scored within noise in the paper
    input_shape=[1, 3, 128, 128],
    inputs=None,
    target="esp32s3",
    num_of_bits=8,
    collate_fn=lambda batch: batch[0].to(device),
    device=device,
    error_report=False,
    skip_export=False,
    export_test_values=False,        # True embeds test tensors; adds ~49.7 KB
    verbose=0,
)
```

The calibration loader must yield the same `[0,1]` normalized tensors used in
training. The paper's calibration sweep uses the first `N` validation images in
fixed order; 50–100 images are enough for this model (F1 varies by <0.7 points
from 50 to 433 samples, within evaluation noise).

QAT is the ESP-PPQ `TrainableGraph` workflow: start from the PTQ graph, fine-tune
for 10 epochs at lr `1e-6`, momentum 0.937, and retain the best validation-F1
epoch. The best epochs in the released artifacts are 6 (KD), 9 (scratch 0.30),
and 10 (pretrained 1.0). The QAT exports omit test tensors, hence the size drop
from 232.98 to 183.31 KB at width 0.30.

## 5. ESP-IDF integration

A complete reference project is in [`../firmware/jtag/`](../firmware/jtag/README.md).
It uses the PTQ export without test tensors and carries the actual USB protocol,
partition table, PSRAM defaults and host benchmark. Add ESP-DL as a managed
component and embed the `.espdl` as a binary blob:

```yaml
# main/idf_component.yml
dependencies:
  idf:
    version: ">=5.4"
  espressif/esp-dl:
    version: "^3.3.2"
```

```cmake
# main/CMakeLists.txt
idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp-dl esp_driver_usb_serial_jtag esp_timer esp_hw_support heap
)

set(MODEL_FILE "${CMAKE_SOURCE_DIR}/models/gesture_ssdlite_no_test.espdl")
target_add_binary_data(${COMPONENT_LIB} "${MODEL_FILE}" BINARY)
```

```cpp
extern const uint8_t model_data_start[]
    asm("_binary_gesture_ssdlite_no_test_espdl_start");

dl::Model model(reinterpret_cast<const char *>(model_data_start),
                fbs::MODEL_LOCATION_IN_FLASH_RODATA);
model.minimize();   // optional; releases intermediate runtime buffers

// Fill the model's input tensor with the int8 HWC image, then:
model.run(model.get_inputs());
// cls_logits: [1, 128, 7] int8; loc_preds: [1, 128, 4] int8
```

The exact tensor accessor names depend on the ESP-DL version; the released
models were produced with ESP-DL 3.3.2. The model is static-shape and batch-1,
so do not reuse it from multiple tasks concurrently without a mutex. The
reference firmware makes a 32-byte-aligned PSRAM copy of the blob before
constructing the model and uses `MEMORY_MANAGER_GREEDY`; on a board without
PSRAM, use the flash-rodata path instead.

The reference firmware sends `READY\n` on USB-Serial-JTAG, then accepts
`0xAA | 49,152 signed INT8 bytes` and replies:

```text
CLASS:<name> SCORE:<0.000-1.000> INFER_US:<int> XFER_IN_US:<int>
```

`SCORE` is the softmax probability of the best foreground class at the best of
the 128 anchors; `CLASS:none` means no anchor cleared 0.5. Boxes are not decoded
because the target assumes one hand per frame. See the firmware README for the
host benchmark and wiring.

## 6. Latency and overhead

From the paper (ESP32-S3, 8 MB octal PSRAM, 16 MB flash, native
USB-Serial-JTAG, 30 repeats, single fixed image):

| Model | Quant. | Inference (ms) | Round trip (ms) | Transfer/protocol share |
|---|---|---:|---:|---:|
| Scratch 0.30 | PTQ | 94.05 ± 0.24 | 190.51 ± 0.67 | 50.63% |
| Scratch 0.30 | QAT | 94.05 ± 0.24 | 191.32 ± 1.05 | 50.84% |
| KD student 0.30 | PTQ | 94.11 ± 0.22 | 191.32 ± 0.92 | 50.81% |
| KD student 0.30 | QAT | 94.10 ± 0.22 | 190.95 ± 0.63 | 50.72% |
| Pretrained 1.0 | PTQ | 468.35 ± 0.38 | 564.88 ± 0.73 | 17.09% |
| Pretrained 1.0 | QAT | 468.90 ± 0.37 | 565.95 ± 1.10 | 17.15% |

Inference time is measured with `esp_timer` around `model.run()` only. The
round trip includes image upload and response parsing, which is ~96–97 ms
regardless of model size. At width 0.30 the fixed transfer cost is about half
the round trip; optimizing the transport is as important as shrinking the model.

These measurements characterize timing and repeatability, not on-device
accuracy over the validation set. Accuracy in the paper is from the host-side
ESP-PPQ reference executor.

## 7. Common failure modes

- **All predictions collapse to one class.** Check the RGB/BGR order and the
  `[0,1] × 128` scaling; a 0–255 input saturates the first layer.
- **Class labels shifted by one.** The model's channel 0 is background; the
  firmware must not treat it as `fist`.
- **Different accuracy from the paper.** The validation split is seed-0 and
  fixed; re-scoring a checkpoint in a fresh process can move one image because
  the 0.5 threshold turns a near-threshold detection into a miss. The paper's
  bootstrap intervals cover that noise.
- **`.espdl` too large.** Re-export with `export_test_values=False`; the PTQ
  test tensors add ~49.7 KB and are only useful for ESP-PPQ debugging.
