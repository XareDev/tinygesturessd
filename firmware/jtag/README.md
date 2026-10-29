# TinyGestureSSD JTAG firmware

A complete ESP-IDF reference application for the released width-0.30
knowledge-distilled student. It receives a 128×128 RGB frame over
USB-Serial-JTAG, runs the INT8 ESP-DL model, and returns the gesture class plus
the on-device inference and receive times.

This is the PTQ variant: it embeds
[`gesture_ssdlite_no_test.espdl`](models/gesture_ssdlite_no_test.espdl), the
183.33 KB PTQ export without ESP-PPQ test tensors. The QAT export is slightly
more accurate (69.20 vs 67.88 macro-F1) and can be swapped in; see
[Model selection](#model-selection).

## Hardware and configuration

| Item | Value |
|---|---|
| Board | ESP32-S3 with 16 MB flash and 8 MB octal PSRAM |
| CPU | 240 MHz |
| Flash | QIO, 80 MHz |
| PSRAM | Octal, 80 MHz |
| Partition | 6 MB factory app, no OTA |
| Image transport | USB-Serial-JTAG (native USB port) |
| Console | UART0, 115200 8N1 |
| Model | `models/gesture_ssdlite_no_test.espdl` |

The source and defaults assume the same N16R8 board profile used in the paper.
Change `sdkconfig.defaults` or run `idf.py menuconfig` if your board differs.
The app copies the embedded model into 32-byte-aligned PSRAM before inference;
a board without working PSRAM will fail at boot.

The console is deliberately on UART0 so `ESP_LOG*` output does not share the
USB channel with the image payload. `READY` and the result line are written
explicitly to USB-Serial-JTAG, so a host on the native USB port does not need
the UART console. If you only have the native USB cable connected, the app still
runs; you just will not see boot logs.

## Build and flash

Use ESP-IDF ≥ 5.4; the component manager fetches `espressif/esp-dl ^3.3.2`.

```bash
cd firmware/jtag
idf.py set-target esp32s3
idf.py build
idf.py -p COM7 flash
```

`build/`, `sdkconfig`, `managed_components/` and `dependencies.lock` are
generated locally and are ignored by Git. Flash over the native USB port. To
watch the UART0 console, connect a separate USB-UART bridge and run
`idf.py -p <uart-port> monitor`, or change the console back to
USB-Serial-JTAG in menuconfig if you accept logs sharing the data channel.

## Host protocol

```text
device -> host: READY\n
host   -> device: 0xAA | 49,152 signed INT8 bytes (128×128×3, HWC)
device -> host: CLASS:<name> SCORE:<0.000-1.000> INFER_US:<int> XFER_IN_US:<int>\n
```

- `0xAA` is the frame magic. There is no length field or checksum: the firmware
  reads exactly 49,152 bytes after the magic and times out after 5 s.
- The image bytes are the training contract, not raw 0–255 pixels:
  `round((pixel / 255) * 2^7)`, clipped to `[-128, 127]`, in HWC order.
- `SCORE` is the softmax probability of the best foreground class at the best
  anchor. It is a relative confidence, not a calibrated probability.
- If no anchor clears 0.5, the reply is `CLASS:none`; a missed detection is part
  of the measured accuracy.
- `INFER_US` is the span around `model->run()` only. `XFER_IN_US` is the board's
  measured USB receive time. Neither includes host write time or the time to
  read the reply.
- Bounding boxes are computed by the model but not decoded: the deployment
  target assumes one hand per frame and uses one global class argmax over the
  128 anchors. There is no NMS.

Class order is `fist`, `palm`, `like`, `ok`, `peace`, `point`; channel 0 of the
model output is background.

## Run the host benchmark

From the repository root:

```bash
pip install pyserial numpy pillow

python firmware/jtag/host/benchmark_latency.py --list-ports
python firmware/jtag/host/benchmark_latency.py \
  --port COM7 --image path/to/hand.jpg --n 30 --csv jtag_timings.csv
```

The script resets the board with DTR/RTS, waits for `READY`, sends the frame,
and separates device inference from round-trip overhead. If your board does not
reset from DTR/RTS, press reset manually and pass `--no-reset`. The model input
exponent is printed on the UART0 console at boot; the host defaults to `-7`
(scale `2^7`), which is the value used by every released TinyGestureSSD model.
Use `--input-exponent` if the log reports something else.

The paper measured 94.1 ms of inference and about 191 ms round trip for this
width-0.30 model on the N16R8 board. Rebuild and measure your own board before
quoting timing numbers.

## Model selection

| File | Quantization | Size | Macro-F1 | Notes |
|---|---|---:|---:|---|
| `models/gesture_ssdlite_no_test.espdl` | PTQ | 187,728 B | 67.88 | embedded here; no test tensors |
| `models/kd/gesture_ssdlite_qat.espdl` (repo model) | QAT | 187,712 B | 69.20 | better accuracy; recommended for deployment |

The firmware file is byte-identical to
`models/kd/gesture_ssdlite_no_test.espdl` (SHA-256
`978DFB132DDE81495700E2645641BC2462E4F0A1A49438C9762C8C9F4269E472`).

To switch to the QAT export:

1. Copy `models/kd/gesture_ssdlite_qat.espdl` into `firmware/jtag/models/`.
2. Change `MODEL_FILE` in `main/CMakeLists.txt`.
3. Change the two `extern` symbol names in `main/main.cpp` from
   `_binary_gesture_ssdlite_no_test_espdl_*` to
   `_binary_gesture_ssdlite_qat_espdl_*`.
4. Rebuild. The input shape, class order and output shapes are unchanged.

Do not rename a model to the current filename without updating the CMake path
and linker symbols; the symbol is derived from the embedded filename.

## Provenance and scope

The application was integrated from the supplied `JTAG_gesture_ptq` archive.
The archive's generated `sdkconfig`, empty `models/` directory and generic
README were not copied: firmware defaults are kept here, the PTQ model is
linked from the released model zoo, and this README documents the actual USB
protocol. The project was build-checked during integration with ESP-IDF v6.0
and ESP-DL 3.3.2; the repository's Python tests do not compile ESP-IDF
projects, so flash and validate timing on your own board.
