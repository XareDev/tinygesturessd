#!/usr/bin/env python3
"""Host driver and latency benchmark for the TinyGestureSSD JTAG firmware.

The firmware reports two separate timings:

  device_infer_ms  model->run() only, measured with esp_timer on the board
  device_xfer_in_ms  the board's measured USB receive time

The host measures the full round trip, so transfer/protocol overhead is the
round trip minus device inference.  The input contract is the training contract:
128x128 RGB, [0,1], scaled by 2^7 and clipped to signed INT8.

Usage:
    python benchmark_latency.py --image test.jpg --n 30
    python benchmark_latency.py --port COM5 --dir test_images/ --n 5 --csv results.csv
    python benchmark_latency.py --list-ports

Requires: pyserial, numpy, pillow
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import serial
import serial.tools.list_ports
from PIL import Image

IMG_SIZE = 128
IMG_BYTES = IMG_SIZE * IMG_SIZE * 3
MAGIC = 0xAA

# All released TinyGestureSSD .espdl files use exponent -7.  main.cpp logs the
# model's actual exponent on UART0 at boot; pass --input-exponent if it differs.
INPUT_EXPONENT = -7


def quantize_image(path: str, exponent: int = INPUT_EXPONENT) -> bytes:
    """Resize, convert to RGB, divide by 255, then quantize to signed INT8."""
    image = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    scale = 2.0 ** (-exponent)
    arr_i8 = np.clip(np.rint(arr * scale), -128, 127).astype(np.int8)
    payload = arr_i8.tobytes()
    assert len(payload) == IMG_BYTES, f"expected {IMG_BYTES} bytes, got {len(payload)}"
    return payload


def list_ports() -> list:
    return sorted(serial.tools.list_ports.comports(), key=lambda p: p.device)


def choose_port() -> str:
    ports = list_ports()
    if not ports:
        sys.exit("no serial ports found -- plug in the board and retry")

    if len(ports) == 1:
        port = ports[0]
        print(f"one port found, using it: {port.device} ({port.description})")
        return port.device

    print("multiple serial ports found:")
    for i, port in enumerate(ports):
        hint = f"  [{port.hwid}]" if port.hwid else ""
        print(f"  {i}: {port.device:<15} {port.description}{hint}")

    while True:
        choice = input(f"select port [0-{len(ports) - 1}]: ").strip()
        if choice.isdigit() and 0 <= int(choice) < len(ports):
            return ports[int(choice)].device
        print("invalid selection, try again")


def reset_board(ser: serial.Serial) -> None:
    """Pulse DTR/RTS the way the archived host scripts did."""
    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(0.1)
    ser.setDTR(True)
    time.sleep(0.5)
    ser.setDTR(False)


def wait_for_ready(ser: serial.Serial, timeout_s: float = 15.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            print(f"  [board] {line}")
        if "READY" in line:
            return
    raise TimeoutError(
        "board never sent READY on the USB channel. Check that:\n"
        "  - the native USB port is selected (not the UART bridge),\n"
        "  - the firmware was flashed and the board was reset,\n"
        "  - sdkconfig.defaults kept the USB-Serial-JTAG data path."
    )


def send_one(ser: serial.Serial, payload: bytes) -> dict:
    t_send_start = time.perf_counter()
    ser.write(bytes([MAGIC]))
    chunk = 4096
    for i in range(0, len(payload), chunk):
        ser.write(payload[i:i + chunk])
    ser.flush()
    t_send_done = time.perf_counter()

    line = ""
    deadline = time.time() + 10.0
    while time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line.startswith("CLASS:"):
            break
    else:
        raise TimeoutError(
            "no CLASS: response before deadline; check the UART0 console for "
            "the firmware's USB timeout message"
        )
    t_result = time.perf_counter()

    fields = dict(token.split(":", 1) for token in line.split())
    infer_us = int(fields["INFER_US"])
    xfer_in_us = int(fields["XFER_IN_US"])

    host_round_trip_ms = (t_result - t_send_start) * 1000
    host_upload_ms = (t_send_done - t_send_start) * 1000
    device_infer_ms = infer_us / 1000
    device_xfer_in_ms = xfer_in_us / 1000
    overhead_ms = host_round_trip_ms - device_infer_ms

    return {
        "class": fields.get("CLASS", "?"),
        "score": float(fields.get("SCORE", 0.0)),
        "host_round_trip_ms": host_round_trip_ms,
        "host_upload_ms": host_upload_ms,
        "device_infer_ms": device_infer_ms,
        "device_xfer_in_ms": device_xfer_in_ms,
        "overhead_ms": overhead_ms,
        "overhead_share_pct": 100 * overhead_ms / host_round_trip_ms,
    }


def summarize(runs: list, label: str) -> None:
    def stats(key: str) -> tuple:
        vals = np.asarray([r[key] for r in runs])
        return vals.mean(), np.median(vals), vals.std(), np.percentile(vals, 95)

    print(f"\n=== {label} (n={len(runs)}) ===")
    print(f"{'metric':<22}{'mean':>10}{'median':>10}{'std':>10}{'p95':>10}   unit")
    for key, unit in [
        ("device_infer_ms", "ms"),
        ("device_xfer_in_ms", "ms"),
        ("host_round_trip_ms", "ms"),
        ("overhead_ms", "ms"),
        ("overhead_share_pct", "%"),
    ]:
        mean, median, std, p95 = stats(key)
        print(f"{key:<22}{mean:>10.2f}{median:>10.2f}{std:>10.2f}{p95:>10.2f}   {unit}")

    mean_overhead_pct = float(np.mean([r["overhead_share_pct"] for r in runs]))
    print(f"\n{mean_overhead_pct:.1f}% of the round trip was not model compute.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--port", help="e.g. COM5 or /dev/ttyACM0")
    ap.add_argument("--list-ports", action="store_true", help="print ports and exit")
    ap.add_argument("--baud", type=int, default=115200, help="ignored by native USB-CDC")
    ap.add_argument("--image", help="single image path")
    ap.add_argument("--dir", help="directory of .jpg/.png images")
    ap.add_argument("--n", type=int, default=20, help="repeats per image (>=1)")
    ap.add_argument("--input-exponent", type=int, default=INPUT_EXPONENT,
                    help="model input exponent; default -7 (scale 2^7)")
    ap.add_argument("--no-reset", action="store_true",
                    help="do not pulse DTR/RTS before waiting for READY")
    ap.add_argument("--csv", help="write raw per-run timings to this CSV")
    args = ap.parse_args()

    if args.list_ports:
        ports = list_ports()
        if not ports:
            sys.exit("no serial ports found")
        for port in ports:
            hint = f"  [{port.hwid}]" if port.hwid else ""
            print(f"{port.device:<15} {port.description}{hint}")
        return

    if not args.image and not args.dir:
        ap.error("pass --image or --dir")
    if args.n < 1:
        ap.error("--n must be at least 1")

    port = args.port if args.port else choose_port()

    if args.image:
        image_paths = [args.image]
    else:
        image_paths = sorted(
            glob.glob(os.path.join(args.dir, "*.jpg"))
            + glob.glob(os.path.join(args.dir, "*.jpeg"))
            + glob.glob(os.path.join(args.dir, "*.png"))
        )
    if not image_paths:
        sys.exit("no images found")

    print(f"connecting to {port} ...")
    with serial.Serial(port, args.baud, timeout=1) as ser:
        if not args.no_reset:
            print("resetting board ...")
            reset_board(ser)
        wait_for_ready(ser)

        all_runs = []
        csv_rows = []
        for path in image_paths:
            print(f"\n--- {path} ---")
            payload = quantize_image(path, args.input_exponent)
            per_image_runs = []
            for i in range(args.n):
                result = send_one(ser, payload)
                per_image_runs.append(result)
                all_runs.append(result)
                csv_rows.append({"image": path, "run": i, **result})
                print(
                    f"  run {i + 1}/{args.n}: class={result['class']:<8} "
                    f"score={result['score']:.3f}  "
                    f"infer={result['device_infer_ms']:.2f} ms  "
                    f"round_trip={result['host_round_trip_ms']:.2f} ms  "
                    f"overhead={result['overhead_ms']:.2f} ms "
                    f"({result['overhead_share_pct']:.0f}%)"
                )
            summarize(per_image_runs, os.path.basename(path))

        if len(image_paths) > 1:
            summarize(all_runs, "ALL IMAGES COMBINED")

        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"\nsaved raw per-run timings: {args.csv}")


if __name__ == "__main__":
    main()
