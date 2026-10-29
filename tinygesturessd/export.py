"""
ONNX Export (fixed shapes) -> ESP-PPQ / ESP-DL.

Static shapes throughout -- ESP-PPQ (and ESP-DL) expect a fixed-shape graph, so
`dynamic_axes=None` and the legacy TorchScript-based exporter (`dynamo=False`) are used
deliberately here.
"""
import os

import numpy as np
import torch
import torch.nn as nn

from . import config as cfg


class ONNXWrapper(nn.Module):
    """Ensures ONNX export sees exactly two plain tensor outputs."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        cls_out, loc_out = self.m(x)
        return cls_out, loc_out


def export_to_onnx(model, onnx_path: str = None):
    onnx_path = onnx_path or os.path.join(cfg.MODELS_DIR, "gesture_ssdlite.onnx")
    export_model = ONNXWrapper(model).to("cpu").eval()
    dummy_input = torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE)

    torch.onnx.export(
        export_model,
        dummy_input,
        onnx_path,
        opset_version=13,
        input_names=["input"],
        output_names=["cls_logits", "loc_preds"],
        do_constant_folding=True,
        dynamic_axes=None,      # fixed shapes required for ESP-PPQ / ESP-DL
        dynamo=False,            # legacy TorchScript exporter -> static-shape graph
    )
    print("exported:", onnx_path)
    return onnx_path, export_model, dummy_input


def verify_onnx(onnx_path: str, export_model: nn.Module, dummy_input: torch.Tensor):
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("onnx.checker passed")

    for t in onnx_model.graph.input:
        print("input :", t.name, [d.dim_value for d in t.type.tensor_type.shape.dim])
    for t in onnx_model.graph.output:
        print("output:", t.name, [d.dim_value for d in t.type.tensor_type.shape.dim])

    # numerically verify the ONNX graph matches the PyTorch model
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_cls, ort_loc = sess.run(None, {"input": dummy_input.numpy()})
    with torch.no_grad():
        torch_cls, torch_loc = export_model(dummy_input)

    max_diff_cls = np.abs(ort_cls - torch_cls.numpy()).max()
    max_diff_loc = np.abs(ort_loc - torch_loc.numpy()).max()
    print(f"max abs diff -> cls: {max_diff_cls:.2e}   loc: {max_diff_loc:.2e}")
    assert np.allclose(ort_cls, torch_cls.numpy(), atol=1e-4)
    assert np.allclose(ort_loc, torch_loc.numpy(), atol=1e-4)
    print("ONNX <-> PyTorch numerical match OK")

    print(f"\n{onnx_path}: {os.path.getsize(onnx_path) / 1024:.1f} KB")
