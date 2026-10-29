"""
Backbone: truncated MobileNetV2, cut right after block 13 (`features[:14]`).
Heads: SSDLite-style depthwise-separable prediction heads.

Channel counts are *inferred* from a dummy forward pass, never hard-coded, so
`WIDTH_MULT` in config.py stays a single knob for the whole model's capacity.
"""
import torch
import torch.nn as nn
import torchvision

from . import config as cfg


def build_truncated_backbone(width_mult: float = cfg.WIDTH_MULT) -> nn.Module:
    full = torchvision.models.mobilenet_v2(weights=None, width_mult=width_mult)
    return nn.Sequential(*list(full.features.children())[:cfg.CUT_INDEX])


def probe_backbone_blocks(width_mult: float = 1.0, img_size: int = cfg.IMG_SIZE):
    """Prints each MobileNetV2 block's output shape -- useful for picking CUT_INDEX."""
    full = torchvision.models.mobilenet_v2(weights=None, width_mult=width_mult)
    x = torch.zeros(1, 3, img_size, img_size)
    print(f"--- block-by-block probe (width_mult={width_mult}) ---")
    with torch.no_grad():
        for i, block in enumerate(full.features):
            x = block(x)
            print(f"{i:2d}  {type(block).__name__:20s}  {tuple(x.shape)}")


class SeparableHead(nn.Module):
    """SSDLite-style prediction head: depthwise 3x3 (+BN+ReLU6) -> pointwise 1x1."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                                    groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU6(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.act(self.bn(self.depthwise(x)))
        return self.pointwise(x)


def reshape_head_output(x, num_anchors_per_cell: int, out_per_anchor: int):
    """(B, A_cell*K, H, W) -> (B, H*W*A_cell, K), row-major cells then ratio-within-cell."""
    B, _, H, W = x.shape
    x = x.view(B, num_anchors_per_cell, out_per_anchor, H, W)
    x = x.permute(0, 3, 4, 1, 2).contiguous()          # (B, H, W, A_cell, K)
    return x.view(B, H * W * num_anchors_per_cell, out_per_anchor)


class SSDLiteGestureDetector(nn.Module):
    def __init__(self, num_classes: int = cfg.NUM_CLASSES, width_mult: float = cfg.WIDTH_MULT,
                 num_anchors_per_cell: int = cfg.NUM_ANCHORS_PER_CELL, img_size: int = cfg.IMG_SIZE):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors_per_cell = num_anchors_per_cell
        self.width_mult = width_mult

        self.backbone = build_truncated_backbone(width_mult)

        # channel count is *inferred*, never hard-coded, so width_mult stays a single knob
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 3, img_size, img_size))
        in_channels, grid = feat.shape[1], feat.shape[2]
        self.in_channels, self.grid_size = in_channels, grid

        self.cls_head = SeparableHead(in_channels, num_anchors_per_cell * (num_classes + 1))
        self.loc_head = SeparableHead(in_channels, num_anchors_per_cell * 4)

    def forward(self, x):
        feat = self.backbone(x)
        cls_out = reshape_head_output(self.cls_head(feat), self.num_anchors_per_cell, self.num_classes + 1)
        loc_out = reshape_head_output(self.loc_head(feat), self.num_anchors_per_cell, 4)
        return cls_out, loc_out                         # (B, num_anchors, C+1), (B, num_anchors, 4)

    def param_summary(self, verbose: bool = True):
        def n(m):
            return sum(p.numel() for p in m.parameters())

        bb, cls_h, loc_h = n(self.backbone), n(self.cls_head), n(self.loc_head)
        total = bb + cls_h + loc_h
        if verbose:
            print(f"{'backbone (mbv2 features[:14])':35s} {bb:>8,}")
            print(f"{'classification head':35s} {cls_h:>8,}")
            print(f"{'localization head':35s} {loc_h:>8,}")
            print("-" * 44)
            print(f"{'TOTAL':35s} {total:>8,}   (width_mult={self.width_mult}, trainable parameters)")
        return total
