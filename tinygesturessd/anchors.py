"""
Anchor generation & box utilities.

Fully vectorized anchor generation via `torch.meshgrid(..., indexing='ij')` -- no Python
loop over cells or aspect ratios. Anchor order is row-major over cells, then aspect ratio
within each cell; the detection heads (model.py) must emit channels in this same order.
"""
import torch


def generate_anchors(img_size, stride, base_size_px, aspect_ratios):
    """
    Vectorized anchor generation.
    Returns center-form anchors (cx, cy, w, h), normalized to [0, 1], shape (G*G*R, 4).
    Order: row-major over grid cells (row-major = row changes slowest), then aspect
    ratio within each cell (fastest-changing) -- must match the head's channel layout.
    """
    grid_size = img_size // stride
    base = base_size_px / img_size

    rows = torch.arange(grid_size, dtype=torch.float32)
    cols = torch.arange(grid_size, dtype=torch.float32)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")     # (G,G), row-major

    cy = ((grid_r + 0.5) * stride / img_size).reshape(-1)          # (G*G,)
    cx = ((grid_c + 0.5) * stride / img_size).reshape(-1)          # (G*G,)

    ratios = torch.tensor(aspect_ratios, dtype=torch.float32)      # (R,)
    R = ratios.shape[0]

    # repeat each cell's center R times contiguously -> ratio is the fast axis
    cx = cx.unsqueeze(1).expand(-1, R).reshape(-1)                 # (G*G*R,)
    cy = cy.unsqueeze(1).expand(-1, R).reshape(-1)

    w = (base * ratios.sqrt()).unsqueeze(0).expand(grid_size * grid_size, -1).reshape(-1)
    h = (base / ratios.sqrt()).unsqueeze(0).expand(grid_size * grid_size, -1).reshape(-1)

    return torch.stack([cx, cy, w, h], dim=-1)


def _generate_anchors_loop_reference(img_size, stride, base_size_px, aspect_ratios):
    """Slow, obviously-correct loop version -- used only to unit-test the vectorized one above."""
    grid_size = img_size // stride
    base = base_size_px / img_size
    out = []
    for row in range(grid_size):
        for col in range(grid_size):
            cx = (col + 0.5) * stride / img_size
            cy = (row + 0.5) * stride / img_size
            for ratio in aspect_ratios:
                w = base * (ratio ** 0.5)
                h = base / (ratio ** 0.5)
                out.append([cx, cy, w, h])
    return torch.tensor(out, dtype=torch.float32)


def verify_anchors(anchors_center, img_size, stride, base_size_px, aspect_ratios):
    """Asserts the vectorized anchors match the loop reference exactly. Call once at startup."""
    ref = _generate_anchors_loop_reference(img_size, stride, base_size_px, aspect_ratios)
    assert torch.allclose(anchors_center, ref), "vectorized anchors don't match the loop reference!"
    print("anchor generation verified against loop reference")


def center_to_corner(boxes):
    """(cx, cy, w, h) -> (x1, y1, x2, y2)"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def corner_to_center(boxes):
    """(x1, y1, x2, y2) -> (cx, cy, w, h)"""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def box_iou(boxes1, boxes2):
    """Vectorized pairwise IoU. boxes1: (N,4), boxes2: (M,4), corner-form -> (N,M)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-8)


def box_iou_batch(gt_boxes_corner, anchors_corner):
    """
    Batched one-box-per-image IoU against the full anchor set -- no Python loop over the batch.
    gt_boxes_corner: (B,4), anchors_corner: (A,4) -> (B,A)
    """
    gt = gt_boxes_corner.unsqueeze(1)          # (B,1,4)
    an = anchors_corner.unsqueeze(0)           # (1,A,4)
    lt = torch.max(gt[..., :2], an[..., :2])
    rb = torch.min(gt[..., 2:], an[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_gt = (gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])
    area_an = (an[..., 2] - an[..., 0]) * (an[..., 3] - an[..., 1])
    union = area_gt + area_an - inter
    return inter / union.clamp(min=1e-8)
