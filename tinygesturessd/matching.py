"""
Target Assignment (Anchor <-> Ground-Truth Matching).

Box regression targets are encoded relative to their matched anchor (standard SSD
parameterization, scaled by VARIANCES). Matching is fully vectorized across the batch --
every image has exactly one ground-truth box, so `box_iou_batch` broadcasts (B,4) against
(A,4) in a single shot with no per-image Python loop. Anchors with IoU >= 0.5 are positive,
IoU < 0.4 are explicitly background, and the gap between is ignored (excluded from the loss
entirely). The single best-IoU anchor per image is always forced positive so every image
contributes at least one positive, even if no anchor clears 0.5.
"""
import torch

from .anchors import box_iou_batch, corner_to_center
from . import config as cfg


def encode_boxes(gt_center, anchor_center, variances=cfg.VARIANCES):
    """Ground-truth center-form boxes -> SSD-style regression targets relative to their anchor."""
    g_cx, g_cy, g_w, g_h = gt_center.unbind(-1)
    a_cx, a_cy, a_w, a_h = anchor_center.unbind(-1)
    v1, v2, v3, v4 = variances
    t_cx = (g_cx - a_cx) / a_w / v1
    t_cy = (g_cy - a_cy) / a_h / v2
    t_w = torch.log(g_w / a_w) / v3
    t_h = torch.log(g_h / a_h) / v4
    return torch.stack([t_cx, t_cy, t_w, t_h], dim=-1)


def decode_boxes(pred_offsets, anchor_center, variances=cfg.VARIANCES):
    """Inverse of encode_boxes -- predicted offsets + anchor -> absolute center-form box."""
    t_cx, t_cy, t_w, t_h = pred_offsets.unbind(-1)
    a_cx, a_cy, a_w, a_h = anchor_center.unbind(-1)
    v1, v2, v3, v4 = variances
    p_cx = t_cx * v1 * a_w + a_cx
    p_cy = t_cy * v2 * a_h + a_cy
    p_w = torch.exp(t_w * v3) * a_w
    p_h = torch.exp(t_h * v4) * a_h
    return torch.stack([p_cx, p_cy, p_w, p_h], dim=-1)


def match_anchors_to_targets(gt_boxes_corner, gt_labels, anchors_center, anchors_corner,
                              pos_iou_thresh=cfg.POS_IOU_THRESH, neg_iou_thresh=cfg.NEG_IOU_THRESH):
    """
    Vectorized single-object-per-image anchor matching.
      gt_boxes_corner: (B,4) normalized (x1,y1,x2,y2)
      gt_labels:       (B,)  0-indexed foreground class id
    Returns:
      cls_targets: (B,A) long  -- 0=background, 1..num_classes=foreground, -1=ignore
      loc_targets: (B,A,4)     -- encoded regression targets (only valid where pos_mask is True)
      pos_mask:    (B,A) bool
      ignore_mask: (B,A) bool  -- 0.4 <= iou < 0.5, excluded from the loss entirely
    """
    B, A = gt_boxes_corner.shape[0], anchors_center.shape[0]

    iou = box_iou_batch(gt_boxes_corner, anchors_corner)            # (B,A)

    pos_mask = iou >= pos_iou_thresh
    forced_idx = iou.argmax(dim=1, keepdim=True)                    # guarantee >=1 positive/image
    pos_mask.scatter_(1, forced_idx, True)

    neg_mask = (iou < neg_iou_thresh) & (~pos_mask)
    ignore_mask = (~pos_mask) & (~neg_mask)

    cls_targets = torch.zeros(B, A, dtype=torch.long, device=gt_boxes_corner.device)
    cls_targets[pos_mask] = (gt_labels.unsqueeze(1).expand(B, A) + 1)[pos_mask]
    cls_targets[ignore_mask] = -1

    gt_center = corner_to_center(gt_boxes_corner)                   # (B,4)
    gt_center_exp = gt_center.unsqueeze(1).expand(B, A, 4)
    anchor_center_exp = anchors_center.unsqueeze(0).expand(B, A, 4)
    loc_targets = encode_boxes(gt_center_exp, anchor_center_exp)    # (B,A,4)

    return cls_targets, loc_targets, pos_mask, ignore_mask
