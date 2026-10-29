"""
Inference & Simplified Post-Processing.

No greedy NMS loop: the deployment target assumes at most one hand per frame, so
post-processing is a confidence threshold, then a single global argmax over all anchors
to pick the one box.
"""
import torch
import torch.nn.functional as F

from . import config as cfg
from .anchors import center_to_corner
from .matching import decode_boxes


@torch.no_grad()
def detect_single_hand(model, image_tensor, anchors_center, conf_thresh: float = 0.5):
    """
    image_tensor: (1,3,H,W). Returns the single highest-confidence detection above
    conf_thresh, or None. Single-hand assumption -> threshold, then one global argmax
    over all anchors.
    """
    model.eval()
    cls_preds, loc_preds = model(image_tensor)               # (1,A,C+1), (1,A,4)

    probs = F.softmax(cls_preds, dim=-1)
    fg_probs = probs[..., 1:]                                  # drop the background column
    best_score_per_anchor, best_cls_per_anchor = fg_probs.max(dim=-1)   # (1,A) each

    # 1) confidence threshold ...
    thresholded = best_score_per_anchor.masked_fill(best_score_per_anchor < conf_thresh, -1.0)
    # 2) ... then a single global argmax over all anchors
    best_score, best_anchor_idx = thresholded.max(dim=-1)

    if best_score.item() < 0:                                  # nothing cleared the threshold
        return None

    b = 0
    a_idx = best_anchor_idx[b].item()
    cls_id = best_cls_per_anchor[b, a_idx].item()
    score = best_score_per_anchor[b, a_idx].item()

    decoded_center = decode_boxes(loc_preds[b, a_idx], anchors_center[a_idx])
    decoded_corner = center_to_corner(decoded_center.unsqueeze(0)).squeeze(0).clamp(0.0, 1.0)

    return {
        "label_id": cls_id,
        "label_name": cfg.CLASS_NAMES[cls_id],
        "score": score,
        "box_norm": decoded_corner,          # (4,) x1,y1,x2,y2 in [0,1]
        "box_px": decoded_corner * cfg.IMG_SIZE,
        "anchor_idx": a_idx,
    }
