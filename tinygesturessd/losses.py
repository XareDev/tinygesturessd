"""
MultiBox Loss (Smooth L1 + Cross-Entropy, 3:1 Hard Negative Mining).

Localization loss only ever touches matched-positive anchors -- `loc_preds[pos_mask]` masks
the rest out entirely. For classification, every anchor gets a per-element cross-entropy loss
first; positive and ignored anchors are excluded from the negative-mining pool via `-inf`, then
a double-argsort (sort, then sort the sort) gives each remaining background anchor its rank by
loss magnitude with no Python loop over the batch or anchors. Keeping the top
`3 * num_positives` hardest negatives per image enforces the 3:1 ratio exactly, even when the
ratio would require a different cutoff per image in the batch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as cfg


class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio: int = cfg.NEG_POS_RATIO):
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, cls_preds, loc_preds, cls_targets, loc_targets, pos_mask, ignore_mask):
        B, A, C = cls_preds.shape

        # ---- localization: positives only ----
        loc_loss = F.smooth_l1_loss(loc_preds[pos_mask], loc_targets[pos_mask], reduction="sum")

        # ---- classification with 3:1 hard negative mining ----
        safe_targets = cls_targets.clamp(min=0)   # -1 (ignore) -> 0 just so CE doesn't error; excluded below
        loss_per_anchor = F.cross_entropy(
            cls_preds.view(-1, C), safe_targets.view(-1), reduction="none"
        ).view(B, A)

        loss_for_ranking = loss_per_anchor.clone()
        loss_for_ranking[pos_mask] = float("-inf")     # never select positives as "negatives"
        loss_for_ranking[ignore_mask] = float("-inf")  # never select ignored anchors either

        num_pos = pos_mask.sum(dim=1, keepdim=True)                          # (B,1)
        num_neg_avail = (~pos_mask & ~ignore_mask).sum(dim=1, keepdim=True)
        num_neg = torch.clamp(self.neg_pos_ratio * num_pos, max=num_neg_avail)

        _, loss_idx = loss_for_ranking.sort(dim=1, descending=True)
        _, rank = loss_idx.sort(dim=1)               # rank[b,a] = position of anchor a in the sorted order
        hard_neg_mask = rank < num_neg

        selected = pos_mask | hard_neg_mask
        cls_loss = loss_per_anchor[selected].sum()

        num_pos_total = num_pos.sum().clamp(min=1).float()
        loc_loss = loc_loss / num_pos_total
        cls_loss = cls_loss / num_pos_total
        return loc_loss + cls_loss, loc_loss, cls_loss
