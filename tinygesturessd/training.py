"""
Training Loop, with Per-Epoch Accuracy / Precision / Recall / F1.

`SELECTION_METRIC` (config.py) picks which validation metric selects the checkpoint that
gets saved -- macro-F1 by default, since it balances precision and recall and weights all
6 gesture classes equally regardless of how many validation examples each one has.

Local addition vs. the original notebook: per-*iteration* (not just per-epoch) loss is
now recorded too, in history["iter_total"] / ["iter_loc"] / ["iter_cls"], so you can plot
the raw training curve instead of only 12 epoch-averaged points.
"""
import os

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm

from . import config as cfg
from .losses import MultiBoxLoss
from .matching import match_anchors_to_targets


@torch.no_grad()
def predict_batch_labels(model, imgs, anchors_center, conf_thresh: float = 0.5):
    """
    Returns pred_labels: (B,) long tensor, project class id, or -1 if nothing cleared
    conf_thresh. Used for fast per-epoch validation (batched version of inference.py's
    single-image detector).
    """
    cls_preds, loc_preds = model(imgs)                       # (B,A,C+1), (B,A,4)
    probs = F.softmax(cls_preds, dim=-1)
    fg_probs = probs[..., 1:]
    best_score_per_anchor, best_cls_per_anchor = fg_probs.max(dim=-1)      # (B,A)

    thresholded = best_score_per_anchor.masked_fill(best_score_per_anchor < conf_thresh, -1.0)
    best_score, best_anchor_idx = thresholded.max(dim=-1)                  # (B,)

    pred_labels = best_cls_per_anchor.gather(1, best_anchor_idx.unsqueeze(1)).squeeze(1)
    pred_labels = torch.where(best_score >= 0, pred_labels, torch.full_like(pred_labels, -1))
    return pred_labels


@torch.no_grad()
def evaluate(model, loader, anchors_center, device, conf_thresh: float = 0.5):
    """Runs the detector over a loader and computes classification-style metrics on the
    gesture label of the single detected hand (a missed detection counts as a wrong
    prediction)."""
    model.eval()
    all_preds, all_gts = [], []
    for imgs, boxes, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = predict_batch_labels(model, imgs, anchors_center, conf_thresh)
        all_preds.append(preds.cpu())
        all_gts.append(labels.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_gts = torch.cat(all_gts).numpy()

    accuracy = accuracy_score(all_gts, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_gts, all_preds, labels=list(range(cfg.NUM_CLASSES)), average="macro", zero_division=0
    )
    n_missed = int((all_preds == -1).sum())
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1_macro": f1, "n_missed": n_missed, "n_total": len(all_gts)}


def train_model(model, train_loader, val_loader, anchors_center, anchors_corner, device,
                 epochs: int = cfg.EPOCHS, lr: float = cfg.LR, weight_decay: float = cfg.WEIGHT_DECAY,
                 selection_metric: str = cfg.SELECTION_METRIC):
    criterion = MultiBoxLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "total": [], "loc": [], "cls": [],
        "val_accuracy": [], "val_precision": [], "val_recall": [], "val_f1_macro": [],
        "iter_total": [], "iter_loc": [], "iter_cls": [],   # per-iteration, never reset
    }

    best_metric_value = -1.0
    best_state_dict = None
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"total": 0.0, "loc": 0.0, "cls": 0.0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for imgs, boxes, labels in pbar:
            imgs, boxes, labels = imgs.to(device), boxes.to(device), labels.to(device)

            cls_preds, loc_preds = model(imgs)
            cls_t, loc_t, pos_m, ign_m = match_anchors_to_targets(boxes, labels, anchors_center, anchors_corner)
            total_loss, loc_loss, cls_loss = criterion(cls_preds, loc_preds, cls_t, loc_t, pos_m, ign_m)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            history["iter_total"].append(total_loss.item())
            history["iter_loc"].append(loc_loss.item())
            history["iter_cls"].append(cls_loss.item())

            running["total"] += total_loss.item()
            running["loc"] += loc_loss.item()
            running["cls"] += cls_loss.item()
            n_batches += 1

            pbar.set_postfix(total=running["total"] / n_batches,
                              loc=running["loc"] / n_batches,
                              cls=running["cls"] / n_batches)

        scheduler.step()
        for k in ("total", "loc", "cls"):
            history[k].append(running[k] / n_batches)

        val_metrics = evaluate(model, val_loader, anchors_center, device)
        for k in ("accuracy", "precision", "recall", "f1_macro"):
            history[f"val_{k}"].append(val_metrics[k])

        is_best = val_metrics[selection_metric] > best_metric_value
        if is_best:
            best_metric_value = val_metrics[selection_metric]
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        print(f"epoch {epoch:2d}/{epochs}  total={history['total'][-1]:.4f}  "
              f"loc={history['loc'][-1]:.4f}  cls={history['cls'][-1]:.4f}  |  "
              f"val_acc={val_metrics['accuracy']:.3f}  val_prec={val_metrics['precision']:.3f}  "
              f"val_rec={val_metrics['recall']:.3f}  val_f1={val_metrics['f1_macro']:.3f}  "
              f"(missed {val_metrics['n_missed']}/{val_metrics['n_total']})"
              f"{'  <-- best so far' if is_best else ''}")

    print(f"\nbest checkpoint: epoch {best_epoch}  {selection_metric}={best_metric_value:.4f}")
    best_model_path = os.path.join(cfg.MODELS_DIR, "best_model.pt")
    torch.save(best_state_dict, best_model_path)
    model.load_state_dict(best_state_dict)   # switch the live model to the best checkpoint
    print(f"loaded best checkpoint into `model` and saved it to {best_model_path}")

    return history, best_epoch, best_metric_value
