"""
Training and evaluation loops for MultimodalSentimentModel.
"""

import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, f1_score


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    start = time.time()

    for batch in loader:
        # ── Move to device ─────────────────────────────────────────────────
        text_ids         = batch["text_ids"].to(device)
        text_mask        = batch["text_mask"].to(device)
        image_pixels     = batch["image_pixels"].to(device)
        gen_text_ids     = batch["gen_text_ids"].to(device)
        gen_text_mask    = batch["gen_text_mask"].to(device)
        gen_image_pixels = batch["gen_image_pixels"].to(device)
        labels           = batch["label"].to(device)

        optimizer.zero_grad()

        # ── Forward ────────────────────────────────────────────────────────
        logits = model(
            text_ids, text_mask,
            image_pixels,
            gen_text_ids, gen_text_mask,
            gen_image_pixels,
        )

        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # ── Metrics ────────────────────────────────────────────────────────
        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return {
        "loss":     total_loss / total,
        "accuracy": correct / total,
        "time":     time.time() - start,
    }


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    label_names: Optional[list] = None,
) -> Dict:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        text_ids         = batch["text_ids"].to(device)
        text_mask        = batch["text_mask"].to(device)
        image_pixels     = batch["image_pixels"].to(device)
        gen_text_ids     = batch["gen_text_ids"].to(device)
        gen_text_mask    = batch["gen_text_mask"].to(device)
        gen_image_pixels = batch["gen_image_pixels"].to(device)
        labels           = batch["label"].to(device)

        logits = model(
            text_ids, text_mask,
            image_pixels,
            gen_text_ids, gen_text_mask,
            gen_image_pixels,
        )

        loss    = criterion(logits, labels)
        preds   = logits.argmax(dim=-1)

        total_loss += loss.item() * labels.size(0)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report   = classification_report(
        all_labels, all_preds,
        target_names=label_names or ["negative", "neutral", "positive"],
        zero_division=0,
    )

    return {
        "loss":     total_loss / total,
        "accuracy": correct / total,
        "macro_f1": macro_f1,
        "report":   report,
    }


def train(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    num_epochs:   int = 20,
    lr:           float = 2e-5,
    weight_decay: float = 1e-2,
    device:       Optional[torch.device] = None,
    save_path:    str = "best_model.pt",
) -> nn.Module:
    """
    Full training loop with cosine LR schedule, early-stop by macro-F1.
    Returns the model loaded with best weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer  = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler  = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr / 100)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_f1 = 0.0
    print(f"Training on {device}  |  epochs={num_epochs}  |  lr={lr}")
    print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        improved = val_metrics["macro_f1"] > best_f1
        if improved:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), save_path)

        print(
            f"Epoch {epoch:03d}/{num_epochs}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"train_acc={train_metrics['accuracy']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"val_F1={val_metrics['macro_f1']:.4f}"
            + (" ✓ saved" if improved else "")
        )

    print("\nBest macro-F1: {:.4f}".format(best_f1))
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model
