import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from dataset import get_dataloaders
from model import build_model


def _run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    num_classes: int,
    pos_label: int | None,
    train: bool,
):
    if train:
        model.train()
    else:
        model.eval()

    loss_sum = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()  # type: ignore[union-attr]

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()  # type: ignore[union-attr]

            loss_sum += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

            if num_classes == 2 and pos_label is not None:
                probs = torch.softmax(outputs, dim=1)[:, pos_label]
                y_score.extend(probs.detach().cpu().tolist())

    total = len(y_true)
    loss_avg = loss_sum / total

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
    }

    average = "binary" if num_classes == 2 else "macro"
    metrics["precision"] = precision_score(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics["recall"] = recall_score(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics["f1"] = f1_score(
        y_true, y_pred, average=average, zero_division=0
    )

    if num_classes == 2 and pos_label is not None and y_score:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_score)
        except ValueError:
            metrics["auc_roc"] = None
    else:
        metrics["auc_roc"] = None

    return loss_avg, metrics


def train(
    backbone: str = "efficientnet_b0",
    stage1_epochs: int = 3,
    stage2_epochs: int = 7,
    stage1_lr: float = 1e-3,
    stage2_lr: float = 3e-4,
    batch_size: int = 32,
    sample_ratio: float | None = None,
    device: str | None = None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, classes = get_dataloaders(
        batch_size=batch_size,
        sample_ratio=sample_ratio,
    )

    model = build_model(
        backbone=backbone,
        num_classes=len(classes),
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # metrics logging setup
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "train_log.csv"
    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "stage",
                    "epoch",
                    "phase",
                    "loss",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "auc_roc",
                ]
            )

    def log_metrics(stage: str, epoch: int, phase: str, loss: float, m: dict):
        with log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    stage,
                    epoch,
                    phase,
                    loss,
                    m.get("accuracy"),
                    m.get("precision"),
                    m.get("recall"),
                    m.get("f1"),
                    m.get("auc_roc"),
                ]
            )

    best_val_loss = float("inf")
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoints_dir / f"{backbone}-best.pt"

    # ---------------- Stage 1: 只训练分类头 ----------------
    print("Stage 1: train classifier head only")
    for name, param in model.named_parameters():
        if "classifier" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=stage1_lr,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    pos_label = None
    if len(classes) == 2:
        if "ai" in classes:
            pos_label = classes.index("ai")
        else:
            pos_label = 1

    for epoch in range(1, stage1_epochs + 1):
        train_loss, train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes=len(classes),
            pos_label=pos_label,
            train=True,
        )
        val_loss, val_metrics = _run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
            num_classes=len(classes),
            pos_label=pos_label,
            train=False,
        )
        scheduler.step(val_loss)

        log_metrics("stage1", epoch, "train", train_loss, train_metrics)
        log_metrics("stage1", epoch, "val", val_loss, val_metrics)

        print(
            f"[Stage1][{epoch}/{stage1_epochs}] "
            f"train_loss={train_loss:.4f} acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_loss:.4f} acc={val_metrics['accuracy']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "backbone": backbone,
                    "classes": classes,
                },
                ckpt_path,
            )
            print(f"Saved best model to {ckpt_path}")

    # ---------------- Stage 2: 解冻全模型 fine-tune ----------------
    print("Stage 2: fine-tune full model")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=stage2_lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    for epoch in range(1, stage2_epochs + 1):
        train_loss, train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes=len(classes),
            pos_label=pos_label,
            train=True,
        )
        val_loss, val_metrics = _run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
            num_classes=len(classes),
            pos_label=pos_label,
            train=False,
        )
        scheduler.step(val_loss)

        log_metrics("stage2", epoch, "train", train_loss, train_metrics)
        log_metrics("stage2", epoch, "val", val_loss, val_metrics)

        print(
            f"[Stage2][{epoch}/{stage2_epochs}] "
            f"train_loss={train_loss:.4f} acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_loss:.4f} acc={val_metrics['accuracy']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "backbone": backbone,
                    "classes": classes,
                },
                ckpt_path,
            )
            print(f"Saved best model to {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientNet to classify AI vs real images."
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b0",
        help="timm backbone name, e.g. efficientnet_b0",
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=3,
        help="number of epochs for head-only training",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=7,
        help="number of epochs for full fine-tuning",
    )
    parser.add_argument(
        "--stage1-lr",
        type=float,
        default=1e-3,
        help="learning rate for head-only training",
    )
    parser.add_argument(
        "--stage2-lr",
        type=float,
        default=3e-4,
        help="learning rate for full fine-tuning",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=None,
        help="if set (0<r<1), train/val on a random subset of data",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu (default: auto-detect)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        backbone=args.backbone,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr,
        batch_size=args.batch_size,
        sample_ratio=args.sample_ratio,
        device=args.device,
    )

