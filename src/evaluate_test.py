import argparse
from pathlib import Path
from typing import List

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader
from torchvision import datasets

from dataset import build_transforms
from model import build_model


def evaluate(
    model_path: str,
    test_dir: str = "data/test",
    batch_size: int = 32,
    device: str | None = None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    test_root = Path(test_dir)
    if not test_root.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    _, val_transform = build_transforms()
    test_ds = datasets.ImageFolder(test_dir, transform=val_transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    checkpoint = torch.load(model_path, map_location=device)
    backbone = checkpoint.get("backbone", "efficientnet_b0")
    classes: List[str] = checkpoint["classes"]

    model = build_model(
        backbone=backbone,
        num_classes=len(classes),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

            # For binary AUC, use probability of positive class (here: 'real')
            if len(classes) == 2:
                if "real" in classes:
                    pos_idx = classes.index("real")
                else:
                    pos_idx = 1
                y_score.extend(probs[:, pos_idx].cpu().tolist())

    # Metrics
    average = "binary" if len(classes) == 2 else "macro"

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    auc_roc = None
    if len(classes) == 2 and y_score:
        try:
            # sklearn 当前版本不再接受 pos_label 参数，
            # 直接用 y_true 中的 1 作为正类；此时我们传入的是 real 的概率。
            auc_roc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc_roc = None

    print("=== Test metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    if auc_roc is not None:
        print(f"AUC-ROC  : {auc_roc:.4f}")
    else:
        print("AUC-ROC  : N/A (not enough positive/negative samples)")

    # Confusion matrix & per-class report
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=classes,
            zero_division=0,
            digits=4,
        )
    )

    # Optional: save metrics to file for later analysis
    out_dir = Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "test_metrics.txt"
    with metrics_path.open("w") as f:
        f.write("=== Test metrics ===\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n")
        if auc_roc is not None:
            f.write(f"AUC-ROC  : {auc_roc:.4f}\n")
        else:
            f.write("AUC-ROC  : N/A\n")
        f.write("\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(f"{cm}\n\n")
        f.write("Classification report:\n")
        f.write(
            classification_report(
                y_true,
                y_pred,
                target_names=classes,
                zero_division=0,
                digits=4,
            )
        )

    print(f"\nSaved detailed metrics to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the held-out test set."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/efficientnet_b0-best.pt",
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/test",
        help="Root directory of test set (ImageFolder layout)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Test batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu (default: auto-detect)",
    )
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()

