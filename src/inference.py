import argparse
from typing import Dict, Any

import torch
from PIL import Image
from torchvision import transforms

from model import build_model


IMG_SIZE = 224

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def predict_image(model_path: str, image_path: str) -> Dict[str, Any]:
    """
    Run inference on a single image.

    Returns:
        {
            "label": "ai" or "real",
            "prob_ai": float,
            "prob_real": float,
        }
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(model_path, map_location=device)
    backbone = checkpoint.get("backbone", "efficientnet_b0")
    classes = checkpoint["classes"]

    model = build_model(
        backbone=backbone,
        num_classes=len(classes),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    img = Image.open(image_path).convert("RGB")
    x = _TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    idx = int(probs.argmax().item())
    label = classes[idx]

    result: Dict[str, Any] = {"label": label}
    if "ai" in classes:
        result["prob_ai"] = float(probs[classes.index("ai")])
    if "real" in classes:
        result["prob_real"] = float(probs[classes.index("real")])

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference on a single image."
    )
    parser.add_argument("--model-path", required=True, help="Path to .pt file")
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    result = predict_image(args.model_path, args.image)
    print(result)


if __name__ == "__main__":
    main()

