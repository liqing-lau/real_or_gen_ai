import argparse
from typing import Dict, Any

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from torchvision import transforms

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

CLASS_NAMES = ["ai", "real"] 

def predict_image_with_resnet(model_path: str, image_path: str) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load image
    img = Image.open(image_path).convert("RGB")
    x = _TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    probs = probs.cpu()

    prob_ai = float(probs[0])
    prob_real = float(probs[1])

    idx = int(probs.argmax().item())
    label = CLASS_NAMES[idx]

    return {
        "label": label,
        "prob_ai": prob_ai,
        "prob_real": prob_real,
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference on a single image."
    )
    parser.add_argument("--model-path", required=True, help="Path to .pt file")
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    result = predict_image_with_resnet(args.model_path, args.image)
    print(result)


if __name__ == "__main__":
    main()