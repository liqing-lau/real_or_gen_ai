from typing import Any

import timm
import torch.nn as nn


def build_model(
    backbone: str = "efficientnet_b0",
    num_classes: int = 2,
    pretrained: bool = True,
) -> Any:
    """
    Construct a classification model using a timm backbone.
    """
    model = timm.create_model(backbone, pretrained=pretrained)

    # timm models expose a helper to reset the classifier head
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes=num_classes)
    else:
        # Fallback for models without reset_classifier helper
        classifier = getattr(model, "classifier", None) or getattr(
            model, "fc", None
        )
        if classifier is None:
            raise ValueError(
                f"Backbone {backbone} does not expose a classifier/fc head"
            )
        in_features = classifier.in_features
        new_head = nn.Linear(in_features, num_classes)
        if hasattr(model, "classifier"):
            model.classifier = new_head
        else:
            model.fc = new_head

    return model

