import torch.nn as nn
from torchvision import models
from typing import Optional
import logging


logger = logging.getLogger("HW3")


def get_pretrained_model(params: dict, load_default_weights: bool = True, weights: Optional[object] = None):
    """
    Returns a pretrained model adapted for CIFAR-10 with optional pretrained weights, resizing, and freezing.

    Args:
        params (dict): Configuration parameters, expects keys:
            - model: str
            - num_classes: int
            - resize_input: bool, True for Option 1 (resize to 224), False for Option 2
            - freeze_features: bool, True to freeze conv layers, False to fine-tune all
        load_default_weights (bool): If True, loads default pretrained ImageNet weights.
        weights (Optional[object]): TorchVision weights object to load instead of default.

    Returns:
        nn.Module: Adapted pretrained model.
    """
    if params["model"] == "vgg":
        if not load_default_weights:
            model = models.vgg16(weights=weights)
        else:
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    else:
        raise ValueError(f"Unknown model: {params["model"]}")

    # ------------------------
    # Option 2: No Resize (32x32)
    # ------------------------
    if not params["resize_input"]:
        if params["model"] == "vgg":
            # Replace avgpool to handle smaller input
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            # Adjust classifier to match smaller feature map (512 * 1 * 1)
            model.classifier = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(512, params["num_classes"])
            )

            # Ensure we fine-tune all layers
            params["freeze_features"] = False

    # ------------------------
    # Option 1: Resize (224x224)
    # ------------------------
    else:
        if params["model"] == "vgg":
            # Replace classifier last layer based on the class number
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, params["num_classes"])

    # ------------------------
    # Freeze features if requested (Option 1)
    # ------------------------
    if params["freeze_features"]:
        if params["model"] == "vgg":
            # Freeze only convolutional layers
            for model_parameter in model.features.parameters():
                model_parameter.requires_grad = False

            # Ensure classifier is trainable
            for model_parameter in model.classifier.parameters():
                model_parameter.requires_grad = True
    else:
        if params["model"] == "vgg":
            # Fine-tune all layers
            for model_parameter in model.parameters():
                model_parameter.requires_grad = True

    # ------------------------
    # Optional: print trainable layers
    # ------------------------
    logger.info(f"\n[Pretrained {params['model']}] Trainable parameters:")
    for name, param in model.named_parameters():
        logger.info(f"{name}: requires_grad={param.requires_grad}")

    return model
