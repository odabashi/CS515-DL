import random
import ssl
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import logging
import json
from parameters import get_params
from models.pretrained import get_pretrained_model
from models import MLP, MNIST_CNN, SimpleCNN, VGG, ResNet, BasicResBlock, MobileNetV2
from train import run_training
from test import run_test
from utils import visualize_model, setup_logger, compute_flops


# Fix for macOS SSL certificate verification error when downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context
setup_logger()
logger = logging.getLogger("HW3")


def set_seed(seed):
    """
    Fix all random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): Seed value applied to all RNG sources.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(params, teacher_model=False) -> nn.Module:
    """
    Instantiate the model specified in `params` (no weights loaded).

    Args:
        params (dict): Configuration dictionary (see `parameters.py`).
        teacher_model (bool): If True, use `params["teacher_model"]` as the architecture key; otherwise use
                              `params["model"]`.

    Returns:
        nn.Module: Randomly initialized model instance.

    Raises:
        ValueError: Unknown model name or architecture/dataset mismatch.
    """
    model_name = params["teacher_model"] if teacher_model else params["model"]
    dataset = params["dataset"]

    if model_name == "mlp":
        return MLP(
            input_size=params["input_size"],
            hidden_sizes=params["hidden_sizes"],
            hidden_activation=params["hidden_activation"],
            num_classes=params["num_classes"],
            enable_dropout=params["enable_dropout"],
            dropout=params["dropout"],
            enable_batch_norm=params["enable_batch_norm"]
        )
    if model_name == "cnn":
        # MNIST_CNN expects 1-channel 28×28; SimpleCNN expects 3-channel 32×32
        if dataset == "mnist":
            return MNIST_CNN(num_classes=params["num_classes"])
        else:
            return SimpleCNN(num_classes=params["num_classes"])

    if model_name == "vgg":
        if dataset == "mnist":
            raise ValueError("VGG is designed for 3-channel images; use cifar10 with vgg.")
        return VGG(depth=params["vgg_depth"], num_class=params["num_classes"])

    if model_name == "resnet":
        if dataset == "mnist":
            raise ValueError("ResNet is designed for 3-channel images; use cifar10 with resnet.")
        return ResNet(BasicResBlock, params["resnet_layers"], num_classes=params["num_classes"])

    if model_name == "mobilenet":
        if dataset == "mnist":
            raise ValueError("MobileNetV2 is designed for 3-channel images; use cifar10 with mobilenet.")
        return MobileNetV2(num_classes=params["num_classes"])

    raise ValueError(f"Unknown model: {model_name}")


def main():
    params = get_params()
    logger.info(f"Run parameters:\n{json.dumps(params, indent=4)}")

    set_seed(params["seed"])
    logger.info(f"Seed set to: {params['seed']}")
    logger.info(f"Dataset: {params['dataset']}  |  Model: {params['model']}")

    device = torch.device(
        params["device"] if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info(f"Using device: {device}")

    teacher_model: Optional[nn.Module] = None

    if params["enable_kd"] or params["eval_transfer"]:
        assert params["teacher_model_path"] is not None, ("Teacher model path must be provided when using Knowledge "
                                                          "Distillation or evaluating Transferability")

        teacher_model = build_model(params, teacher_model=True).to(device)
        teacher_model.load_state_dict(torch.load(params["teacher_model_path"], map_location=device))
        teacher_model.eval()

        # Freeze teacher
        for p in teacher_model.parameters():
            p.requires_grad = False

        logger.info("Teacher model loaded and frozen for Knowledge Distillation")

    if params["pretrained"]:
        model = get_pretrained_model(params=params, load_default_weights=True).to(device)
    else:
        model = build_model(params).to(device)
    logger.info(model)

    flops, params_count = compute_flops(model)

    logger.info(f"FLOPs: {flops}")
    logger.info(f"Model Parameters count: {params_count}")

    visualize_model(model, params)

    if params["mode"] in ("train", "both"):
        run_training(model, params, device, teacher_model)

    if params["mode"] in ("test", "both"):
        run_test(model, params, device, teacher_model=teacher_model)


if __name__ == "__main__":
    main()
