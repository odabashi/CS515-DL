import random
import ssl
import numpy as np
import torch
import logging
import json
from datetime import datetime
from parameters import get_params
from models.pretrained import get_pretrained_model
from models import MLP, MNIST_CNN, SimpleCNN, VGG, ResNet, BasicResBlock, MobileNetV2
from train import run_training
from test import run_test
from utils import visualize_model, setup_logger


# Fix for macOS SSL certificate verification error when downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context
setup_logger()
logger = logging.getLogger("HW2")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(params):

    model_name = params["model"]
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

    if params["pretrained"]:
        model = get_pretrained_model(params=params, load_default_weights=True).to(device)
    else:
        model = build_model(params).to(device)
    logger.info(model)

    visualize_model(model, params)

    training_start_time = datetime.now()
    if params["mode"] in ("train", "both"):
        run_training(model, params, device)
    training_end_time = datetime.now()
    training_elapsed = (training_end_time - training_start_time).total_seconds()
    logger.info(f"Training took {training_elapsed:.2f}s")

    if params["mode"] in ("test", "both"):
        test_start_time = datetime.now()
        run_test(model, params, device)
        test_end_time = datetime.now()
        test_elapsed = (test_end_time - test_start_time).total_seconds()
        logger.info(f"Testing (Together with plotting) took {test_elapsed:.2f}s")


if __name__ == "__main__":
    main()
