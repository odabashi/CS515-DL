from typing import Dict, List
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import datasets
import logging
from train import get_transforms
from utils import ClassificationMetrics, plot_confusion_matrix, plot_tsne, measure_runtime
from data_loaders import Cifar10cDataset, CIFAR_10_C_CORRUPTIONS


logger = logging.getLogger("HW3")


def get_test_data_loader(params):
    tf = get_transforms(params, train=False)

    if params["dataset"] == "mnist":
        test_ds = datasets.MNIST(params["data_dir"], train=False, download=True, transform=tf)
    elif params["dataset"] == "cifar10":
        test_ds = datasets.CIFAR10(params["data_dir"], train=False, download=True, transform=tf)
    else:
        test_ds = None

    return DataLoader(test_ds, batch_size=params["batch_size"], shuffle=False, num_workers=params["num_workers"])


def run_eval(model, data_loader, num_classes, device, metrics=None):
    correct, n = 0, 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            # Save for t-SNE
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

            # Calculate Support
            correct += preds.eq(labels).sum().item()
            n += imgs.size(0)
            for p, t in zip(preds, labels):
                class_correct[t] += (p == t).item()
                class_total[t] += 1

            if metrics:
                # Calculate Evaluation Scores
                metrics.update(preds, labels)

    return class_correct, class_total, correct, n, all_logits, all_labels, metrics


@measure_runtime
@torch.no_grad()
def run_test(model, params, device, teacher_model=None):
    data_loader = get_test_data_loader(params)

    # Load best weights
    model.load_state_dict(torch.load(params["save_path"], map_location=device))
    model.eval()

    metrics = ClassificationMetrics(params["num_classes"], device)

    class_correct, class_total, correct, n, all_logits, all_labels, metrics = run_eval(model, data_loader,
                                                                                       params["num_classes"], device,
                                                                                       metrics)

    results = metrics.compute()

    logger.info("\n============ Test Metrics ============")

    logger.info(f"=> Accuracy:  {(correct / n):.4f} ({correct}/{n}), Precision: {results['precision']:.4f}, "
                f"Recall: {results['recall']:.4f}, F1 Score: {results['f1']:.4f}")
    logger.info("======================================")
    logger.info(f"- Support:")
    for i in range(params["num_classes"]):
        acc = class_correct[i] / class_total[i]
        logger.info(f"\tClass {i}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")

    logger.info("Visualizing Confusion Matrix")
    logger.info(results["confusion_matrix"])
    plot_confusion_matrix(results["confusion_matrix"])

    if params.get("plot_tsne", False):
        logger.info("Generating t-SNE visualization...")
        logits_np = torch.cat(all_logits).numpy()
        labels_np = torch.cat(all_labels).numpy()
        plot_tsne(logits_np, labels_np)

    if params.get("eval_corrupted", False):
        logger.info("Evaluating model against CIFAR-10-C corrupted images...")
        corruption_results = run_corrupted_test(model, params, device)


def run_corrupted_test(model, params, device) -> Dict[str, float]:
    """
    Evaluate the model on the full CIFAR-10-C benchmark.

    Iterates over all 19 corruption types and records accuracy for each type. Logs a summary table and computes the
    mean corruption accuracy (mCA) and mean corruption error (mCE) averaged across all corruptions.

    Args:
        model (nn.Module): Trained model. Expected to already be on `device` and in eval mode.
        params (dict): Configuration dictionary.
        device (torch.device): Compute device.

    Returns:
        Dict[str, float]: Nested dict results[corruption_name] = accuracy.

    Raises:
        FileNotFoundError: Propagated from Cifar10cDataset if the CIFAR-10-C directory or any `.npy` file is missing.
    """
    cifar10c_dir = params["cifar10c_dir"]

    if not os.path.isdir(cifar10c_dir):
        raise FileNotFoundError(
            f"CIFAR-10-C directory not found: {cifar10c_dir}\n Download from https://zenodo.org/record/2535967 and "
            f"extract there."
        )

    # Only `ToTensor` + `Normalize` transforms are applied (no augmentation) to match the original benchmark protocol
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(params["mean"], params["std"]),
    ])

    model.eval()

    results: Dict[str, float] = {}
    all_accuracies: List[float] = []  # flat list for global mCA

    logger.info("\n====== CIFAR-10-C Corruption Evaluation ======")
    logger.info(f"{'Corruption':<20} " + "  Mean")
    logger.info("-" * 30)

    for corruption in CIFAR_10_C_CORRUPTIONS:
        ds = Cifar10cDataset(
            data_dir=cifar10c_dir,
            corruption=corruption,
            severity=-1,
            transform=tf,
        )
        data_loader = DataLoader(
            ds,
            batch_size=params["batch_size"],
            shuffle=False,
            num_workers=params["num_workers"],
        )
        _, _, correct, n, _, _, _ = run_eval(model, data_loader, params["num_classes"], device)
        single_corruption_accuracy = (correct / n)
        results[corruption] = single_corruption_accuracy
        all_accuracies.append(single_corruption_accuracy)

        logger.info(f"{corruption:<20}  {single_corruption_accuracy:.4f}")

    mCA = float(np.mean(all_accuracies))  # mean corruption accuracy
    mCE = 1.0 - mCA  # mean corruption error

    logger.info("=" * 30)
    logger.info(f"Mean Corruption Accuracy (mCA): {mCA:.4f}")
    logger.info(f"Mean Corruption Error    (mCE): {mCE:.4f}")
    logger.info("=" * 30)

    return results

