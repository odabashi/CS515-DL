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
from utils import (ClassificationMetrics, plot_confusion_matrix, plot_tsne, measure_runtime, pgd_l2_attack,
                   pgd_linf_attack)
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

    if params.get("pgd_eval"):
        logger.info("Evaluating model robustness under PGD attack...")
        pgd_results = run_pgd_eval(model, params, device)


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
    logger.info(f"{'Corruption':<20} " + "  Accuracy")
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


def run_pgd_eval(model, params, device) -> Dict[str, float]:
    """
    Evaluate model robustness under PGD adversarial attack.

    Uses the standard test set (no corruption), but feeds adversarially
    perturbed images to the model. Supports both L-inf and L2 threat models.

    Args:
        model (nn.Module): Trained model, already on `device` and in eval mode.
        params (dict): Configuration dictionary. Expected keys under params:
            - eps (float): perturbation budget in pixel scale for both L-inf and L2
            - alpha (float): step size in pixel scale
            - steps (int): number of PGD iterations
            - mean (list): mean of the dataset
            - std (list): standard deviation of the dataset
        device (torch.device): Compute device.

    Returns:
        Dict[str, float]: {
            "clean_accuracy": float,
            "adv_accuracy": float,
            "robust_accuracy_drop": float,
        }
    """
    alpha = params["pgd_alpha"]
    eps_l2 = params["pgd_eps_l2"]
    eps_linf = params["pgd_eps_linf"]
    steps = params["pgd_steps"]
    mean = params["mean"]
    std = params["std"]

    # Freeze model params explicitly during attack
    for p in model.parameters():
        p.requires_grad_(False)

    data_loader = get_test_data_loader(params)
    model.eval()

    results = {}

    logger.info("============ PGD Adversarial Evaluation ============")
    for norm in ["L2", "L-Inf"]:
        eps = eps_l2 if norm == "L2" else eps_linf
        attack_fn = pgd_linf_attack if norm == "L-Inf" else pgd_l2_attack
        logger.info("=" * 50)
        logger.info(f"Norm: {norm} | eps: {eps:.5f} | alpha: {alpha:.5f} | steps: {steps}")
        logger.info("-" * 50)

        clean_correct, clean_acc, adv_correct, adv_acc, n = 0, 0, 0, 0, 0

        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # --- Clean accuracy (no grad needed) ---
            with torch.no_grad():
                clean_preds = torch.argmax(model(imgs), dim=1)
                clean_correct += clean_preds.eq(labels).sum().item()

            # --- Adversarial accuracy ---
            # attack_fn handles model.eval() internally and returns normalized adv images
            adv_imgs = attack_fn(
                model=model,
                images=imgs,
                labels=labels,
                mean=mean,
                std=std,
                eps=eps,
                # alpha=alpha,
                steps=steps,
            )

            with torch.no_grad():
                adv_preds = torch.argmax(model(adv_imgs), dim=1)
                adv_correct += adv_preds.eq(labels).sum().item()

            n += imgs.size(0)

        clean_acc = clean_correct / n
        adv_acc = adv_correct / n
        acc_drop = clean_acc - adv_acc

        logger.info(f"Clean Accuracy:          {clean_acc:.4f} ({clean_correct}/{n})")
        logger.info(f"Adversarial Accuracy:    {adv_acc:.4f}  ({adv_correct}/{n})")
        logger.info(f"Robust Accuracy Drop:    {acc_drop:.4f}")
        logger.info("=" * 30)

        results[norm] = {
            "clean_accuracy":        clean_acc,
            "adv_accuracy":          adv_acc,
            "robust_accuracy_drop":  acc_drop,
        }

    return results
