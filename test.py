import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging
from train import get_transforms
from utils import ClassificationMetrics, plot_confusion_matrix


logger = logging.getLogger("HW1")


@torch.no_grad()
def run_test(model, params, device):
    tf = get_transforms(params)

    if params["dataset"] == "mnist":
        test_ds = datasets.MNIST(params["data_dir"], train=False, download=True, transform=tf)
    else:
        test_ds = None

    loader = DataLoader(test_ds, batch_size=params["batch_size"], shuffle=False, num_workers=params["num_workers"])

    # Load best weights
    model.load_state_dict(torch.load(params["save_path"], map_location=device))
    model.eval()

    metrics = ClassificationMetrics(params["num_classes"], device)

    correct, n = 0, 0
    class_correct = [0] * params["num_classes"]
    class_total = [0] * params["num_classes"]

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)

        # Calculate Support
        correct += preds.eq(labels).sum().item()
        n += imgs.size(0)
        for p, t in zip(preds, labels):
            class_correct[t] += (p == t).item()
            class_total[t] += 1

        # Calculate Evaluation Scores
        metrics.update(preds, labels)

    results = metrics.compute()

    logger.info("\n=== Test Metrics ===")

    logger.info(f"- Accuracy:  {results['accuracy']:.4f} ({correct}/{n})")
    logger.info(f"- Precision: {results['precision']:.4f}")
    logger.info(f"- Recall:    {results['recall']:.4f}")
    logger.info(f"- F1 Score:  {results['f1']:.4f}")
    logger.info(f"- Support:")
    for i in range(params["num_classes"]):
        acc = class_correct[i] / class_total[i]
        logger.info(f"\tClass {i}: {acc:.4f}  ({class_correct[i]}/{class_total[i]})")

    plot_confusion_matrix(results["confusion_matrix"])
