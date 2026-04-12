import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import logging
from train import get_transforms
from utils import ClassificationMetrics, plot_confusion_matrix, plot_tsne, measure_runtime


logger = logging.getLogger("HW3")


@measure_runtime
@torch.no_grad()
def run_test(model, params, device):
    tf = get_transforms(params, train=False)

    if params["dataset"] == "mnist":
        test_ds = datasets.MNIST(params["data_dir"], train=False, download=True, transform=tf)
    elif params["dataset"] == "cifar10":
        test_ds = datasets.CIFAR10(params["data_dir"], train=False, download=True, transform=tf)
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

    all_logits = []
    all_labels = []

    for imgs, labels in loader:
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

        # Calculate Evaluation Scores
        metrics.update(preds, labels)

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

        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()

        plot_tsne(logits, labels)

