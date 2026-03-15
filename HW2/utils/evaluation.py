from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix
)


class ClassificationMetrics:
    """
    Wrapper around TorchMetrics for MNIST classification.
    """
    def __init__(self, num_classes, device):
        self.precision = MulticlassPrecision(num_classes=num_classes, average="macro").to(device)
        self.recall = MulticlassRecall(num_classes=num_classes, average="macro").to(device)
        self.f1 = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    def update(self, predictions, targets):
        self.precision.update(predictions, targets)
        self.recall.update(predictions, targets)
        self.f1.update(predictions, targets)
        self.confusion_matrix.update(predictions, targets)

    def compute(self):
        return {
            "precision": self.precision.compute().item(),
            "recall": self.recall.compute().item(),
            "f1": self.f1.compute().item(),
            "confusion_matrix": self.confusion_matrix.compute().cpu()
        }
