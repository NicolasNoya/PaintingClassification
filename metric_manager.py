
import torch
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MetricManager:
    """
    A class to manage and compute various evaluation metrics for model performance.
    This class uses PyTorch's torchmetrics library to compute accuracy, F1 score, precision, and recall. 
    """
    def __init__(self):
        """
        Initializes the MetricManager with the required metrics.
        """
        self.accuracy = Accuracy(task="binary").to(device)
        self.f1_score = F1Score(num_classes=2, average='macro').to(device)
        self.precision = Precision(num_classes=2, average='macro').to(device)
        self.recall = Recall(num_classes=2, average='macro').to(device)

    def update_metrics(self, preds, targets):
        """
        Evaluate the model's performance using various metrics.
        Args:
            preds (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth labels.
        Returns:
            dict: Dictionary containing accuracy, F1 score, precision, and recall.
        """
        self.accuracy.update(preds, targets)
        self.f1_score.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)

    def compute_metrics(self):
        """
        Computes the metrics and returns them as a dictionary.
        Returns:
            dict: Dictionary containing accuracy, F1 score, precision, and recall.
        """
        return {
            "accuracy": self.accuracy.compute().item(),
            "f1_score": self.f1_score.compute().item(),
            "precision": self.precision.compute().item(),
            "recall": self.recall.compute().item()
        }
    
    def reset_metrics(self):
        """
        Resets the metrics to their initial state.
        This is useful for starting a new evaluation phase.
        """
        self.accuracy.reset()
        self.f1_score.reset()
        self.precision.reset()
        self.recall.reset()