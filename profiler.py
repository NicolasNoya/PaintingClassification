import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

CLASSES = [0, 1]

class Profiler:
    """
    A class to handle profiling, this class will log embeddings, confusion matrices,
    metrics, and other relevant information to TensorBoard for visualization and analysis.
    To use this class just create an instance of it and call the methods as needed.
    Args:
        log_dir (str): The directory where the TensorBoard logs will be saved.
    
    To see the logged data, run the following command in the terminal:
        tensorboard --logdir=log_dir/
    
    Then open your web browser and go to http://localhost:6006/
    """
    def __init__(self, log_dir='log_dir/'):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def embeddings_projector(self, embedding_tensor, labels):
        """
        This method logs the embeddings of the images into the TensorBoard projector.
        This will help visualize the embeddings in a 3D space and unreveal hidden patterns
        of the data.
        Args:
            embedding_tensor (torch.Tensor or numpy.ndarray): The tensor containing the embeddings.
            labels (torch.Tensor or numpy.ndarray): The labels corresponding to the embeddings.
        """
        # Ensure embedding_tensor is a torch tensor
        if not isinstance(embedding_tensor, torch.Tensor):
            embedding_tensor = torch.tensor(embedding_tensor)

        # Ensure labels is a torch tensor
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)

        # Check if embedding_tensor and labels have the same number of samples
        if embedding_tensor.size[0] != labels.size[0]:
            raise ValueError("Embedding tensor and labels must have the same number of samples.")

        # Log embeddings
        features = embedding_tensor.view(-1, embedding_tensor.size(-1))
        self.writer.add_embedding(features, metadata=labels.tolist())
        self.writer.close()

    def check_confusion_matrix(self, cm):
        """
        Computes and logs the confusion matrix as a heatmap.
        
        Args:
            cm (numpy.ndarray): The confusion matrix to log.
        """
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

        # Log to TensorBoard
        self.writer.add_figure("Confusion Matrix", fig)
        plt.close(fig)

    def log_metric(self, value, metric_name="Val Loss", step=0):
        """
        Logs a single metric value to TensorBoard.

        Args:
            value (float): The current value of the metric.
            metric_name (str): The name of the metric (e.g., "Loss", "Dice Score").
            step (int): The training step or epoch number.
        """
        self.writer.add_scalar(metric_name, value, step)
