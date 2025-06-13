import torch
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

class CustomResNet50(torch.nn.Module):
    """
    Wrapper class around a pretrained ResNet-50 model for binary or multi-class image classification,
    with support for fine-tuning and model checkpointing.

    This class freezes all layers of the pretrained ResNet-50 except for the final fully connected layer,
    which is replaced and trained to adapt to a custom dataset (e.g., "figuratif" vs "abstrait").

    Parameters:
    ----------
    device : torch.device
        The device (e.g., "cuda" or "cpu") on which the model will be trained and evaluated.

    save_path : str
        Path to save the best model checkpoint during training (based on validation loss).

    n_classes : int
        Number of output classes for the final classification layer.

    Methods:
    -------
    forward(x):
        Forward pass through the model.

    train_model(data_loader, optimizer, criterion, device=None):
        Runs one epoch of training on the given data loader.

    test_model(data_loader, criterion, device=None):
        Evaluates the model on validation or test data.

    fit(train_loader, val_loader, criterion, optimizer, n_epochs, scheduler=None):
        Full training loop across epochs, with optional learning rate scheduler
        and automatic checkpointing of the best model.

    save_checkpoint():
        Saves the model weights to the specified path.

    load_checkpoint():
        Loads the model weights from the specified path.
    """

    def __init__(self, device, save_path, n_classes):
        super(CustomResNet50, self).__init__()

        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        #freeze parameters of the model.
        #we freeze everything and we only train last layer so it learns to classify abstrait and figuratif
        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features=in_features, out_features=n_classes)

        for name, param in self.model.named_parameters():
            if name.startswith("layer4") or name.startswith("fc"):
                param.requires_grad = True

        self.device = device
        self.save_path = save_path
        self.model.to(self.device)

    def forward(self, x):
        return self.model(x)
