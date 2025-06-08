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

        self.device = device
        self.save_path = save_path
        self.model.to(self.device)

    def forward(self, x):
        return self.model(x)

    def train_model(self, data_loader, optimizer, criterion, device=None):
        if device is None:
            device = self.device

        self.model.train()
        cumulative_loss = 0.0
        cumulative_correct = 0
        total_samples = 0

        for inputs, targets in tqdm(data_loader, desc="Training", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            y_pred = self.model(inputs)
            loss = criterion(y_pred, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            cumulative_loss += loss.item() * batch_size
            total_samples += batch_size
            _, predicted = y_pred.max(1)
            cumulative_correct += predicted.eq(targets).sum().item()

        avg_loss = cumulative_loss / total_samples
        avg_accuracy = 100.0 * (cumulative_correct / total_samples)
        return avg_loss, avg_accuracy

    def test_model(self, data_loader, criterion, device=None):
        if device is None:
            device = self.device

        self.model.eval()
        cumulative_loss = 0.0
        cumulative_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                y_pred = self.model(inputs)
                loss = criterion(y_pred, targets)

                batch_size = inputs.size(0)
                cumulative_loss += loss.item() * batch_size
                total_samples += batch_size
                _, predicted = y_pred.max(1)
                cumulative_correct += predicted.eq(targets).sum().item()

        avg_loss = cumulative_loss / total_samples
        avg_accuracy = 100.0 * (cumulative_correct / total_samples)
        return avg_loss, avg_accuracy

    def save_checkpoint(self):
        torch.save(self.model.state_dict(), self.save_path)

    def load_checkpoint(self):
        state_dict = torch.load(self.save_path)
        self.model.load_state_dict(state_dict)

    def fit(self, train_loader, val_loader, criterion, optimizer, n_epochs, scheduler=None):
        best_val_loss = float('inf')

        for epoch in range(1, n_epochs + 1):
            train_loss, train_acc = self.train_model(train_loader, optimizer, criterion, self.device)
            val_loss, val_acc     = self.test_model(val_loader,     criterion, self.device)

            print(f"Epoch {epoch}/{n_epochs} — "
                  f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.2f}% | "
                  f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.2f}%")

            #this is if we use scheduler (to change the lr)
            if scheduler is not None:
                scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint()
                print(f"  Saved best model (val_loss = {val_loss:.4f})")
