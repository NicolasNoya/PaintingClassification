#%%
import torch
from typing import Literal

import tqdm

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss, MSELoss

from metric_manager import MetricManager
from paintings_dataset import PaintingsDataset, PaddingOptions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """
    This is the Trainer class that will be used to train the models.
    It will handle the training loop, validation, and testing of the models.
    It will also import everything needed to train the models, such as the dataset and the model itself.
    Args:
    # TODO: Add more arguments to the constructor
    """
    def __init__(
                    self, 
                    model: Literal["two_branch_resnet"], 
                    epochs: int = 10, 
                    batch_size: int = 32,   
                    freeze_layers: float = 0.8, 
                    learning_rate: float = 0.001, 
                    loss_function: torch.optim = CrossEntropyLoss,
                    optimizer: torch.optim = Adam,
                    save_model_path: str = "weights/",
                    validation_split: float = 0.2,
                    test_split: float = 0.2,
                    logging_interval: int = 10,
                    num_workers: int = 4,
                    input_size: int = 224,
                    augmentation: bool = False, 
                    use_fp16: bool = True, # To train the model in half precision
                    data_path: str = "data",
                    padding: PaddingOptions = PaddingOptions.ZERO,
                ):
        # Model
        self.model_name = model
        self.freeze_layers = freeze_layers
        self.transform = False
        if self.model_name == "two_branch_resnet":
            from models.two_branch_rnn import TwoBranchRNN
            self.model_instance = TwoBranchRNN(freeze_layers=freeze_layers).to(device)
            self.transform = True
        else:
            raise ValueError("Unsupported model type")
        
        # Training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_function = loss_function()
        self.optimizer = optimizer(self.model_instance.parameters(), lr=learning_rate)
        self.save_model_path = save_model_path
        self.validation_split = validation_split
        self.test_split = test_split
        self.logging_interval = logging_interval
        self.best_loss = float('inf')
        self.best_f1_score = 0.0
        
        # Computation configuration
        self.num_workers = num_workers
        self.use_fp16 = use_fp16

        # Metrics
        self.train_metric_manager = MetricManager()
        self.val_metric_manager = MetricManager()
        self.test_metric_manager = MetricManager()

        # Data configuration
        self.data_path = data_path
        self.augmentation = augmentation
        self.input_size = input_size
        self.padding = padding

        # Initialize the dataset and dataloader
        self.dataset = PaintingsDataset(self.data_path,
                                        augment=self.augmentation, 
                                        transform=self.transform, 
                                        padding=self.padding, 
                                        image_input_size=self.input_size)
        len_test = int(len(self.dataset) * self.test_split)
        len_train = int(len(self.dataset) * (1 - self.validation_split - self.test_split))
        len_val = len(self.dataset) - len_train - len_test
        val_train_dataset, self.test_dataset = torch.utils.data.random_split(
            self.dataset, 
            [len_train + len_val, len_test],
        )
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            val_train_dataset, 
            [len_train, len_val],
        )
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def train(self):
        self.model_instance.train()
        # for epochs, img_dict in enumerate(tqdm.tqdm(range(self.epochs), desc=f"Training Epoch {epochs + 1}/{self.epochs} - Metrics: {self.train_metric_manager.compute_metrics()}")):
        metric_dict = {}
        for epochs in range(self.epochs):
            self.train_metric_manager.reset_metrics()
            running_loss = 0.0
            # for img_dict in self.train_dataloader:
            for i, img_dict in enumerate(tqdm.tqdm(self.train_dataloader, desc=f"Training Epoch {epochs + 1}/{self.epochs} - Metrics: {metric_dict}")):
                if self.model_name == "two_branch_resnet":
                    images = img_dict['image'].to(device)
                    images = (images/images.max()).float()  # Normalize the images
                    labels = img_dict['label'].to(device)
                    image_transformed = img_dict['transformed_image'].to(device)
                    image_transformed = (image_transformed/image_transformed.max()).float()
                    # Forward pass
                    outputs = self.model_instance(images, image_transformed)
                    # outputs = self.model_instance(images, images)
                else:
                    raise ValueError("Unsupported model type for training")
                
                outputs = outputs.float().to(device)
                labels = labels.long().to(device)
                loss = self.loss_function(outputs, labels)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
                self.train_metric_manager.update_metrics(outputs.argmax(dim=1), labels)
            metric_dict = self.train_metric_manager.compute_metrics()
            if (epochs) % self.logging_interval == 0:
                continue
                print(f"Epoch [{epochs + 1}/{self.epochs}], Step [{epochs + 1}/{len(self.train_dataloader)}], Loss: {running_loss / self.logging_interval:.4f}")
                running_loss = 0.0
                val_loss, val_dict = self.validate()
                # Check the f1 score to save and the loss 
                if val_dict["f1_score"] > self.best_f1_score:
                    torch.save(self.model_instance.state_dict(), self.save_model_path+"best_f1_model.pth")
                    print(f"Model saved to {self.save_model_path}, best F1 score: {val_dict['f1_score']:.4f}")
                if val_loss  < self.best_loss:
                    torch.save(self.model_instance.state_dict(), self.save_model_path+"best_loss_model.pth")
                    print(f"Model saved to {self.save_model_path}, best loss: {val_loss:.4f}")
                    
                # update metrics

        return metric_dict 
    
    def validate(self):
        self.model_instance.eval()
        self.val_metric_manager.reset_metrics()
        running_loss = 0.0
        for img_dict in self.val_dataloader:
            if self.model_name == "two_branch_resnet":
                images = img_dict['image']
                labels = img_dict['label']
                image_transformed = img_dict['transformed_image']
                # Forward pass
                outputs = self.model_instance(images, image_transformed)
            else:
                raise ValueError("Unsupported model type for training")
            
            loss = self.loss_function(outputs, labels)
            running_loss += loss.item()
            self.val_metric_manager.update_metrics(outputs.argmax(dim=1), labels)

        metric_dict = self.val_metric_manager.compute_metrics()

        return running_loss, metric_dict 

    def test(self):
        self.model_instance.eval()
        self.test_metric_manager.reset_metrics()
        running_loss = 0.0
        for img_dict in self.test_dataloader:
            if self.model_name == "two_branch_resnet":
                images = img_dict['image']
                labels = img_dict['label']
                image_transformed = img_dict['transformed_image']
                # Forward pass
                outputs = self.model_instance(images, image_transformed)
            else:
                raise ValueError("Unsupported model type for training")
            
            loss = self.loss_function(outputs, labels)
            running_loss += loss.item()
            self.test_metric_manager.update_metrics(outputs.argmax(dim=1), labels)

        metric_dict = self.test_metric_manager.compute_metrics()

        return running_loss, metric_dict 

if __name__=="__main__":
    print("Starting training...")
    trainer = Trainer(
        model="two_branch_resnet", 
        epochs=10, 
        batch_size=32, 
        freeze_layers=0.8, 
        learning_rate=0.001, 
        loss_function=CrossEntropyLoss, 
        optimizer=Adam, 
        save_model_path="weights/",
        validation_split=0.2,
        test_split=0.2,
        logging_interval=1,
        num_workers=4,
        input_size=224,
        augmentation=True,
        use_fp16=True,
        data_path="data",
        padding=PaddingOptions.ZERO
    )
    train_metrics_dict = trainer.train()
    print(f"Training completed. Final metrics: {train_metrics_dict}")
