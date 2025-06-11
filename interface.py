#%%
import torch
from typing import Literal
from enum import Enum

import tqdm

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from metric_manager import MetricManager
from paintings_dataset import PaintingsDataset, PaddingOptions
from profiler import Profiler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2025)
torch.cuda.manual_seed(2025)

class TrainTestVal(str, Enum):
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'
    ALL = 'all'


class Interface:
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
                    weighted_loss: torch.Tensor = torch.Tensor([0.7, 0.3]),
                    profiling_path: str = "log_dir/",
                ):
        # Profiler
        self.profiler = Profiler(log_dir=profiling_path)

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
        self.loss_function = loss_function(weight=weighted_loss.to(device))
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


    def rnn_train(self):
        """
        Training rutine for the model RNN.
        This method will train the model on the training dataset and validate it on the validation dataset. 
        """
        self.model_instance.train()
        # for epochs, img_dict in enumerate(tqdm.tqdm(range(self.epochs), desc=f"Training Epoch {epochs + 1}/{self.epochs} - Metrics: {self.train_metric_manager.compute_metrics()}")):
        metric_dict = {"f1_score":0, "accuracy": 0}
        for epochs in range(self.epochs):
            self.train_metric_manager.reset_metrics()
            running_loss = 0.0
            # for img_dict in self.train_dataloader:
            for i, img_dict in enumerate(tqdm.tqdm(self.train_dataloader, desc=f"Training Epoch {epochs + 1}/{self.epochs} - Metrics: f1 score: {metric_dict['f1_score']}, accuracy: {metric_dict['accuracy']}")):
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
                print(f"Epoch [{epochs + 1}/{self.epochs}], Step [{epochs + 1}/{len(self.train_dataloader)}], Loss: {running_loss / self.logging_interval:.4f}")
                self.profiler.log_metric(running_loss, metric_name="Train Loss", step=epochs)
                self.profiler.log_metric(metric_dict["f1_score"], metric_name="Train F1 Score", step=epochs)
                self.profiler.log_metric(metric_dict["accuracy"], metric_name="Train Accuracy", step=epochs)
                running_loss = 0.0
                val_loss, val_dict = self.validate()
                self.profiler.log_metric(val_loss, metric_name="Val Loss", step=epochs)
                self.profiler.log_metric(val_dict["f1_score"], metric_name="Val F1 Score", step=epochs)
                self.profiler.log_metric(val_dict["accuracy"], metric_name="Val Accuracy", step=epochs)
                # Check the f1 score to save and the loss 
                if val_dict["f1_score"] > self.best_f1_score:
                    torch.save(self.model_instance.state_dict(), self.save_model_path+f"best_f1_model{val_dict['f1_score']:.4f}.pth")
                    print(f"Model saved to {self.save_model_path}, best F1 score: {val_dict['f1_score']:.4f}")
                    self.best_f1_score = val_dict["f1_score"]
                if val_loss  < self.best_loss:
                    torch.save(self.model_instance.state_dict(), self.save_model_path+f"best_loss_model{val_loss:.4f}.pth")
                    print(f"Model saved to {self.save_model_path}, best loss: {val_loss:.4f}")
                    self.best_loss = val_loss
                print("The validation scores are:")
                print(val_dict)
                print(val_loss)
                # update metrics
        # Get the final confusion matrix
        val_metric_dict = self.val_metric_manager.compute_metrics()
        self.profiler.check_confusion_matrix(val_metric_dict["confusion_matrix"])
        return metric_dict 
    

    def validate(self):
        """
        Validate rutine for the model.
        This method will evaluate the model on the validation dataset and return the loss and metrics
        dictionary.
        """
        self.model_instance.eval()
        self.val_metric_manager.reset_metrics()
        running_loss = 0.0
        for img_dict in self.val_dataloader:
            if self.model_name == "two_branch_resnet":
                images = img_dict['image'].to(device)
                labels = img_dict['label'].to(device)
                image_transformed = img_dict['transformed_image'].to(device)
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
        """
        Test rutine for the model.
        This method will evaluate the model on the test dataset and return the loss and metrics
        dictionary.
        """
        self.model_instance.eval()
        self.test_metric_manager.reset_metrics()
        running_loss = 0.0
        for img_dict in self.test_dataloader:
            if self.model_name == "two_branch_resnet":
                images = img_dict['image'].to(device)
                labels = img_dict['label'].to(device)
                image_transformed = img_dict['transformed_image'].to(device)
                # Forward pass
                outputs = self.model_instance(images, image_transformed)
            else:
                raise ValueError("Unsupported model type for training")
            
            loss = self.loss_function(outputs, labels)
            running_loss += loss.item()
            self.test_metric_manager.update_metrics(outputs.argmax(dim=1), labels)

        metric_dict = self.test_metric_manager.compute_metrics()

        return running_loss, metric_dict 
    

    def project_embeddings(self, dset: TrainTestVal = TrainTestVal.TRAIN):
        """
        This method logs the embeddings of the images into the TensorBoard projector.
        This will help visualize the embeddings in a 3D space and unreveal hidden patterns
        of the data.
        Args:
            embedding_tensor (torch.Tensor or numpy.ndarray): The tensor containing the embeddings.
            labels (torch.Tensor or numpy.ndarray): The labels corresponding to the embeddings.
        """
        if dset == TrainTestVal.TRAIN:
            dataloader = self.train_dataloader
        elif dset == TrainTestVal.VAL:
            dataloader = self.val_dataloader
        elif dset == TrainTestVal.TEST:
            dataloader = self.test_dataloader
        elif dset == TrainTestVal.ALL:
            dataloader = DataLoader(
                self.dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=self.num_workers
            )
        else:
            raise ValueError("Invalid dataset type. Use TrainTestVal.TRAIN, TrainTestVal.VAL, or TrainTestVal.TEST.")
        
        embedding_tensor = []
        labels = []
        for img_dict in dataloader:
            if self.model_name == "two_branch_resnet":
                images = img_dict['image'].to(device)
                image_transformed = img_dict['transformed_image'].to(device)
                labels.extend(img_dict['label'].tolist())
                # Forward pass
                outputs = self.model_instance(images, image_transformed)
            else:
                raise ValueError("Unsupported model type for training")
            
            embedding_tensor.append(outputs.detach().cpu().numpy())
        embedding_tensor = torch.tensor(embedding_tensor).view(-1, outputs.size(-1))
        labels = torch.tensor(labels)
        self.profiler.embeddings_projector(embedding_tensor, labels)


if __name__=="__main__":
    print("Starting training...")
    trainer = Interface(
        model="two_branch_resnet", 
        epochs=50, 
        batch_size=64, 
        freeze_layers=0.8, 
        learning_rate=0.001, 
        loss_function=CrossEntropyLoss, 
        optimizer=Adam, 
        save_model_path="weights/",
        validation_split=0.2,
        test_split=0.2,
        logging_interval=5,
        num_workers=8,
        input_size=224,
        augmentation=True,
        use_fp16=True,
        data_path="data",
        padding=PaddingOptions.ZERO
    )
    train_metrics_dict = trainer.rnn_train()
    print(f"Training completed. Final metrics: {train_metrics_dict}")
