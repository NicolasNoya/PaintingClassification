
#%%
import torch
from enum import Enum

import tqdm

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from metric_manager import MetricManager
from paintings_dataset import PaintingsDataset, PaddingOptions
from profiler import Profiler

from PIL import ImageDraw, ImageFont
import torchvision.transforms.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2025)
torch.cuda.manual_seed(2025)

# Put here waht dataset to use for the embedding projection.
class TrainTestVal(str, Enum):
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'
    ALL = 'all'

# Put here the name of the models.
class ModelsName(str, Enum):
    TWO_RESNET = 'two_branch_resnet'
    RESNET = 'resnet50'


class Interface:
    """
    This is the Trainer class that will be used to train the models.
    It will handle the training loop, validation, and testing of the models.
    It will also import everything needed to train the models, such as the dataset and the model itself.
    Args:
        model (ModelsName): Model architecture to use ('two_branch_resnet' or 'resnet50').
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training and evaluation.
        freeze_layers (float): Fraction of layers to freeze (0 = train all, 1 = freeze all).
        learning_rate (float): Learning rate for the optimizer.
        loss_function (Callable): Loss function class (e.g., CrossEntropyLoss or BCEWithLogitsLoss).
        optimizer (Callable): Optimizer class (e.g., Adam, SGD).
        save_model_path (str): Path to save trained model weights.
        validation_split (float): Ratio of data to use for validation.
        test_split (float): Ratio of data to use for testing.
        logging_interval (int): Interval (in epochs) to log metrics and validate.
        num_workers (int): Number of worker threads for DataLoader.
        input_size (int): Input image size (default 224x224).
        augmentation (bool): Whether to apply basic data augmentation.
        use_fp16 (bool): If True, trains the model using mixed precision (float16).
        data_path (str): Base path to dataset directories (should include train/val/test).
        padding (PaddingOptions): Padding strategy for input images.
        weighted_loss (Tensor): Optional manual class weights for the loss function.
        profiling_path (str): Path to save TensorBoard logs and profiler data.
        load_model_path (str): Path to a checkpoint to resume training from.
        custom_augment_figuratif (Callable): Optional transform for figurative images only.
        custom_augment_abstrait (Callable): Optional transform for abstract images only.
        double_abstract (bool): If True, includes both original and augmented abstract images.
        n_transforms_augmented (int): Number of transforms to apply from augmentation pool (if used).
    """
    def __init__(
                    self, 
                    model: ModelsName = ModelsName.TWO_RESNET, 
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
                    weighted_loss: torch.Tensor = None,
                    profiling_path: str = "log_dir/",
                    load_model_path: str = None,
                    custom_augment_figuratif=None,
                    custom_augment_abstrait=None,
                    double_abstract: bool = False,
                    n_transforms_augmented = 2
                ):

        # Profiler
        self.profiler = Profiler(log_dir=profiling_path)

        # Model
        self.model_name = model
        self.freeze_layers = freeze_layers
        self.transform = False
        if self.model_name == ModelsName.TWO_RESNET:
            from models.two_branch_rnn import TwoBranchRNN
            self.model_instance = TwoBranchRNN(freeze_layers=freeze_layers).to(device)
            self.transform = True
        elif self.model_name == ModelsName.RESNET:
            from models.CustomResnet50 import CustomResNet50
            n_classes = 2
            if loss_function == torch.nn.BCEWithLogitsLoss:
                n_classes = 1

            self.model_instance = CustomResNet50(device=device,save_path=save_model_path,n_classes=n_classes).to(device)
            self.transform = False
        else:
            raise ValueError("Unsupported model type")
        
        if load_model_path:
            try:
                checkpoint = torch.load(load_model_path, map_location=device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.start_epoch = self.load_checkpoint(load_model_path)
                else:
                    self.model_instance.load_state_dict(checkpoint)
                    print(f"Loaded model weights from {load_model_path}")
            except Exception as e:
                print(f"Failed to load model from {load_model_path}: {e}")

        # Training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.model_instance.parameters(), lr=learning_rate,weight_decay=1e-4)
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
        
        self.dataset_train = PaintingsDataset(self.data_path+'train/',
                                        augment=self.augmentation, 
                                        transform=self.transform, 
                                        padding=self.padding, 
                                        image_input_size=self.input_size,
                                        custom_augment_abstrait=custom_augment_abstrait,
                                        custom_augment_figuratif=custom_augment_figuratif,
                                        double_abstract=double_abstract)
        
        self.dataset_val = PaintingsDataset(self.data_path+'val/',
                                        augment=False, 
                                        transform=self.transform, 
                                        padding=self.padding, 
                                        image_input_size=self.input_size,
                                        custom_augment_abstrait=None,
                                        custom_augment_figuratif=None)
        
        self.dataset_test = PaintingsDataset(self.data_path+'test/',
                                        augment=False, 
                                        transform=self.transform, 
                                        padding=self.padding, 
                                        image_input_size=self.input_size,
                                        custom_augment_abstrait=None,
                                        custom_augment_figuratif=None,)

        n_figurative = self.dataset_train.len_figurative #only for training
        n_abstract = self.dataset_train.len_abstract
        total = n_figurative + n_abstract

        class_weights = torch.tensor([
            total / n_figurative,  
            total / n_abstract    
        ], dtype=torch.float32).to(device)

        if weighted_loss is None or len(weighted_loss) != 2:
            weighted_loss = class_weights

        self.train_dataloader = DataLoader(self.dataset_train, 
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        num_workers=self.num_workers)
            
        self.val_dataloader = DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_dataloader = DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        if loss_function == torch.nn.BCEWithLogitsLoss:
            self.loss_function = loss_function()
        else:
            self.loss_function = loss_function(label_smoothing=0.1,weight=weighted_loss.to(device))
    



    def rnn_train(self):
        """
        Training rutine for the model RNN.
        This method will train the model on the training dataset and validate it on the validation dataset. 
        """
        self.model_instance.train()
        # for epochs, img_dict in enumerate(tqdm.tqdm(range(self.epochs), desc=f"Training Epoch {epochs + 1}/{self.epochs} - Metrics: {self.train_metric_manager.compute_metrics()}")):
        metric_dict = {"f1_score":0, "accuracy": 0}
        start = getattr(self, "start_epoch", 0)
        for epochs in range(start, self.epochs):
            self.train_metric_manager.reset_metrics()
            running_loss = 0.0
            # for img_dict in self.train_dataloader:
            for i, img_dict in enumerate(tqdm.tqdm(self.train_dataloader, desc=f"Training Epoch {epochs + 1}/{self.epochs} - Metrics: f1 score: {metric_dict['f1_score']}, accuracy: {metric_dict['accuracy']}")):
                if self.model_name == ModelsName.TWO_RESNET:
                    images = img_dict['image'].to(device)
                    images = (images/images.max()).float()  # Normalize the images
                    labels = img_dict['label'].to(device)
                    image_transformed = img_dict['transformed_image'].to(device)
                    image_transformed = (image_transformed/image_transformed.max()).float()
                    # Forward pass
                    outputs = self.model_instance(images, image_transformed)
                    # outputs = self.model_instance(images, images)
                elif self.model_name == ModelsName.RESNET:
                    images = img_dict['image'].to(device)
                    labels = img_dict['label'].to(device)

                    # Forward pass
                    outputs = self.model_instance(images)
                
                else:
                    raise ValueError("Unsupported model type for training")
                
                outputs = outputs.float().to(device)

                if isinstance(self.loss_function,torch.nn.BCEWithLogitsLoss):
                    labels = labels.float().to(device).unsqueeze(1)
                else:
                    labels = labels.long().to(device)   

                loss = self.loss_function(outputs, labels)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
                preds = None
                if isinstance(self.loss_function,torch.nn.BCEWithLogitsLoss):
                    preds = (outputs > 0).long()
                else:
                    preds = outputs.argmax(dim=1)

                self.train_metric_manager.update_metrics(preds, labels.long())
            metric_dict = self.train_metric_manager.compute_metrics()
            if (epochs) % self.logging_interval == 0 or (epochs+1) == self.epochs:
                print(f"Epoch [{epochs + 1}/{self.epochs}], Loss: {running_loss / self.logging_interval:.4f}")
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
        with torch.no_grad():
            for img_dict in self.val_dataloader:
                if self.model_name == ModelsName.TWO_RESNET:
                    images = img_dict['image'].to(device)
                    labels = img_dict['label'].to(device)
                    image_transformed = img_dict['transformed_image'].to(device)
                    # Forward pass
                    outputs = self.model_instance(images, image_transformed)
                elif self.model_name == ModelsName.RESNET:
                    images = img_dict['image'].to(device)
                    labels = img_dict['label'].to(device)
                    # Forward pass
                    outputs = self.model_instance(images)
                else:
                    raise ValueError("Unsupported model type for training")
                
                if isinstance(self.loss_function,torch.nn.BCEWithLogitsLoss):
                    labels = labels.float().to(device).unsqueeze(1)
                else:
                    labels = labels.long().to(device)   
                
                loss = self.loss_function(outputs, labels)
                running_loss += loss.item()

                preds = None

                if isinstance(self.loss_function,torch.nn.BCEWithLogitsLoss):
                    preds = (outputs > 0).long()
                else:
                    preds = outputs.argmax(dim=1)
                    
                self.val_metric_manager.update_metrics(preds, labels)

        metric_dict = self.val_metric_manager.compute_metrics()

        return running_loss, metric_dict 


    def test(self):
        """
        Test rutine for the model.
        This method will evaluate the model on the test dataset and return the loss and metrics
        dictionary.
        """
        correct_abstract = []
        wrong_abstract = []
        correct_figurative = []
        wrong_figurative = []

        self.model_instance.eval()
        self.test_metric_manager.reset_metrics()
        running_loss = 0.0
        with torch.no_grad():                    
            for img_dict in self.test_dataloader:
                if self.model_name == ModelsName.TWO_RESNET:
                    images = img_dict['image'].to(device)
                    labels = img_dict['label'].to(device)
                    image_transformed = img_dict['transformed_image'].to(device)
                    # Forward pass
                    outputs = self.model_instance(images, image_transformed)
                elif self.model_name == ModelsName.RESNET:
                    images = img_dict['image'].to(device)
                    labels = img_dict['label'].to(device)
                    # Forward pass
                    outputs = self.model_instance(images)
                else:
                    raise ValueError("Unsupported model type for training")
                
                if isinstance(self.loss_function, torch.nn.BCEWithLogitsLoss):
                    labels = labels.float().to(device)
                    labels_for_loss = labels.unsqueeze(1)
                    probs  = torch.sigmoid(outputs).squeeze(1)       # 0-1 prob. “abstracto”
                    preds  = (probs > 0.5).long()

                    labels_for_metrics = labels.long()

                    loss = self.loss_function(outputs, labels_for_loss)
                    running_loss += loss.item()
                    self.test_metric_manager.update_metrics(preds, labels_for_metrics)
                    
                else:   # CrossEntropy
                    labels = labels.long().to(device).squeeze()
                    probs  = torch.softmax(outputs, dim=1)[:, 1]     
                    preds  = (probs > 0.5).long()                    
                            
                    loss = self.loss_function(outputs, labels)
                    running_loss += loss.item()
                    self.test_metric_manager.update_metrics(preds, labels)

                for i in range(len(labels)):
                    label = labels[i].item()
                    pred = preds[i].item()
                    prob = probs[i].item()
                    img = images[i].cpu()

                    
                    pil_img = F.to_pil_image(img)
                    draw    = ImageDraw.Draw(pil_img)
                    txt  = f"{prob:.2f}"
                    # intenta usar una fuente TTF; si no, usa la default
                    try:
                        font = ImageFont.truetype("arial.ttf", 32)
                    except:
                        font = ImageFont.load_default()

                    if hasattr(draw, "textbbox"):
                        x0, y0, x1, y1 = draw.textbbox((0, 0), txt, font=font)
                        text_w, text_h = x1 - x0, y1 - y0
                    else:
                        text_w, text_h = draw.textsize(txt, font=font)

                    draw.rectangle([(0, 0), (text_w + 10, text_h + 10)], fill=(0, 0, 0, 128))
                    draw.text((5, 5), txt, fill=(255, 255, 255), font=font)


                    if label == 1 and pred == 1 and len(correct_abstract) < 5:
                        correct_abstract.append((pil_img, label, pred, prob))
                    elif label == 1 and pred == 0 and len(wrong_abstract) < 5:
                        wrong_abstract.append((pil_img, label, pred, prob))
                    elif label == 0 and pred == 0 and len(correct_figurative) < 5:
                        correct_figurative.append((pil_img, label, pred, prob))
                    elif label == 0 and pred == 1 and len(wrong_figurative) < 5:
                        wrong_figurative.append((pil_img, label, pred, prob))


        metric_dict = self.test_metric_manager.compute_metrics()

        return running_loss, metric_dict, [correct_abstract,wrong_abstract,correct_figurative,wrong_figurative]
    

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
            if self.model_name == ModelsName.TWO_RESNET:
                images = img_dict['image'].to(device)
                image_transformed = img_dict['transformed_image'].to(device)
                labels.extend(img_dict['label'].tolist())
                # Forward pass
                outputs = self.model_instance(images, image_transformed)
                embedding = self.model_instance.get_embeddings(images, image_transformed)
            elif self.model_name == ModelsName.RESNET:
                images = img_dict['image'].to(device)
                labels = img_dict['label'].to(device)
                # Forward pass
                outputs = self.model_instance(images)
            else:
                raise ValueError("Unsupported model type for training")
            
            list_of_arrays = [embedding.detach().cpu().numpy()[i] for i in range(embedding.shape[0])]
            embedding_tensor+= list_of_arrays
            # print("The shape of the embedding is", embedding.shape)
        # embedding_tensor = np.array(embedding_tensor)
        # print("The shape of the tensor is: ", embedding_tensor.shape)
        embedding_tensor = torch.tensor(embedding_tensor).view(-1, embedding.size(-1))
        labels = torch.tensor(labels)
        self.profiler.embeddings_projector(embedding_tensor, labels)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.model_instance.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.best_f1_score = checkpoint.get('best_f1_score', 0.0)
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Checkpoint loaded from {path}, starting at epoch {start_epoch}")
        return start_epoch



if __name__=="__main__":
    print("Starting training...")
    trainer = Interface(
        model= ModelsName.TWO_RESNET, 
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
        padding=PaddingOptions.ZERO,
        load_model_path = "./weights/best_f1_model0.7863.pth",
    )

    trainer.project_embeddings(TrainTestVal.TEST)

    # train_metrics_dict = trainer.rnn_train()
    # print(f"Training completed. Final metrics: {train_metrics_dict}")
