#%%
import torch
from typing import Literal
from enum import Enum
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

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

# Augmentations for the abstract images in the training set
class SkewTransform:
    def __init__(self, magnitude_range=(0.1, 0.4), direction="horizontal"):
        self.magnitude_range = magnitude_range
        self.direction = direction
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)

        width, height = img.size
        magnitude = random.uniform(*self.magnitude_range)

        if self.direction == "horizontal":
            xshift = magnitude * width
            matrix = (1, xshift / height, 0, 0, 1, 0)
        else:  # vertical
            yshift = magnitude * height
            matrix = (1, 0, 0, yshift / width, 1, 0)

        skewed = img.transform(img.size, Image.AFFINE, matrix)
        return self.to_tensor(skewed)

class RandomStretchFixedSide:
    def __init__(self, deform_side: Literal["width", "height"], deform_range=(300, 450)):
        self.deform_side = deform_side
        self.deform_range = deform_range
        self.center_crop = transforms.CenterCrop(224)

    def __call__(self, img):
        deform_value = random.randint(*self.deform_range)

        if self.deform_side == "width":
            size = (224, deform_value)
        else:
            size = (deform_value, 224)

        img = transforms.Resize(size)(img)
        img = self.center_crop(img)
        return img

transform_pool = [
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomVerticalFlip(p=1.0),
    RandomStretchFixedSide("width", deform_range=(300, 450)),
    RandomStretchFixedSide("height", deform_range=(300, 450)),
    SkewTransform(magnitude_range=(0.1, 0.4), direction="horizontal"),
    SkewTransform(magnitude_range=(0.1, 0.4), direction="vertical"),
]

class AugmentedOnlyTrainDataset(Dataset):
    def __init__(self, base_dataset, n_augments=3, transform_pool=None, n_transforms_per_aug=2):
        self.base_dataset = base_dataset
        self.samples = []
        self.transform_pool = transform_pool
        self.n_transforms_per_aug = n_transforms_per_aug

        if transform_pool is None:
            self.transform_pool = [
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomVerticalFlip(p=1.0),
                RandomStretchFixedSide("width", deform_range=(300, 450)),
                RandomStretchFixedSide("height", deform_range=(300, 450)),
                SkewTransform(magnitude_range=(0.1, 0.4), direction="horizontal"),
                SkewTransform(magnitude_range=(0.1, 0.4), direction="vertical"),
            ]

        for i in range(len(base_dataset)):
            label = base_dataset[i]['label']
            self.samples.append(i)
            if label == 1:
                for j in range(1, n_augments + 1):
                    self.samples.append((i, True, j))

        random.shuffle(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        if isinstance(sample_info, tuple):
            base_idx, _, _ = sample_info # the third was the aug_id
            sample = self.base_dataset[base_idx]
            sample = sample.copy()
            image = sample['image'].clone()

            # composition
            selected = random.sample(self.transform_pool, self.n_transforms_per_aug)
            for t in selected:
                image = t(image)

            sample['image'] = image
            #sample['aug_index'] = aug_id

            if self.base_dataset.transform:
                
                C, H, W = image.shape
                transform_size = min(min(H, W) * 0.3, self.base_dataset.image_input_size)
                center_crop = transforms.CenterCrop(transform_size)
                patch = center_crop(image.clone())
                if transform_size < self.base_dataset.image_input_size:
                    patch = torch.nn.functional.interpolate(
                        patch.unsqueeze(0),
                        size=(self.base_dataset.image_input_size, self.base_dataset.image_input_size),
                        mode='bilinear'
                    ).squeeze(0)
                sample['transformed_image'] = patch.float()

            return sample
        else:
            return self.base_dataset[sample_info]

    def __len__(self):
        return len(self.samples)

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
                    model: ModelsName.TWO_RESNET, 
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
                    custom_augment_figuratif=None,                      #???????delete?????
                    custom_augment_abstrait=None,                       #???????delete?????
                    transform_pool: list = transform_pool,
                    n_transforms_per_aug: int = 2
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
            if load_model_path:
                self.model_instance.load_state_dict(torch.load(load_model_path))
            self.transform = True
        elif self.model_name == ModelsName.RESNET:
            from models.CustomResnet50 import CustomResNet50
            self.model_instance = CustomResNet50(device=device,save_path=save_model_path,n_classes=2).to(device)
            self.transform = False
        else:
            raise ValueError("Unsupported model type")
        
        # Training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
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
                                        image_input_size=self.input_size,
                                        custom_augment_abstrait=custom_augment_abstrait,
                                        custom_augment_figuratif=custom_augment_figuratif)
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

        print(f"Train dataset size before augmentation: {len(self.train_dataset)}")
        # augmenting the abstract images of the training set
        self.train_dataset = AugmentedOnlyTrainDataset(
                            self.train_dataset,
                            n_augments=n_augments_abstrait,
                            transform_pool=transform_pool,
                            n_transforms_per_aug=2
                        )

        print(f"Train dataset size after augmentation: {len(self.train_dataset)}")

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        n_figurative = self.dataset.len_figurative
        n_abstract = self.dataset.len_abstract
        total = n_figurative + n_abstract

        class_weights = torch.tensor([
            total / n_figurative,  
            total / n_abstract    
        ], dtype=torch.float32).to(device)

        if weighted_loss is None or len(weighted_loss) != 2:
            weighted_loss = class_weights

        self.loss_function = loss_function(weight=weighted_loss.to(device))




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
                labels = labels.long().to(device)
                loss = self.loss_function(outputs, labels)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
                self.train_metric_manager.update_metrics(outputs.argmax(dim=1), labels)
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
        correct_abstract = []
        wrong_abstract = []
        correct_figurative = []
        wrong_figurative = []

        self.model_instance.eval()
        self.test_metric_manager.reset_metrics()
        running_loss = 0.0
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
            
            loss = self.loss_function(outputs, labels)
            running_loss += loss.item()
            self.test_metric_manager.update_metrics(outputs.argmax(dim=1), labels)

            preds = outputs.argmax(dim=1)

            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].item()
                img = images[i].cpu()

                if label == 1 and pred == 1 and len(correct_abstract) < 5:
                    correct_abstract.append((img, label, pred))
                elif label == 1 and pred == 0 and len(wrong_abstract) < 5:
                    wrong_abstract.append((img, label, pred))
                elif label == 0 and pred == 0 and len(correct_figurative) < 5:
                    correct_figurative.append((img, label, pred))
                elif label == 0 and pred == 1 and len(wrong_figurative) < 5:
                    wrong_figurative.append((img, label, pred))

            

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
        n_augments_abstrait=3,
        transform_pool=transform_pool,
        n_transforms_per_aug=2
    )

    trainer.project_embeddings(TrainTestVal.TEST)

    # train_metrics_dict = trainer.rnn_train()
    # print(f"Training completed. Final metrics: {train_metrics_dict}")
