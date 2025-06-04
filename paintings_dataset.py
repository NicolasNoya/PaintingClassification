#%%
# Basic implementation of a custom dataset for loading paintings.
# TODO: Check the papers for the default data augmentation and transformation techniques.

import os 
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from typing import Union, Dict, Literal

default_augment = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    ])

default_transform = None

class PaintingsDataset(Dataset):
    """
    This is the custom dataset class for loading paintings from a directory.
    The input should be a the path to a ./data folder with two subfolders:
    
    ###############################################################################
    IMPORTANT: The directory structure should be as follows:
    - abstrait-v2 (where the abstract paintings are stored)
    - figuratif - aleat (where the figurative paintings are stored)
    ###############################################################################

    The dataset will return a list with the images and their corresponding labels.
    The labels are 0 for figurative and 1 for abstract.

    If the `augment` parameter is set to True, the dataset will apply data augmentation techniques.
    If the `transform` parameter is set to True, the dataset will apply transformations to the images and
    return both the transformed image and the non-transformed image.
    
    In case of `augment` being True, there will be a default augmentation applied to the images or 
    a custom augmentation can be passed as a parameter named `custom_augment`(if None we will use custom).
    In case of `transform` being True, the dataset will apply transformations to the images or 
    a custom transformation can be passed as a parameter named `custom_transform`(if None we will use custom).
    """
    def __init__(self, data_path, augment= False, transform=False, custom_augment=None, custom_transform=None, padding: Literal['zero','mirror','replicate'] = 'zero', image_input_size: int = 224):

        # Path to the data directory
        self.data_path = data_path
        
        # Paths to the abstract and figurative paintings directories
        self.abstract_path = os.path.join(data_path, 'abstrait-v2')
        self.figurative_path = os.path.join(data_path, 'figuratif - aleat')
        self.abstract_list = os.listdir(self.abstract_path)
        self.figurative_list = os.listdir(self.figurative_path)
        self.abstract_list.sort()
        self.figurative_list.sort()

        # length of the datasets
        self.len_abstract = len(self.abstract_list)
        self.len_figurative = len(self.figurative_list)
        self.total_length = self.len_abstract + self.len_figurative

        # Augmentation and transformation parameters
        self.augment = augment
        self.transform = transform
        self.augmentation = (custom_augment if custom_augment is not None else default_augment)
        self.transformation = (custom_transform if custom_transform is not None else default_transform)

        # Padding configuration
        if padding not in ['zero', 'mirror', 'replicate']:
            raise ValueError("Padding must be one of 'zero', 'mirror', or 'replicate'.")
        self.padding = padding
        self.image_input_size = image_input_size


    def __len__(self):
        return self.total_length

    def __getitem__(self, idx: int)->Dict[str, Union[torch.Tensor, int]]:
        """
        Get an item from the dataset using the index `idx` as the index to access the images.
        The method returns a tuple containing the image tensor and its corresponding label.
        If `transform` is True, it will return the transformed image as well as.
        If `augment` is True, it will apply data augmentation techniques to all the images outputed.
        """
        output = {}
        if idx < self.len_figurative:
            img_path = os.path.join(self.figurative_path, self.figurative_list[idx])
            label = 0
        else:
            img_path = os.path.join(self.abstract_path, self.abstract_list[idx - self.len_figurative])
            label = 1

        image = read_image(img_path)
        output['image'] = image
        output['label'] = label

        if self.augment:
            output['image'] = self.augmentation(output['image']) 
        if self.transform:
            output['image'] = self.transformation(output['image'])

        # Add the padding
        C, H, W = output['image'].shape
        # If the image is too big we just resize using nearest neighbor
        if H > self.image_input_size or W > self.image_input_size:
            HW_max = max(H, W)
            scale_factor = self.image_input_size / HW_max
            output['image'] = torch.nn.functional.interpolate(output['image'].unsqueeze(0), 
                                                                scale_factor=scale_factor, 
                                                                mode='nearest').squeeze(0) 
        C, H, W = output['image'].shape
        posH = max(0, (self.image_input_size - H) // 2)
        posW = max(0, (self.image_input_size - W) // 2)
        padding_tuple = (posW, posW + (self.image_input_size - W) % 2, posH, posH + (self.image_input_size - H) % 2)

        if self.padding == 'zero':
            padder = torch.nn.ZeroPad2d(padding_tuple)
        elif self.padding == 'mirror':
            padder = torch.nn.ReflectionPad2d(padding_tuple)
        elif self.padding == 'replicate':
            padder = torch.nn.ReplicationPad2d(padding_tuple) 
        
        output['image'] = padder(output['image'])
        return output

    def change_padding(self, padding: Literal['zero', 'mirror', 'replicate'], image_input_size: int = 224): 
        """
        Change the padding method used in the dataset.
        """
        if padding not in ['zero', 'mirror', 'replicate']:
            raise ValueError("Padding must be one of 'zero', 'mirror', or 'replicate'.")
        self.padding = padding
        self.image_input_size = image_input_size


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example usage
    dataset = PaintingsDataset(data_path='data', padding='replicate', augment=True)
    print(f"Dataset length: {len(dataset)}")
    random_index = torch.randint(0, len(dataset), (1,)).item()
    print(f"Random sample index: {random_index}")
    sample = dataset[7500]
    print(f"Sample image shape: {sample['image'].shape}, Label: {sample['label']}")
    plt.imshow(sample['image'].permute(1, 2, 0))  # Permute to (H, W, C) for plotting
    plt.show()
    print("Image size:", sample['image'].shape)