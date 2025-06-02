# Basic implementation of a custom dataset for loading paintings.
# TODO: Add data augmentation and transformations.

import os 
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from typing import Union, Dict


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
    def __init__(self, data_path, augment= False, transform=False, custom_augment=None, custom_transform=None):

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
        self.custom_augment = custom_augment
        self.custom_transform = custom_transform


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
            pass
        if self.transform:
            pass
        
        return output

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example usage
    dataset = PaintingsDataset(data_path='data')
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample image shape: {sample['image'].shape}, Label: {sample['label']}")
    plt.imshow(sample['image'].permute(1, 2, 0))  # Permute to (H, W, C) for plotting
    plt.show()