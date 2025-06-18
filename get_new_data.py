# save_dataset_splitter.py

import os
import torch
from tqdm import tqdm
from torchvision.utils import save_image
from paintings_dataset import PaintingsDataset


class DatasetSplitterSaver:
    def __init__(self, data_path: str, output_root: str, validation_split=0.2, test_split=0.2,
                 transform=False, padding=None, augment_abstrait=None):
        self.data_path = data_path
        self.output_root = output_root
        self.validation_split = validation_split
        self.test_split = test_split
        self.augment_abstrait = augment_abstrait

        self.dataset = PaintingsDataset(data_path=self.data_path,
                                        transform=transform,
                                        padding=padding)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def split_dataset(self):
        total_len = len(self.dataset)
        len_test = int(total_len * self.test_split)
        len_train = int(total_len * (1 - self.validation_split - self.test_split))
        len_val = total_len - len_train - len_test

        val_train_dataset, self.test_dataset = torch.utils.data.random_split(
            self.dataset, [len_train + len_val, len_test]
        )

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            val_train_dataset, [len_train, len_val]
        )

    def save_dataset_split(self, dataset, split_name: str):
        output_dir = os.path.join(self.output_root, split_name)
        os.makedirs(os.path.join(output_dir, "figuratif"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "abstrait"), exist_ok=True)

        for idx, sample in tqdm(enumerate(dataset), total=len(dataset), desc=f"Saving {split_name} set"):
            img_tensor = sample['image']
            label = sample['label']
            filename = f"img_{idx:05d}.png"

            if label == 0:
                save_path = os.path.join(output_dir, "figuratif", filename)
                save_image(img_tensor, save_path)
            else:
                orig_path = os.path.join(output_dir, "abstrait", f"orig_{filename}")
                save_image(img_tensor, orig_path)

                if split_name == "train" and self.augment_abstrait is not None:
                    augmented = self.augment_abstrait(img_tensor)
                    aug_path = os.path.join(output_dir, "abstrait", f"aug_{filename}")
                    save_image(augmented, aug_path)

    def process_and_save(self):
        self.split_dataset()
        self.save_dataset_split(self.train_dataset, "train")
        self.save_dataset_split(self.val_dataset, "val")
        self.save_dataset_split(self.test_dataset, "test")


if __name__ == "__main__":
    splitter = DatasetSplitterSaver(
        data_path='data',
        output_root='new_dataset/',
        transform=False,
        padding=None,
        augment_abstrait=None
    )
    splitter.process_and_save()
