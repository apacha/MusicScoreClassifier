import os
from glob import glob
from typing import Dict

import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset


class ScoreClassificationDataset(Dataset):

    """
    :param dataset_directory: Absolute path to the directory that must contain two
                              folders named 'scores' and 'other' that will be used for the classification
    """
    def __init__(self, dataset_directory, transform=None) -> None:
        self.transform = transform
        self.dataset_directory = dataset_directory
        score_image_paths = glob(os.path.join(dataset_directory, "scores/*.*"))
        other_image_paths = glob(os.path.join(dataset_directory, "other/*.*"))
        self.all_images = score_image_paths + other_image_paths
        self.classes = [0] * len(score_image_paths) + [1] * len(other_image_paths)
        self.classnames = {0: "score", 1:"other"}

    def __len__(self) -> int:
        return len(self.all_images)

    def __getitem__(self, index) -> Dict[str, object]:
        image = Image.open(self.all_images[index])

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'class': self.classes[index]}

        return sample

    def show_sample(self, index):
        """Show image with landmarks"""
        plt.imshow(self[index]["image"])
        plt.show()