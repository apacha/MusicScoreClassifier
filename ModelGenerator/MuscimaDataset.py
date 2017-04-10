import os
import tarfile

import shutil

from Dataset import Dataset


class MuscimaDataset(Dataset):
    """ This dataset contains the Musicma Handwritten music scores database which consists of 
        1000 handwritten music scores from http://www.cvc.uab.es/cvcmuscima/index_database.html """

    def __init__(self, directory: str):
        super().__init__(directory)
        self.url = "http://www.cvc.uab.es/cvcmuscima/CVCMUSCIMA_WI.zip"
        self.training_directory = os.path.join(self.directory, "training", "scores")
        self.validation_directory = os.path.join(self.directory, "training", "scores")

    def is_dataset_cached_on_disk(self) -> bool:
        pass

    def download_dataset(self):
        self.download_file(self.url)
        pass

dataset = MuscimaDataset("data")
dataset.download_dataset()