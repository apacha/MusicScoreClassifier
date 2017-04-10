import os
import tarfile

import shutil

from Dataset import Dataset


class PascalVocDataset(Dataset):
    """ This dataset contains the Pascal VOC 2006 challenge database which consists over 
        5000 images of ten categories from http://host.robots.ox.ac.uk/pascal/VOC/databases.html#VOC2006 """

    def __init__(self, directory: str):
        super().__init__(directory)
        self.url = "http://host.robots.ox.ac.uk/pascal/VOC/download/voc2006_trainval.tar"
        self.training_directory = os.path.join(self.directory, "training", "other")
        self.validation_directory = os.path.join(self.directory, "training", "other")

    def is_dataset_cached_on_disk(self) -> bool:
        pass

    def download_dataset(self):
        self.download_file(self.url)
        downloaded_archive = "voc2006_trainval.tar"
        tar = tarfile.open(downloaded_archive, "r:")
        tar.extractall()
        tar.close()
        pass

extracted_archive = "VOCdevkit"
png_images = os.path.join(extracted_archive, "VOC2006", "PNGImages")
files = [os.path.abspath(os.path.join(png_images,f)) for f in os.listdir(png_images)]
target_directory = os.path.join(directory)
shutil.move()