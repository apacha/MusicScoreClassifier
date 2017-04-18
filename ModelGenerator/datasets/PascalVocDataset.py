import os
import shutil
import tarfile

import numpy

from datasets.Dataset import Dataset


class PascalVocDataset(Dataset):
    """ This dataset contains the Pascal VOC 2006 challenge database which consists over 
        2618 images of ten categories from http://host.robots.ox.ac.uk/pascal/VOC/databases.html#VOC2006 """

    def __init__(self, destination_directory: str):
        super().__init__(destination_directory)
        self.url = "http://host.robots.ox.ac.uk/pascal/VOC/download/voc2006_trainval.tar"
        self.dataset_filename = "voc2006_trainval.tar"
        self.training_directory = os.path.join(self.destination_directory, "training", "other")
        self.validation_directory = os.path.join(self.destination_directory, "validation", "other")
        self.dataset_size = 2618
        self.number_of_training_samples = int(numpy.math.ceil(self.dataset_size * 0.9))
        self.number_of_validation_samples = self.dataset_size - self.number_of_training_samples

    def download_and_extract_dataset(self, cleanup_data_directory=False):
        if self.is_dataset_cached_on_disk():
            print("Pascal VOC Dataset already downloaded and extracted")
            return

        if not os.path.exists(self.dataset_filename):
            self.download_file(self.url)

        temp_directory = os.path.abspath(os.path.join(".", "VOCdevkit"))
        absolute_image_directory = os.path.abspath(os.path.join(".", "VOCdevkit", "VOC2006", "PNGImages"))
        if cleanup_data_directory:
            self.clean_up_dataset_directories()
        self.extract_dataset_into_temp_folder(temp_directory)
        self.split_images_into_training_and_validation_set(absolute_image_directory)
        self.clean_up_temp_directory(temp_directory)

    def extract_dataset_into_temp_folder(self, temp_directory: str):
        print("Extracting Pascal VOC dataset into temp directory")
        if os.path.exists(temp_directory):
            shutil.rmtree(temp_directory)
        tar = tarfile.open(self.dataset_filename, "r:")
        tar.extractall()
        tar.close()
