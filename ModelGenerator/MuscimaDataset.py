import os
import zipfile
import numpy
import shutil

from Dataset import Dataset
import random


class MuscimaDataset(Dataset):
    """ This dataset contains the Musicma Handwritten music scores database which consists of 
        1000 handwritten music scores from http://www.cvc.uab.es/cvcmuscima/index_database.html """

    def __init__(self, directory: str):
        super().__init__(directory)
        self.url = "http://www.cvc.uab.es/cvcmuscima/CVCMUSCIMA_WI.zip"
        self.dataset_filename = "CVCMUSCIMA_WI.zip"
        self.training_directory = os.path.join(self.directory, "training", "scores")
        self.validation_directory = os.path.join(self.directory, "validation", "scores")

    def is_dataset_cached_on_disk(self) -> bool:
        if len(os.listdir(self.training_directory)) == 900 and len(os.listdir(self.validation_directory)) == 100:
            return True

        return False

    def download_and_extract_dataset(self):
        if self.is_dataset_cached_on_disk():
            print("Dataset already downloaded and extracted")
            return

        if not os.path.exists(self.dataset_filename):
            self.download_file(self.url)

        absolute_image_directory = os.path.abspath(os.path.join(".", "temp", "scores"))
        self.clean_up_dataset_directories()
        self.extract_dataset_into_temp_folder()
        self.copy_images_from_subdirectories_into_single_directory(absolute_image_directory)
        self.split_images_into_training_and_validation_set(absolute_image_directory)
        self.clean_up_temp_directory()

    def clean_up_dataset_directories(self):
        """ Removes the dataset directories. Removes corrupted data or has no effect if nothing is in there """
        shutil.rmtree(self.training_directory)
        shutil.rmtree(self.validation_directory)

    def extract_dataset_into_temp_folder(self):
        print("Extracting dataset into temp directory")
        archive = zipfile.ZipFile(self.dataset_filename, "r")
        archive.extractall("temp")
        archive.close()

    @staticmethod
    def copy_images_from_subdirectories_into_single_directory(absolute_image_directory: str) -> str:
        os.makedirs(absolute_image_directory)
        relative_path_to_writers = os.path.join(".", "temp", "CVCMUSCIMA_WI", "PNG_GT_GRAY")
        absolute_path_to_writers = os.path.abspath(relative_path_to_writers)
        writer_directories_without_mac_system_directory = [os.path.join(absolute_path_to_writers, f)
                                                           for f in os.listdir(absolute_path_to_writers)
                                                           if f != ".DS_Store"]
        for writer_directory in writer_directories_without_mac_system_directory:
            images = [os.path.join(absolute_path_to_writers, writer_directory, f)
                      for f in os.listdir(writer_directory)
                      if f != ".DS_Store"]
            for image in images:
                destination_file = os.path.join(absolute_image_directory,
                                                os.path.basename(writer_directory) + "_" + os.path.basename(image))
                shutil.copyfile(image, destination_file)

        return absolute_image_directory

    def split_images_into_training_and_validation_set(self, absolute_image_directory: str):
        print("Creating training and validation sets")
        os.makedirs(self.training_directory)
        os.makedirs(self.validation_directory)
        validation_sample_indices = self.get_indices_of_validation_set()
        validation_files = numpy.array(os.listdir(absolute_image_directory))[validation_sample_indices]
        [shutil.move(os.path.abspath(os.path.join(absolute_image_directory, image)), self.validation_directory)
         for image in validation_files]

        training_files = os.listdir(absolute_image_directory)
        [shutil.move(os.path.abspath(os.path.join(absolute_image_directory, image)), self.training_directory)
         for image in training_files]

    def get_indices_of_validation_set(self, dataset_size: int = 1000, validation_size: int = 100) -> list:
        """ Returns a reproducible set of random sample indices from the entire dataset population """
        random.seed(0)
        validation_sample_indices = random.sample(range(0, dataset_size), validation_size)
        return validation_sample_indices

    def clean_up_temp_directory(self):
        print("Deleting temp directory")
        shutil.rmtree(os.path.abspath(os.path.join(".", "temp")))
