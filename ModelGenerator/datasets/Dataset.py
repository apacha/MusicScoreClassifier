import os
import random
import shutil
import urllib.parse as urlparse
import urllib.request as urllib2
from abc import ABC, abstractmethod

import numpy


class Dataset(ABC):
    """ The abstract base class for the datasets used to train the model """

    def __init__(self,
                 destination_directory: str):
        """
        :param destination_directory: The root directory, into which the data will be placed.
        Inside of this directory, the following structure contains the data:
         
         directory
         |- training
         |   |- other
         |   |- scores
         |
         |- validation
         |   |- other
         |   |- scores
         
        """
        self.destination_directory = os.path.abspath(destination_directory)
        self.training_directory = os.path.join(self.destination_directory, "training")
        self.validation_directory = os.path.join(self.destination_directory, "validation")
        self.dataset_size = 0
        self.number_of_training_samples = 0
        self.number_of_validation_samples = 0

    def is_dataset_cached_on_disk(self) -> bool:
        if not (os.path.exists(self.training_directory) and os.path.exists(self.validation_directory)):
            return False

        if len(os.listdir(self.training_directory)) == self.number_of_training_samples \
                and len(os.listdir(self.validation_directory)) == self.number_of_validation_samples:
            return True

        return False

    @abstractmethod
    def download_and_extract_dataset(self, cleanup_data_directory=False):
        """ Starts the download of the dataset and extracts it into the directory specified in the constructor """
        pass

    def get_random_validation_sample_indices(self, dataset_size: int = 1000, validation_sample_size: int = 100) -> list:
        """  Returns a reproducible set of random sample indices from the entire dataset population        """
        random.seed(0)
        validation_sample_indices = random.sample(range(0, dataset_size), validation_sample_size)
        return validation_sample_indices

    def split_images_into_training_and_validation_set(self, absolute_path_to_images: str,
                                                      destination_training_directory: str = None,
                                                      destination_validation_directory: str = None,
                                                      dataset_size: int = None,
                                                      number_of_validation_samples: int = None):
        print("Creating training and validation sets")
        if destination_training_directory is None:
            destination_training_directory = self.training_directory
        if destination_validation_directory is None:
            destination_validation_directory = self.validation_directory
        if dataset_size is None:
            dataset_size = self.dataset_size
        if number_of_validation_samples is None:
            number_of_validation_samples = self.number_of_validation_samples

        os.makedirs(destination_training_directory, exist_ok=True)
        os.makedirs(destination_validation_directory, exist_ok=True)
        validation_sample_indices = self.get_random_validation_sample_indices(dataset_size,
                                                                              number_of_validation_samples)
        validation_files = numpy.array(os.listdir(absolute_path_to_images))[validation_sample_indices]
        for image in validation_files:
            shutil.copy(os.path.abspath(os.path.join(absolute_path_to_images, image)),
                        destination_validation_directory)

        training_files = os.listdir(absolute_path_to_images)
        for image in training_files:
            shutil.copy(os.path.abspath(os.path.join(absolute_path_to_images, image)), destination_training_directory)

    def clean_up_temp_directory(self, temp_directory):
        print("Deleting temp directory")
        shutil.rmtree(temp_directory)

    def clean_up_dataset_directories(self):
        """ Removes the dataset directories. Removes corrupted data or has no effect if nothing is in there """
        shutil.rmtree(self.training_directory, ignore_errors=True)
        shutil.rmtree(self.validation_directory, ignore_errors=True)

    def download_file(self, url, desc=None) -> str:
        u = urllib2.urlopen(url)
        scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
        filename = os.path.basename(path)
        if not filename:
            filename = 'downloaded.file'
        if desc:
            filename = os.path.join(desc, filename)

        with open(filename, 'wb') as f:
            meta = u.info()
            meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
            meta_length = meta_func("Content-Length")
            file_size = None
            if meta_length:
                file_size = int(meta_length[0])
            print("Downloading: {0} Bytes: {1}".format(url, file_size))

            file_size_dl = 0
            block_sz = 8192
            status_counter = 0
            status_output_interval = 100
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break

                file_size_dl += len(buffer)
                f.write(buffer)
                status = "{0:16}".format(file_size_dl)
                if file_size:
                    status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
                status += chr(13)
                status_counter += 1
                if status_counter == status_output_interval:
                    status_counter = 0
                    print(status)
                    # print(status, end="", flush=True) Does not work unfortunately
            print()

        return os.path.abspath(filename)
