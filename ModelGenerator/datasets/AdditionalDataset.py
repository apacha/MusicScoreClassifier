import os

import numpy

from datasets.Dataset import Dataset


class AdditionalDataset(Dataset):
    """ Loads images from a custom directory and splits them into train and validation
        directories with a random generator """

    def __init__(self, destination_directory: str,
                 source_directory: str = "C:\\Users\\Alex\\Dropbox\\Doktorat\\MusicScoresDataset"):
        """
        Create and initializes a new dataset.
        :param destination_directory: The root directory, into which the data will be placed.
        :param source_directory: The root directory, that contains additional samples. Note that the directory
        must contain two subfolders: scores and other, from where the samples will be copied
        """
        super().__init__(destination_directory)
        self.source_directory = source_directory
        self.scores_directory = os.path.join(self.source_directory, "scores")
        self.other_samples_directory = os.path.join(self.source_directory, "other")

        self.number_of_score_samples = len(os.listdir(self.scores_directory))
        self.number_of_other_samples = len(os.listdir(self.other_samples_directory))

    def is_dataset_cached_on_disk(self):
        return True

    def download_and_extract_dataset(self, cleanup_data_directory=False):

        self.split_images_into_training_and_validation_set(self.scores_directory,
                                                           os.path.join(self.training_directory, "scores"),
                                                           os.path.join(self.validation_directory, "scores"),
                                                           self.number_of_score_samples,
                                                           int(numpy.math.ceil(self.number_of_score_samples * 0.1)))
        self.split_images_into_training_and_validation_set(self.other_samples_directory,
                                                           os.path.join(self.training_directory, "other"),
                                                           os.path.join(self.validation_directory, "other"),
                                                           self.number_of_other_samples,
                                                           int(numpy.math.ceil(self.number_of_other_samples * 0.1)))
