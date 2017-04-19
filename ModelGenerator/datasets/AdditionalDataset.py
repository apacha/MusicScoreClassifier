import os
import shutil

import numpy

from datasets.Dataset import Dataset


class AdditionalDataset(Dataset):
    """ Loads images from a custom directory and splits them into train and validation
        directories with a random generator """

    def __init__(self, destination_directory: str,
                 source_directory: str):
        """
        Create and initializes a new dataset.
        :param destination_directory: The root directory, into which the data will be placed.
        :param source_directory: The root directory, that contains additional samples. Note that the directory
        must contain two subfolders: scores and other, from where the samples will be copied
        """
        super().__init__(destination_directory)
        self.source_directory = source_directory

    def download_and_extract_dataset(self):
        print("Copying additional images from {0} and its subdirectories".format(self.source_directory))
        path_to_all_files = [os.path.join(self.source_directory, "scores", file) for file in os.listdir(os.path.join(self.source_directory, "scores"))]
        destination_score_image = os.path.join(self.destination_directory, "scores")
        print("Copying {0} score images...".format(len(path_to_all_files)))
        for score_image in path_to_all_files:
            shutil.copy(score_image, destination_score_image)

        path_to_all_files = [os.path.join(self.source_directory, "other", file) for file in os.listdir(os.path.join(self.source_directory, "other"))]
        destination_other_image = os.path.join(self.destination_directory, "other")
        print("Copying {0} other images...".format(len(path_to_all_files)))
        for other_image in path_to_all_files:
            shutil.copy(other_image, destination_other_image)

# datasest = AdditionalDataset('C:\\Users\\Alex\\Repositories\\MusicScoreClassifier\\ModelGenerator\\data',
#                              'C:\\Users\\Alex\\Dropbox\\Doktorat\\MusicScoresDataset')
# datasest.download_and_extract_dataset()