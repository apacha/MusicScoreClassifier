import os
import unittest

from TrainModel import delete_dataset_directory, download_datasets


class OnlineDatasetTest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def test_download_dataset(self):
        delete_dataset_directory("data")
        download_datasets("data")
        self.assertGreaterEqual(2, len(os.listdir("data")))