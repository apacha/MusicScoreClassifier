import os

from Dataset import Dataset


class PascalVocDataset(Dataset):
    """ This dataset contains the Pascal VOC 2006 challenge database which consists over 
        5000 images of ten categories """

    def __init__(self, directory: str):
        super().__init__(directory)
        self.url = "http://host.robots.ox.ac.uk/pascal/VOC/download/voc2006_trainval.tar"
        self.training_directory = os.path.join(self.directory, "training", "other")
        self.validation_directory = os.path.join(self.directory, "training", "other")

    def is_dataset_cached_on_disk(self) -> bool:
        pass

    def download_dataset(self):
        pass
