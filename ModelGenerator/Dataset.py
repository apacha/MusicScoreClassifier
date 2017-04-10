from abc import ABC


class Dataset(ABC):
    def __init__(self,
                 directory: str):
        self.directory = directory
