from MuscimaDataset import MuscimaDataset
from PascalVocDataset import PascalVocDataset


print("Downloading and extracting datasets...")

pascalVocDataset = PascalVocDataset("data")
pascalVocDataset.download_and_extract_dataset()

muscimaDataset = MuscimaDataset("data")
muscimaDataset.download_and_extract_dataset()
