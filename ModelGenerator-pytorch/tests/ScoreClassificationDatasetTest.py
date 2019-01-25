import unittest
import sys
import pytest
from PIL.Image import Image
from torchvision.transforms import Resize

from datasets.ScoreClassificationDataset import ScoreClassificationDataset


class ScoreClassificationDatasetTest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.dataset = ScoreClassificationDataset("data")

    def test_length(self):
        self.assertEqual(5618, len(self.dataset))

    def test_get_item(self):
        first_item = self.dataset[0]
        self.assertEqual(0, first_item["class"])

    def test_get_last_item(self):
        last_item = self.dataset[len(self.dataset) - 1]
        self.assertEqual(1, last_item["class"])

    def test_get_item_expect_an_image(self):
        first_item = self.dataset[0]
        self.assertIsNotNone(first_item["image"])
        self.assertIsInstance(first_item["image"], Image)

    @pytest.mark.skip(reason="No assertion. Just shows first images")
    def test_show_first_image(self):
        self.dataset.show_sample(0)

    def test_show_first_image(self):
        self.dataset.show_sample(0)

    def test_resize_image(self):
        resizer = Resize((128,128))
        dataset = ScoreClassificationDataset("data", resizer)
        first_item = dataset[0]
        self.assertEqual(128, first_item["image"].width)
