import os

import numpy
from PIL import Image
from scipy import ndimage
from skimage.transform import resize

image_directory = "C:\\Users\\Alex\\Repositories\\MusicScoreClassifier\\ModelGenerator\\data\\test"
target_directory = "C:\\Users\\Alex\\Repositories\\MusicScoreClassifier\\ModelGenerator\\data\\test_set_resized"

os.makedirs(target_directory, exist_ok=True)
for image_class in ["other", "scores"]:
    index = 0
    image_class_directory = os.path.join(image_directory, image_class)
    for file in os.listdir(image_class_directory):
        # print(file)
        image = ndimage.imread(os.path.join(image_class_directory, file))
        if image.ndim == 3:
            output_shape = (128, 128, 3)
        else:
            output_shape = (128, 128)

        resized = resize(image, output_shape=output_shape, preserve_range=True)
        destination_file_name = os.path.join(target_directory, image_class + "_" + str(index) + ".png")
        resized_image = Image.fromarray(resized.astype(numpy.uint8))
        resized_image.save(destination_file_name)
        index += 1
