import os
from argparse import ArgumentParser

import numpy
from PIL import Image
from scipy import ndimage
from skimage.transform import resize
from tqdm import tqdm


def copy_and_resize(image_directory: str, target_directory: str, rescaled_width: int = 128, rescaled_height: int = 128):
    os.makedirs(target_directory, exist_ok=True)
    for image_class in ["other", "scores"]:
        index = 0
        image_class_directory = os.path.join(image_directory, image_class)
        class_target_directory = os.path.join(target_directory, image_class)
        os.makedirs(class_target_directory, exist_ok=True)
        for file in tqdm(os.listdir(image_class_directory)):
            # print(file)
            image = ndimage.imread(os.path.join(image_class_directory, file))
            if image.ndim == 3:
                output_shape = (rescaled_width, rescaled_height, 3)
            else:
                output_shape = (rescaled_width, rescaled_height)

            resized = resize(image, output_shape=output_shape, preserve_range=True, mode='constant')
            destination_file_name = os.path.join(class_target_directory, image_class + "_" + str(index) + ".png")
            resized_image = Image.fromarray(resized.astype(numpy.uint8))
            resized_image.save(destination_file_name)
            index += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--image_directory",
        dest="image_directory",
        type=str,
        default="C:\\Users\\Alex\\Repositories\\MusicScoreClassifier\\ModelGenerator\\data\\test",
        help="The directory that contains the entire image dataset (including two subfolders called other and scores for the two classes.",
    )
    parser.add_argument(
        "-t",
        "--target_directory",
        dest="target_directory",
        type=str,
        default="C:\\Users\\Alex\\Repositories\\MusicScoreClassifier\\ModelGenerator\\data\\entire_dataset_128x128",
        help="The destination folder into which all files should be copied to",
    )

    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)

    args = parser.parse_args()

    copy_and_resize(args.image_directory, args.target_directory, int(args.width), int(args.height))
