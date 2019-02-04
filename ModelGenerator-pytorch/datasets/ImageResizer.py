from glob import glob

from PIL import Image
from tqdm import tqdm


def resize_images(path: str, width=224, height=224):
    for extension in ["png", "jpg"]:
        all_image_paths = glob(f"{path}/**/*.{extension}")
        for image_path in tqdm(all_image_paths):
            image = Image.open(image_path) #type:Image
            image = image.resize(size=(width, height), resample=Image.LANCZOS)
            image.save(image_path)


if __name__ == "__main__":
    resize_images("data/training")
    resize_images("data/validation")
    resize_images("data/test")