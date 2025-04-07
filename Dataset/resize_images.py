import os
import cv2
import numpy as np
from PIL import Image


def resize(img_dir):
    print(img_dir)
    for path, subdirs, files in os.walk(img_dir):
        images = os.listdir(path)  # List of all image names in this directory
        total_images = len(images)
        for i, image_name in enumerate(images):
            if image_name.endswith(".jpg"):
                image = cv2.imread(path + image_name, 1)  # Read each image as BGR
                crop_size = (640 - 512) / 2
                image = Image.fromarray(image)
                image = image.crop((crop_size, crop_size, 640-crop_size, 640-crop_size))  # Crop from top left corner
                image = np.array(image)

                cv2.imwrite(img_dir + image_name, image)
                print(f"Resize {img_dir}: {i+1}/{total_images}")


def resize_all():
    resize("Dataset/train/")
    resize("Dataset/train_masks/")
    resize("Dataset/valid/")
    resize("Dataset/valid_masks/")
    resize("Dataset/test/")
    resize("Dataset/test_masks/")

#resize_all()