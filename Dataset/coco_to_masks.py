import numpy as np
import json
import cv2
import os

# --- Mask Creation Functions ---

def create_masks(annotations, image_dir, mask_dir, label=""):
    print(f'Creating {label} masks...')
    os.makedirs(mask_dir, exist_ok=True)
    total_images = len(annotations['images'])
    done = 0

    for img, ann in zip(annotations['images'], annotations['annotations']):
        path = os.path.join(image_dir, img['file_name'])
        mask_path = os.path.join(mask_dir, img['file_name'])

        # Load image
        image = cv2.imread(path)
        if image is None:
            print(f"Error loading {path}")
            continue

        # Process annotation
        segmentation = ann['segmentation']
        segmentation = np.array(segmentation[0], dtype=np.int32).reshape(-1, 2)

        # Create mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [segmentation], color=(255, 255, 255))

        # Save mask
        cv2.imwrite(mask_path, mask)

        done += 1
        print(f"{label}: {done}/{total_images}")


def make_masks():
    train_path = "Dataset/train/"
    test_path = "Dataset/test/"
    val_path = "Dataset/valid/"

    with open('Dataset/train/_annotations.coco.json', 'r') as f:
        train_annotations = json.load(f)
    with open('Dataset/test/_annotations.coco.json', 'r') as f:
        test_annotations = json.load(f)
    with open('Dataset/valid/_annotations.coco.json', 'r') as f:
        val_annotations = json.load(f)

    create_masks(train_annotations, train_path, 'Dataset/train_masks/', label="Train")
    create_masks(val_annotations, val_path, 'Dataset/valid_masks/', label="Validation")
    create_masks(test_annotations, test_path, 'Dataset/test_masks/', label="Test")

    print('Mask creation complete')

#make_masks()
