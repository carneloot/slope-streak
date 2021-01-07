import cv2 as cv
import numpy as np

import re
from os import listdir

IMAGES_DIRECTORY = 'images'

DATASET_DIRECTORY = 'dataset'

TRAIN_DIRECTORY      = 'train_images'
VALIDATION_DIRECTORY = 'validation_images'
TEST_DIRECTORY       = 'test_images'

TRAIN_VALIDATION_TEST_SPLIT = (0.6, 0.2, 0.2)

CROP_SIZE = 200

def get_images(images_directory):
    images = []
    labels = []
    names = []

    images_on_dataset = listdir(images_directory)

    total_images = len(images_on_dataset)

    for image_on_dataset in images_on_dataset:
        match_result = re.match(f'.+-(\d+)_(\d+)_(\d+)_(\d)\.jpg', image_on_dataset)
        label = match_result.groups()[-1]
        label = int(label)

        img = cv.imread(f'{images_directory}/{image_on_dataset}', 0)

        img = img / 255.0

        img = img.reshape((CROP_SIZE, CROP_SIZE, 1))

        images.append(img)
        labels.append(label)
        names.append(image_on_dataset)

    # print(f'{total_images} images collected')

    images = np.array(images)
    labels = np.array(labels).reshape((total_images, 1))

    return images, labels, names