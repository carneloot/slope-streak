from util import TRAIN_VALIDATION_TEST_SPLIT, DATASET_DIRECTORY, TRAIN_DIRECTORY, VALIDATION_DIRECTORY, TEST_DIRECTORY
from os import listdir, path, makedirs
from shutil import copy, rmtree
from math import floor, inf

import numpy as np
import tensorflow as tf

def copy_to_dir(images, dest_dir, delete_first=True):
    if delete_first:
        if path.exists(dest_dir):
            rmtree(dest_dir)
        makedirs(dest_dir)

    for image_path_bytes in images.numpy():
        image_path = image_path_bytes.decode('utf=8')

        basename = path.basename(image_path)
        new_path = f'{dest_dir}/{basename}'

        copy(image_path, new_path)

def split_list(path_list, maximum_num = None):
    total_images = len(path_list)
    if maximum_num is not None:
        maximum_num = floor(maximum_num)
        total_images = min(total_images, maximum_num)

    perc_train, perc_val, perc_test = TRAIN_VALIDATION_TEST_SPLIT

    num_train = floor(total_images * perc_train)
    num_val = floor(total_images * perc_val)
    num_test = floor(total_images * perc_test)

    # Caso sobre algum, coloca no treino
    num_train += (total_images - (num_train + num_val + num_test))

    path_list = tf.random.shuffle(path_list)

    path_list = tf.slice(path_list, [0], [total_images])

    return tf.split(path_list, [num_train, num_val, num_test])

images = listdir(DATASET_DIRECTORY)
images = list(map(lambda filename: f'{DATASET_DIRECTORY}/{filename}', images))

yes_images = list(filter(lambda img_path: img_path.endswith('1.jpg'), images))
no_images = list(filter(lambda img_path: img_path.endswith('0.jpg'), images))

print('Total yes:', len(yes_images))
print('Total no:', len(no_images))

yes_train, yes_val, yes_test = split_list(yes_images)
no_train, no_val, no_test = split_list(no_images, len(yes_images) * 1.5)

copy_to_dir(yes_train, TRAIN_DIRECTORY)
copy_to_dir(no_train, TRAIN_DIRECTORY, False)

copy_to_dir(yes_val, VALIDATION_DIRECTORY)
copy_to_dir(no_val, VALIDATION_DIRECTORY, False)

copy_to_dir(yes_test, TEST_DIRECTORY)
copy_to_dir(no_test, TEST_DIRECTORY, False)
