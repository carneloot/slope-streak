from util import get_images, TEST_DIRECTORY
from os import listdir

import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import models

import config_gpu

THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def calculate_metrics(predicted, threshold, print_all):
    predicted = predicted > threshold
    predicted = predicted.astype(int)

    confusion_matrix = tf.math.confusion_matrix(labels.flatten(), predicted.flatten()).numpy()

    true_negative = confusion_matrix[0, 0]
    true_positive = confusion_matrix[1, 1]
    false_positive = confusion_matrix[0, 1]
    false_negative = confusion_matrix[1, 0]

    acc = (true_negative + true_positive) / (true_negative + true_positive + false_negative + false_positive)

    if print_all:
        print('true_positive:', true_positive)
        print('true_negative:', true_negative)
        print('false_positive:', false_positive)
        print('false_negative:', false_negative)

        print(confusion_matrix)

    print(acc, end='')

def test_model(model_path, images, labels, thresholds, print_all=False):
    model = models.load_model(
        model_path,
        custom_objects={
            'Addons>F1Score': tfa.metrics.F1Score
        }
    )

    model.evaluate(images, labels, verbose=0)

    predicted = model.predict(images)

    for thresh in thresholds:
        if print_all:
            print(f'\n\nThreshold {thresh}:')
        elif thresh != thresholds[0]:
            print(',', end='')

        calculate_metrics(predicted, thresh, print_all)

    tf.keras.backend.clear_session()

MODEL_PATH = 'models/50_32_conv_1609964632.h5'

images, labels, _ = get_images(TEST_DIRECTORY)

all_models = listdir('models/')

print('filename', end=',')
print(','.join(str(x) for x in THRESHOLDS))

for model_filename in all_models:
    print(model_filename, end=',')

    test_model(f'models/{model_filename}', images, labels, THRESHOLDS)

    print()
