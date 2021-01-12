from util import get_images, TRAIN_DIRECTORY, VALIDATION_DIRECTORY, CROP_SIZE

from time import time
from math import floor
from sys import argv

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import datasets, layers, models

import config_gpu

import matplotlib.pyplot as plt

train_images, train_labels, _ = get_images(TRAIN_DIRECTORY)
val_images, val_labels, _ = get_images(VALIDATION_DIRECTORY)

total_images = train_images.shape[0]

epochs = 10
if len(argv) > 1:
    epochs = int(argv[1])

model = models.Sequential()

# Modelando a rede
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(CROP_SIZE, CROP_SIZE, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),
        # tfa.metrics.F1Score(num_classes=1, average='macro'),
    ]
)

model.summary()

model_filename = f'{epochs}__32_conv_32_dense_{floor(time())}'

history = model.fit(
    train_images,
    train_labels,
    epochs=epochs,
    validation_data=(val_images, val_labels),
    verbose=2,
    shuffle=True,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'./best/{model_filename}.h5',
            monitor='val_binary_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        # tf.keras.callbacks.EarlyStopping(
        #     monitor='val_binary_accuracy',
        #     min_delta=1e-4,
        #     patience=50,
        #     verbose=1,
        # )
    ]
)

# plt.plot(history.history['f1_score'], label='F1 do Treino')
# plt.plot(history.history['val_f1_score'], label = 'F1 da Validacao')
plt.plot(history.history['binary_accuracy'], label='Acuracia do Treino')
plt.plot(history.history['val_binary_accuracy'], label = 'Acuracia da Validacao')
plt.xlabel('Epoca')
plt.ylabel('Metricas')
plt.ylim([0.5, 1])
plt.legend(loc='upper left')

plt.savefig(f'./model_graphs/{model_filename}.jpg')

model.save(f'./models/{model_filename}.h5')