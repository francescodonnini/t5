from typing import List

from keras import activations, layers, models


def create_model(
        width: int,
        height: int,
        data_augmentation: layers.Layer=None):
    l: List[layers.Layer] = []
    if data_augmentation is not None:
        l.append(data_augmentation)
    l.extend([
        layers.Rescaling(1./255, input_shape=(width, height, 1)),
        layers.Conv2D(96, (11, 11), (4, 4),
                      activation=activations.relu),
        layers.BatchNormalization(),
        layers.MaxPool2D((3, 3), (2, 2)),
        layers.Conv2D(256, (5, 5), (2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool2D((3, 3), (2, 2)),
        layers.Conv2D(384, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(384, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool2D((3, 3), (2, 2)),
        layers.Flatten(),
        layers.Dense(4096),
        layers.Dropout(0.5),
        layers.Dense(4096),
        layers.Dropout(0.5),
        layers.Dense(2, activation=activations.sigmoid)])
    model = models.Sequential(l)
    return model
