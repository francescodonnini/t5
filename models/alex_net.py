from typing import Tuple, Optional

from keras import layers, models, Model

from models.common import model_head


def conv_block(filters: int, kernel_size: int|Tuple[int, int], strides: int|Tuple[int, int]=1, padding: int|str='same', activation: Optional[str]=None) -> Model:
    m = models.Sequential()
    m.add(layers.Conv2D(filters, kernel_size, strides=strides, padding=padding))
    m.add(layers.BatchNormalization())
    if activation is not None:
        m.add(layers.Activation(activation))
    return m

def create_model(
        width: int,
        height: int,
        data_augmentation: layers.Layer=None):
    m = model_head((width, height, 1), data_augmentation=data_augmentation)
    m.add(conv_block(96, 11, 4, padding='valid', activation='relu'))
    m.add(layers.MaxPool2D(3, 2))
    m.add(conv_block(256, 5, activation='relu'))
    m.add(layers.MaxPool2D(3, 2))
    m.add(conv_block(384, 3, activation='relu'))
    m.add(conv_block(384, 3, activation='relu'))
    m.add(conv_block(256, 3, activation='relu'))
    m.add(layers.MaxPool2D(3, 2))
    m.add(layers.Flatten())
    m.add(layers.Dense(9216, activation='relu'))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(4096, activation='relu'))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(4096, activation='relu'))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(2, activation='softmax'))
    return m
