from keras import layers

from models.common import model_head, conv_block


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
    m.add(layers.Dense(4096, activation='relu'))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(4096, activation='relu'))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(2, activation='softmax'))
    return m
