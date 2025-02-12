from typing import Tuple, List, Iterable

from keras import activations, layers, models


class Residual(layers.Layer):
    def __init__(self, filters: int, strides: int=1, use_conv1x1=False):
        super(Residual, self).__init__()
        self.block = models.Sequential([
            layers.Conv2D(filters, 3, strides, 'same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters, 3, 1, 'same'),
            layers.BatchNormalization(),
        ])
        self.conv1x1 = None
        if use_conv1x1:
            self.conv1x1 = layers.Conv2D(filters, 1, strides)
        self.b1 = layers.BatchNormalization()
        self.b2 = layers.BatchNormalization()

    def call(self, inputs):
        z = self.block(inputs)
        if self.conv1x1:
            inputs = self.conv1x1(inputs)
        z = layers.Add()([z, inputs])
        return activations.relu(z)


def create_model(width: int, height: int, arch: Iterable[Tuple[int, int]], data_augmentation: layers.Layer=None):
    m = models.Sequential()
    m.add(layers.Input(shape=(height, width, 1)))
    if data_augmentation is not None:
        m.add(data_augmentation)
    m.add(layers.Rescaling(1. / 255))
    m.add(layers.Conv2D(64, 7, 2, 'same'))
    m.add(layers.BatchNormalization())
    m.add(layers.ReLU())
    m.add(layers.MaxPool2D(3, 2, 'same'))
    for i, b in enumerate(arch):
        m.add(block(*b, first_block=i==0))
    m.add(layers.GlobalAvgPool2D())
    m.add(layers.Dense(2, activation='softmax'))
    return m


def block(num_residuals: int, num_channels: int, first_block: bool=False):
    m = models.Sequential()
    if not first_block:
        m.add(Residual(num_channels, use_conv1x1=True, strides=2))
    for _ in range(num_residuals - 1):
        m.add(Residual(num_channels))
    return m