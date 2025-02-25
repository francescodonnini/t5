from typing import Tuple, Iterable

from keras import activations, layers

from core.common import model_head, conv_block


class Residual(layers.Layer):
    def __init__(self, filters: int, strides: int=1, use_conv1x1=False):
        super(Residual, self).__init__()
        self.block = layers.Pipeline([
            layers.Conv2D(filters, 3, strides, 'same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters, 3, 1, 'same'),
            layers.BatchNormalization(),
        ])
        if use_conv1x1:
            self.skip = layers.Conv2D(filters, 1, strides)
        else:
            self.skip = layers.Identity()

    def call(self, inputs):
        z = layers.add([self.skip(inputs), self.block(inputs)])
        return activations.relu(z)


def create_model(
        width: int,
        height: int,
        arch: Iterable[Tuple[int, int]],
        data_augmentation: layers.Layer=None):
    m = model_head((width, height, 1), data_augmentation)
    m.add(conv_block(64, 7, 2, activation='relu'))
    m.add(layers.MaxPool2D(3, 2, 'same'))
    for i, b in enumerate(arch):
        m.add(residual_block(*b, first_block=i==0))
    m.add(layers.GlobalAvgPool2D())
    m.add(layers.Dense(1, activation='sigmoid'))
    return m


def residual_block(num_residuals: int, num_channels: int, first_block: bool=False) -> layers.Layer:
    l = []
    if not first_block:
        l.append(Residual(num_channels, use_conv1x1=True, strides=2))
    for _ in range(num_residuals - 1):
        l.append(Residual(num_channels))
    return layers.Pipeline(l)