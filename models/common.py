from keras import layers, models, Model

from typing import Tuple, Optional


def model_head(shape: Tuple[int, int, int], data_augmentation: layers.Layer=None) -> models.Sequential:
    m = models.Sequential()
    m.add(layers.Input(shape=shape))
    if data_augmentation is not None:
        m.add(data_augmentation)
    m.add(layers.Rescaling(1. / 255))
    m.add(layers.Normalization())
    return m

def conv_block(
        filters: int,
        kernel_size: int|Tuple[int, int],
        strides: int|Tuple[int, int]=1,
        padding: int|str='same',
        activation: Optional[str]=None) -> Model:
    m = models.Sequential()
    if type(padding) is int:
        m.add(layers.ZeroPadding2D((padding, padding)))
        padding = 'valid'
    m.add(layers.Conv2D(filters, kernel_size, strides=strides, padding=padding))
    m.add(layers.BatchNormalization())
    if activation is not None:
        m.add(layers.Activation(activation))
    return m