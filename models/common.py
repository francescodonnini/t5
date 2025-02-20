from typing import Tuple, Optional, List

import numpy as np
from keras import layers, models


class SequentialWithThreshold(models.Sequential):
    def __init__(self, threshold: float):
        super(SequentialWithThreshold, self).__init__()
        if not isinstance(threshold, float):
            raise TypeError('Threshold must be a float')
        elif threshold < 0.0 or threshold > 1.0:
            raise ValueError('Threshold must be between 0 and 1')
        self.threshold = threshold

    def predict(self, x, batch_size=None, verbose="auto", steps=None, callbacks=None):
        y = super().predict(x, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks)
        return np.astype(y > self.threshold, int)

    def get_threshold(self) -> float:
        return self.threshold

    def set_threshold(self, threshold: float):
        self.threshold = threshold

def model_head(shape: Tuple[int, int, int], data_augmentation: layers.Layer=None) -> models.Sequential:
    m = SequentialWithThreshold(0.5)
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
        activation: Optional[str]=None) -> layers.Layer:
    l: List[layers.Layer] = []
    if type(padding) is int:
        l.append(layers.ZeroPadding2D((padding, padding)))
        padding = 'valid'
    l.append(layers.Conv2D(filters, kernel_size, strides=strides, padding=padding))
    l.append(layers.BatchNormalization())
    if activation is not None:
        l.append(layers.Activation(activation))
    return layers.Pipeline(l)