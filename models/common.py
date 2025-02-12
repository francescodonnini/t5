from keras import layers, models
from typing import Tuple


def model_head(shape: Tuple[int, int, int], data_augmentation: layers.Layer=None) -> models.Sequential:
    m = models.Sequential()
    m.add(layers.Input(shape=shape))
    if data_augmentation is not None:
        m.add(data_augmentation)
    m.add(layers.Rescaling(1. / 255))
    return m
