from keras import layers, models

from models.v4.base import Stem, InceptionA, ReductionB, InceptionC, ReductionA
from models.v4.resnet_vx import InceptionAVx
import models.v4.resnet_v1 as v1
import models.v4.resnet_v2 as v2

def inception(width: int, height: int, a: int=4, b: int=7, c: int=3, data_augmentation: layers.Layer=None, **kwargs):
    m = models.Sequential()
    if data_augmentation is not None:
        m.add(data_augmentation)
    m.add(layers.Rescaling(1. / 255, input_shape=(width, height, 1)))
    m.add(Stem())
    for _ in range(a):
        m.add(InceptionA(**kwargs))
    m.add(ReductionA(192, 224, 256, 384, **kwargs))
    for _ in range(b):
        m.add(ReductionB(**kwargs))
    m.add(ReductionB(**kwargs))
    for _ in range(c):
        m.add(InceptionC(**kwargs))
    m.add(layers.AveragePooling2D((2, 2)))
    m.add(layers.Dropout(0.2))
    m.add(layers.Flatten())
    m.add(layers.Dense(2, activation='softmax'))
    return m


def resnet_v1(width: int, height: int, a: int=5, b: int=10, c: int=5, data_augmentation: layers.Layer=None, **kwargs):
    m = models.Sequential()
    if data_augmentation is not None:
        m.add(data_augmentation)
    m.add(layers.Rescaling(1. / 255, input_shape=(width, height, 1)))
    m.add(v1.StemV1(**kwargs))
    for _ in range(a):
        m.add(v1.inception_a(**kwargs))
    m.add(ReductionA(192, 192, 256, 384, **kwargs))
    for _ in range(b):
        m.add(v1.inception_b(**kwargs))
    m.add(v1.reduction_b(**kwargs))
    for _ in range(c):
        m.add(v1.inception_c(**kwargs))
    return common_layers(m)


def resnet_v2(width: int, height: int, a: int=5, b: int=10, c: int=5, data_augmentation: layers.Layer=None, **kwargs):
    m = models.Sequential()
    if data_augmentation is not None:
        m.add(data_augmentation)
    m.add(layers.Rescaling(1. / 255, input_shape=(width, height, 1)))
    m.add(Stem(**kwargs))
    for _ in range(a):
        m.add(v2.inception_a(**kwargs))
    m.add(ReductionA(256, 256, 384, 384, **kwargs))
    for _ in range(b):
        m.add(v2.inception_b(**kwargs))
    m.add(v2.reduction_b(**kwargs))
    for _ in range(c):
        m.add(v2.inception_c(**kwargs))
    return common_layers(m)


def common_layers(m: models.Sequential):
    m.add(layers.AveragePooling2D((2, 2)))
    m.add(layers.Dropout(0.2))
    m.add(layers.Flatten())
    m.add(layers.Dense(2, activation='softmax'))
    return m