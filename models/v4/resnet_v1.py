from keras import layers

from models.v4.base import conv, ReductionA
from models.v4.resnet_vx import InceptionWVx, ReductionBVx, InceptionAVx

class StemV1(layers.Layer):
    def __init__(self, **kwargs):
        super(StemV1, self).__init__(**kwargs)
        self.c1 = conv(32, 3, strides=2, padding='valid')
        self.c2 = conv(32, 3, padding='valid')
        self.c3 = conv(64, 3)
        self.c4 = layers.MaxPool2D(3, 2, padding='valid')
        self.c5 = conv(80, 1)
        self.c6 = conv(192, 3, padding='valid')
        self.c7 = conv(256, 3, 2, padding='valid')

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        return self.c7(x)


def inception_a(**kwargs):
    return InceptionAVx(32, 32, 256, **kwargs)

def inception_b(**kwargs):
    return InceptionWVx(
        [128, 128, 128, 128, 896],
        [1, 1, (1, 7), (7, 1), 1],
        **kwargs)

def inception_c(**kwargs):
    return InceptionWVx(
        [192, 192, 192, 192, 1792],
        [1, 1, (1, 3), (3, 1), 1],
        **kwargs)

def reduction_b(**kwargs):
    return ReductionBVx(256, 256, 256, **kwargs)


def reduction_a(**kwargs):
    return ReductionA(192, 192, 256, 384, **kwargs)