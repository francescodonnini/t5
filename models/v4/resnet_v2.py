from models.v4 import ReductionA
from models.v4.resnet_vx import InceptionAVx, InceptionWVx, ReductionBVx


def inception_a(**kwargs):
    return InceptionAVx(48, 64, 384, **kwargs)

def inception_b(**kwargs):
    return InceptionWVx(
        [192, 128, 160, 192, 1152],
        [1, 1, (1, 7), (7, 1), 1],
        **kwargs)

def inception_c(**kwargs):
    return InceptionWVx(
        [192, 192, 224, 256, 2048],
        [1, 1, (1, 3), (3, 1), 1],
        **kwargs)


def reduction_b(**kwargs):
    return ReductionBVx(256, 288, 320, **kwargs)


def reduction_a(**kwargs):
    return ReductionA(256, 256, 384, 384, **kwargs)