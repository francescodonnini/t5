from typing import Tuple

from keras import layers, Model

from core import alex_net, googlenet, resnet, inception_v3, v4
from core.v4 import inception


def create_model(
        name: str,
        resize: Tuple[int, int]=(224, 224),
        data_augmentation: layers.Layer=None,
        **kwargs) -> Model:
    width, height = resize
    if name == 'alex-net':
        return alex_net.create_model(width, height, data_augmentation=data_augmentation)
    elif name == 'googlenet':
        return googlenet.create_model(width, height, data_augmentation=data_augmentation)
    elif name == 'resnet-18':
        return resnet.create_model(width, height,((2, 64), (2, 128), (2, 512)), data_augmentation=data_augmentation)
    elif name == 'inception-v3':
        return inception_v3.create_model(width, height, data_augmentation=data_augmentation, **kwargs)
    else:
        raise ValueError(f'unknown model name: {name}')
