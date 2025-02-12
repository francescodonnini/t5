from typing import Tuple

from keras import layers, Model

from models import alex_net, googlenet, resnet, v4


def create_model(
        name: str,
        resize: Tuple[int, int]=(224, 224),
        data_augmentation: layers.Layer=None) -> Model:
    width, height = resize
    if name == 'alex_net':
        return alex_net.create_model(width, height, data_augmentation=data_augmentation)
    elif name == 'googlenet':
        return googlenet.create_model(width, height, data_augmentation=data_augmentation)
    elif name == 'resnet-18':
        return resnet.create_model(width, height,((2, 64), (2, 128), (2, 512)), data_augmentation=data_augmentation)
    elif name == 'inception-v4':
        return v4.inception(width, height, data_augmentation=data_augmentation)
    elif name == 'resnet-v1':
        return v4.resnet_v1(width, height, data_augmentation=data_augmentation)
    elif name == 'resnet-v2':
        return v4.resnet_v2(width, height, data_augmentation=data_augmentation)
    else:
        raise ValueError(f'unknown model name: {name}')
