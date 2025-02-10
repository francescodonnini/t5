from typing import List

from keras import layers, models


class InceptionV1(layers.Layer):
    def __init__(self, d1, d2, d3, d4, **kwargs):
        super(InceptionV1, self).__init__(**kwargs)
        self.b1_1 = layers.Conv2D(d1, 1, activation='relu')
        self.b2_1 = layers.Conv2D(d2[0], 1, activation='relu')
        self.b2_2 = layers.Conv2D(d2[1], 3, padding='same', activation='relu')
        self.b3_1 = layers.Conv2D(d3[0], 1, activation='relu')
        self.b3_2 = layers.Conv2D(d3[1], 5, padding='same', activation='relu')
        self.b4_1 = layers.MaxPool2D(3, 1, padding='same')
        self.b4_2 = layers.Conv2D(d4, 1, activation='relu')

    def call(self, inputs):
        b1 = self.b1_1(inputs)
        b2 = self.b2_2(self.b2_1(inputs))
        b3 = self.b3_2(self.b3_1(inputs))
        b4 = self.b4_2(self.b4_1(inputs))
        return layers.Concatenate()([b1, b2, b3, b4])


def create_model(
        width: int,
        height: int,
        data_augmentation: layers.Layer=None):
    l: List[layers.Layer] = []
    if data_augmentation is not None:
        l.append(data_augmentation)
    l.extend([
        layers.Rescaling(1./255, input_shape=(width, height, 1)),
        layers.Conv2D(64, 7, strides=2, padding='same', activation="relu"),
        layers.MaxPool2D(3, strides=2, padding='same'),
        layers.Conv2D(64, 1, activation="relu"),
        layers.Conv2D(192, 3, padding='same', activation="relu"),
        layers.MaxPool2D(3, 2, padding='same'),
        InceptionV1(64, (96, 128), (16, 32), 32),
        InceptionV1(128, (128, 192), (32, 96), 64),
        layers.MaxPool2D(3, 2, padding='same'),
        InceptionV1(192, (96, 208), (16, 48), 64),
        InceptionV1(160, (112, 224), (24, 64), 64),
        InceptionV1(128, (128, 256), (24, 64), 64),
        InceptionV1(112, (144, 288), (32, 64), 64),
        InceptionV1(256, (160, 320), (32, 128), 128),
        layers.MaxPool2D(3, 2, padding='same'),
        InceptionV1(256, (160, 320), (32, 128), 128),
        InceptionV1(384, (192, 384), (48, 128), 128),
        layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        layers.Dense(2, activation="softmax")])
    model = models.Sequential(l)
    return model
