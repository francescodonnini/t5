from keras import layers

from models.common import model_head, conv_block


class InceptionV1(layers.Layer):
    def __init__(self, d1, d2, d3, d4, **kwargs):
        super(InceptionV1, self).__init__(**kwargs)
        self.col1 = layers.Conv2D(d1, 1, activation='relu')
        self.col2 = layers.Pipeline([
            layers.Conv2D(d2[0], 1, activation='relu'),
            layers.Conv2D(d2[1], 3, padding='same', activation='relu')
        ])
        self.col3 = layers.Pipeline([
            layers.Conv2D(d3[0], 1, activation='relu'),
            layers.Conv2D(d3[1], 5, padding='same', activation='relu')
        ])
        self.col4 = layers.Pipeline([
            layers.MaxPool2D(3, 1, padding='same'),
            layers.Conv2D(d4, 1, activation='relu')
        ])

    def call(self, inputs):
        return layers.Concatenate()([self.col1(inputs), self.col2(inputs), self.col3(inputs), self.col4(inputs)])


def create_model(
        width: int,
        height: int,
        data_augmentation: layers.Layer=None):
    m = model_head((width, height, 1), data_augmentation)
    m.add(conv_block(64, 7, 2, padding='same', activation="relu"))
    m.add(layers.MaxPool2D(3, 2, padding='same'))
    m.add(conv_block(64, 1, activation="relu"))
    m.add(conv_block(192, 3, padding='same', activation="relu"))
    m.add(layers.MaxPool2D(3, 2, padding='same'))
    m.add(InceptionV1(64, (96, 128), (16, 32), 32))
    m.add(InceptionV1(128, (128, 192), (32, 96), 64))
    m.add(layers.MaxPool2D(3, 2, padding='same'))
    m.add(InceptionV1(192, (96, 208), (16, 48), 64))
    m.add(InceptionV1(160, (112, 224), (24, 64), 64))
    m.add(InceptionV1(128, (128, 256), (24, 64), 64))
    m.add(InceptionV1(112, (144, 288), (32, 64), 64))
    m.add(InceptionV1(256, (160, 320), (32, 128), 128))
    m.add(layers.MaxPool2D(3, 2, padding='same'))
    m.add(InceptionV1(256, (160, 320), (32, 128), 128))
    m.add(InceptionV1(384, (192, 384), (48, 128), 128))
    m.add(layers.GlobalAvgPool2D())
    m.add(layers.Flatten())
    m.add(layers.Dense(1024, activation='linear'))
    m.add(layers.Dropout(0.4))
    m.add(layers.Dense(2, activation="softmax"))
    return m
