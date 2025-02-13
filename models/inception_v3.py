from models.common import model_head, conv_block
from keras import layers, models

class InceptionV3A(layers.Layer):
    def __init__(self, **kwargs):
        super(InceptionV3A, self).__init__(**kwargs)
        self.b1x1 = conv_block(64, 1)
        self.b5x5 = layers.Pipeline([
            conv_block(48, 1, activation='relu'),
            conv_block(64, 3, activation='relu'),
            conv_block(64, 3, activation='relu')
        ])
        self.b3x3 = layers.Pipeline([
            conv_block(64, 1, activation='relu'),
            conv_block(96, 3, activation='relu'),
        ])
        self.b_pool = layers.Pipeline([
            layers.AvgPool2D(3, 1, padding='same'),
            conv_block(64, 1)
        ])

    def call(self, inputs):
        return layers.Concatenate()([
            self.b1x1(inputs),
            self.b5x5(inputs),
            self.b3x3(inputs),
            self.b_pool(inputs)
        ])


class InceptionV3B(layers.Layer):
    def __init__(self, n: int=7, **kwargs):
        super(InceptionV3B, self).__init__(**kwargs)
        self.col1 = layers.Pipeline([
            conv_block(128, 1, activation="relu"),
            conv_block(128, (n, 1), activation="relu"),
            conv_block(128, (1, n), activation="relu"),
            conv_block(128, (n, 1), activation="relu"),
            conv_block(192, (1, n), activation="relu")])
        self.col2 = layers.Pipeline([
            conv_block(128, 1, activation="relu"),
            conv_block(128, (n, 1), activation="relu"),
            conv_block(192, (1, n), activation="relu"),
        ])
        self.col3 = layers.Pipeline([
            layers.AvgPool2D(3, 1, padding="same"),
            conv_block(192, 1, activation="relu"),
        ])
        self.col4 = conv_block(192, 1, activation="relu")

    def call(self, inputs):
        return layers.Concatenate()([self.col1(inputs), self.col2(inputs), self.col3(inputs), self.col4(inputs)])


class InceptionV3C(layers.Layer):
    def __init__(self, **kwargs):
        super(InceptionV3C, self).__init__(**kwargs)
        self.c11 = conv_block(448, 1, activation="relu")
        self.c21 = conv_block(384, 3, activation="relu")
        self.c31_1 = conv_block(384, (1, 3), activation="relu")
        self.c31_2 = conv_block(384, (3, 1), activation="relu")
        self.c12 = conv_block(448, 1, activation="relu")
        self.c22_1 = conv_block(384, (1, 3), activation="relu")
        self.c22_2 = conv_block(384, (3, 1), activation="relu")
        self.col3 = layers.Pipeline([
            layers.AvgPool2D(3, 1, padding="same"),
            conv_block(192, 1, activation="relu"),
        ])
        self.col4 = conv_block(320, 1, activation="relu")

    def call(self, inputs):
        x1 = self.c11(inputs)
        x1 = self.c21(x1)
        x1 = layers.Concatenate()([self.c31_1(x1), self.c31_2(x1)])
        x2 = self.c12(inputs)
        x2 = layers.Concatenate()([self.c22_1(x2), self.c22_2(x2)])
        return layers.Concatenate()([x1, x2, self.col3(inputs), self.col4(inputs)])


class ReductionA(layers.Layer):
    def __init__(self, **kwargs):
        super(ReductionA, self).__init__(**kwargs)
        self.col1 = conv_block(384, 3, 2, padding="valid", activation='relu')
        self.col2 = layers.Pipeline([
            conv_block(64, 1, activation='relu'),
            conv_block(96, 3, activation='relu'),
            conv_block(96, 3, 2, padding='valid', activation='relu'),
        ])
        self.col3 = layers.MaxPool2D(3, 2)

    def call(self, inputs):
        return layers.Concatenate()([self.col1(inputs), self.col2(inputs), self.col3(inputs)])


class ReductionB(layers.Layer):
    def __init__(self, **kwargs):
        super(ReductionB, self).__init__(**kwargs)
        self.col1 = layers.Pipeline([
            conv_block(192, 1, activation='relu'),
            conv_block(320, 3, 2, padding='valid', activation='relu'),
        ])
        self.col2 = layers.Pipeline([
            conv_block(192, 1, activation='relu'),
            conv_block(192, (1, 7), activation='relu'),
            conv_block(192, (7, 1), activation='relu'),
            conv_block(192, 3, 2, padding='valid', activation='relu'),
        ])
        self.col3 = layers.MaxPool2D(3, 2)

    def call(self, inputs):
        return layers.Concatenate()([self.col1(inputs), self.col2(inputs), self.col3(inputs)])


def create_model(
        width: int,
        height: int,
        a: int=3,
        b: int=5,
        c: int=2,
        data_augmentation: layers.Layer=None):
    m = model_head((width, height, 1), data_augmentation)
    m.add(conv_block(32, 3, 2, 'valid', 'relu'))
    m.add(conv_block(32, 3, 1, 'valid', 'relu'))
    m.add(conv_block(64, 3, 1, activation='relu'))
    m.add(layers.MaxPool2D(3, 2))
    m.add(conv_block(80, 1, padding='valid', activation='relu'))
    m.add(conv_block(192, 3, padding='valid', activation='relu'))
    m.add(conv_block(288, 3, 2, padding='valid', activation='relu'))
    for _ in range(a):
        m.add(InceptionV3A())
    m.add(ReductionA())
    for _ in range(b):
        m.add(InceptionV3B())
    m.add(ReductionB())
    for _ in range(c):
        m.add(InceptionV3C())
    m.add(layers.GlobalAvgPool2D(5))
    m.add(layers.Flatten())
    m.add(layers.Dense(2, activation='softmax'))
    return m
