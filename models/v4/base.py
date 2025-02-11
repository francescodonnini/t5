from typing import Tuple

from keras import layers


class ConvBlock(layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: int | Tuple[int, int],
                 strides: int,
                 padding: int | str,
                 activation: str,
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)
        self.norm = layers.BatchNormalization()
        self.activation = layers.Activation(activation)

    def call(self, inputs) -> layers.Layer:
        x = self.conv(inputs)
        x = self.norm(x)
        return self.activation(x)

def conv(
        filters: int,
        kernel_size: int|Tuple[int, int],
        strides: int = 1,
        padding: int | str = 'valid',
        activation: str='relu',
        **kwargs):
    return ConvBlock(filters, kernel_size, strides, padding, activation, **kwargs)


class Stem(layers.Layer):
    def __init__(self, **kwargs):
        super(Stem, self).__init__(**kwargs)
        self.c1 = conv(32, 3, padding='valid', strides=2)
        self.c2 = conv(32, 3, padding='valid')
        self.c3 = conv(64, 3, padding='same')
        self.c4 = conv(96, 3, 2, padding='valid')
        self.c4p = layers.MaxPool2D(3, 2, padding='valid')
        self.c6_1 = conv(64, 1, padding='same')
        self.c6_2 = conv(96, 3, padding='valid')
        self.c7_1 = conv(64, 1, padding='same')
        self.c7_2 = conv(64, (7, 1), padding='same')
        self.c7_3 = conv(64, (1, 7), padding='same')
        self.c7_4 = conv(96, 3, padding='valid')
        self.c9p = conv(192, 3, padding='valid')
        self.c9 = layers.MaxPool2D(strides=2, padding='valid')
        self.concat1 = layers.Concatenate()
        self.concat2 = layers.Concatenate()
        self.concat3 = layers.Concatenate()

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.c2(x)
        x = self.c3(x)
        xp = self.c4p(x)
        x = self.c4(x)
        x = self.concat1([xp, x])
        xp = self.c6_1(x)
        xp = self.c6_2(xp)
        x = self.c7_1(x)
        x = self.c7_2(x)
        x = self.c7_3(x)
        x = self.c7_4(x)
        x = self.concat2([xp, x])
        xp = self.c9p(x)
        x = self.c9(x)
        return self.concat3([xp, x])


class InceptionA(layers.Layer):
    def __init__(self, **kwargs):
        super(InceptionA, self).__init__(**kwargs)
        self.c1_1 = layers.AvgPool2D(3, 1)
        self.c2_1 = conv(96, 1, padding='same')
        self.c1_2 = conv(96, 1, padding='same')
        self.c1_3 = conv(64, 1, padding='same')
        self.c2_3 = conv(96, 3, padding='same')
        self.c1_4 = conv(64, 1, padding='same')
        self.c2_4 = conv(96, 1, padding='same')
        self.c3_4 = conv(96, 3, padding='same')
        self.c4_4 = conv(96, 3, padding='same')
        self.concat = layers.Concatenate()

    def call(self, inputs):
        x1 = self.c1_1(inputs)
        x1 = self.c2_1(x1)
        x2 = self.c1_2(inputs)
        x3 = self.c1_3(inputs)
        x3 = self.c2_3(x3)
        x4 = self.c1_4(inputs)
        x4 = self.c2_4(x4)
        x4 = self.c3_4(x4)
        x4 = self.c4_4(x4)
        return self.concat([x1, x2, x3, x4])


class InceptionB(layers.Layer):
    def __init__(self, **kwargs):
        super(InceptionB, self).__init__(**kwargs)
        self.c1_1 = layers.AvgPool2D(3, 1)
        self.c2_1 = conv(128, 1, padding='same')
        self.c1_2 = conv(192, 1, padding='same')
        self.c2_2 = conv(224, (1, 7), padding='same')
        self.c3_2 = conv(256, (1, 7), padding='same')
        self.c1_3 = conv(192, 1, padding='same')
        self.c2_3 = conv(192, (1, 7), padding='same')
        self.c3_3 = conv(224, (7, 1), padding='same')
        self.c4_3 = conv(244, (1, 7), padding='same')
        self.c5_3 = conv(256, (7, 1), padding='same')
        self.concat = layers.Concatenate()

    def call(self, inputs):
        x1 = self.c1_1(inputs)
        x1 = self.c2_1(x1)
        x2 = self.c1_2(inputs)
        x2 = self.c2_2(x2)
        x2 = self.c3_2(x2)
        x3 = self.c1_3(inputs)
        x3 = self.c2_3(x3)
        x3 = self.c3_3(x3)
        x3 = self.c4_3(x3)
        x3 = self.c5_3(x3)
        return self.concat([x1, x2, x3])


class InceptionC(layers.Layer):
    def __init__(self, **kwargs):
        super(InceptionC, self).__init__(**kwargs)
        self.c1_1 = layers.AvgPool2D(3, 1)
        self.c2_1 = conv(256, 1, padding='same')
        self.c1_2 = conv(256, 1, padding='same')
        self.c1_3 = conv(384, 1, padding='same')
        self.c2_3 = conv(256, (1, 3), padding='same')
        self.c2_3p = conv(256, (3, 1), padding='same')
        self.c1_4 = conv(384, 1, padding='same')
        self.c2_4 = conv(448, (1, 3), padding='same')
        self.c3_4 = conv(512, (3, 1), padding='same')
        self.c4_4 = conv(256, (3, 1), padding='same')
        self.c4_4p = conv(256, (1, 3), padding='same')
        self.concat = layers.Concatenate()

    def call(self, inputs):
        x1 = self.c1_1(inputs)
        x1 = self.c2_1(x1)
        x2 = self.c1_2(inputs)
        x3 = self.c1_3(inputs)
        x3_1 = self.c2_3(x3)
        x3_2 = self.c2_3p(x3)
        x4 = self.c1_4(inputs)
        x4 = self.c2_4(x4)
        x4 = self.c3_4(x4)
        x4_1 = self.c4_4(x4)
        x4_2 = self.c4_4p(x4)
        return self.concat([x1, x2, x3_1, x3_2, x4_1, x4_2])


class ReductionA(layers.Layer):
    def __init__(self, k: int, l: int, m: int, n: int, **kwargs):
        super(ReductionA, self).__init__(**kwargs)
        self.c1_1 = layers.MaxPool2D(3, 2, padding='valid')
        self.c1_2 = conv(n, 3, 2, padding='valid')
        self.c1_3 = conv(k, 1, padding='same')
        self.c2_3 = conv(l, 3, padding='same')
        self.c3_3 = conv(m, 3, 2, padding='valid')
        self.concat = layers.Concatenate()

    def call(self, inputs):
        x1 = self.c1_1(inputs)
        x2 = self.c1_2(inputs)
        x3 = self.c1_3(inputs)
        x3 = self.c2_3(x3)
        x3 = self.c3_3(x3)
        return self.concat([x1, x2, x3])


class ReductionB(layers.Layer):
    def __init__(self, **kwargs):
        super(ReductionB, self).__init__(**kwargs)
        self.c1_1 = layers.MaxPool2D(3, 2, padding='valid')
        self.c1_2 = conv(192, 1, padding='same')
        self.c2_2 = conv(192, 3, 2, padding='valid')
        self.c1_3 = conv(256, 1, padding='same')
        self.c2_3 = conv(256, (1, 7), 2, padding='same')
        self.c3_3 = conv(320, (7, 1), padding='same')
        self.c4_3 = conv(320, 3, 2, padding='valid')
        self.concat = layers.Concatenate()

    def call(self, inputs):
        x1 = self.c1_1(inputs)
        x2 = self.c1_2(x1)
        x2 = self.c2_2(x2)
        x3 = self.c1_3(inputs)
        x3 = self.c2_3(x3)
        x3 = self.c3_3(x3)
        return self.concat([x1, x2, x3])
