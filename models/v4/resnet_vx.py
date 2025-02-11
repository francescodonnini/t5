from typing import List, Tuple

from keras import activations, layers

from models.v4.base import conv


class InceptionAVx(layers.Layer):
    def __init__(self, f23: int, f33: int, f: int,  **kwargs):
        super(InceptionAVx, self).__init__(**kwargs)
        self.c1_1 = conv(32, 1, padding='same')
        self.c1_2 = conv(32, 1, padding='same')
        self.c2_2 = conv(32, 3, padding='same')
        self.c1_3 = conv(32, 1, padding='same')
        self.c2_3 = conv(f23, 3, padding='same')
        self.c3_3 = conv(f33, 3, padding='same')
        self.concat = layers.Concatenate()
        self.conv = conv(f, 1, padding='same', activation='linear')
        self.add = layers.Add()

    def call(self, inputs):
        x1 = self.c1_1(inputs)
        x2 = self.c1_2(inputs)
        x2 = self.c2_2(x2)
        x3 = self.c1_3(inputs)
        x3 = self.c2_3(x3)
        x3 = self.c3_3(x3)
        x = self.concat([x1, x2, x3])
        x = self.conv(x)
        x = self.add([inputs, x])
        return activations.relu(x)


class InceptionWVx(layers.Layer):
    def __init__(self, filters: List[int], kernel_size: List[int|Tuple[int, int]], **kwargs):
        super(InceptionWVx, self).__init__(**kwargs)
        self.c1_1 = conv(filters[0], kernel_size[0], padding='same')
        self.c1_2 = conv(filters[1], kernel_size[1], padding='same')
        self.c2_2 = conv(filters[2], kernel_size[2], padding='same')
        self.c3_2 = conv(filters[3], kernel_size[3], padding='same')
        self.concat = layers.Concatenate()
        self.conv = conv(filters[4], kernel_size[4], padding='same', activation='linear')
        self.add = layers.Add()

    def call(self, inputs):
        x1 = self.c1_1(inputs)
        x2 = self.c1_2(inputs)
        x2 = self.c2_2(x2)
        x2 = self.c3_2(x2)
        x = self.concat([x1, x2])
        x = self.conv(x)
        x = self.add([inputs, x])
        return activations.relu(x)


class ReductionBVx(layers.Layer):
    def __init__(self, f23: int, f24: int, f34: int, **kwargs):
        super(ReductionBVx, self).__init__(**kwargs)
        self.c1_1 = layers.MaxPool2D(3, 2, padding='valid')
        self.c1_2 = conv(256, 1, padding='same')
        self.c2_2 = conv(384, 3, 2, padding='valid')
        self.c1_3 = conv(256, 1, padding='same')
        self.c2_3 = conv(f23, 3, 2, padding='valid')
        self.c1_4 = conv(256, 1, padding='same')
        self.c2_4 = conv(f24, 3, padding='same')
        self.c3_4 = conv(f34, 3, 2, padding='valid')
        self.concat = layers.Concatenate()

    def call(self, inputs):
        x1 = self.c1_1(inputs)
        x2 = self.c1_2(inputs)
        x2 = self.c2_2(x2)
        x3 = self.c1_3(inputs)
        x3 = self.c2_3(x3)
        x4 = self.c1_4(inputs)
        x4 = self.c2_4(x4)
        x4 = self.c3_4(x4)
        return self.concat([x1, x2, x3, x4])
