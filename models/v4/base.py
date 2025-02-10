from keras import layers


class Stem(layers.Layer):
    def __init__(self, **kwargs):
        super(Stem, self).__init__(**kwargs)
        self.c1 = layers.Conv2D(32, 3, padding='valid', strides=2)
        self.c2 = layers.Conv2D(32, 3, padding='valid')
        self.c3 = layers.Conv2D(64, 3, padding='same')
        self.c4_1 = layers.MaxPool2D(3, 2, padding='valid')
        self.c4_2 = layers.Conv2D(3, 3, 2, padding='valid')
        self.c5 = layers.Concatenate()([self.c4_1, self.c4_2])
        self.c6_1 = layers.Conv2D(64, 1, padding='same')
        self.c6_2 = layers.Conv2D(96, 3, padding='valid')
        self.c7_1 = layers.Conv2D(64, 1, padding='same')
        self.c7_2 = layers.Conv2D(64, (7, 1), padding='same')
        self.c7_3 = layers.Conv2D(64, (1, 7), padding='same')
        self.c7_4 = layers.Conv2D(96, 3, padding='valid')
        self.c8 = layers.Concatenate()([self.c7_1, self.c7_2])
        self.c9_1 = layers.Conv2D(192, 3, padding='valid')
        self.c9_2 = layers.MaxPool2D(strides=2, padding='valid')
        self.c10 = layers.Concatenate()([self.c9_1, self.c9_2])

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4_1(x)
        x = self.c4_2(x)
        x = self.c5(x)
        x = self.c6_1(x)
        x = self.c6_2(x)
        x = self.c7_1(x)
        x = self.c7_2(x)
        x = self.c7_3(x)
        x = self.c7_4(x)
        x = self.c8(x)
        x = self.c9_1(x)
        x = self.c9_2(x)
        return self.c10(x)


class InceptionA(layers.Layer):
    def __init__(self, **kwargs):
        super(InceptionA, self).__init__(**kwargs)
        self.c1_1 = layers.AvgPool2D(3, 1)
        self.c2_1 = layers.Conv2D(96, 1, padding='same')
        self.c1_2 = layers.Conv2D(96, 1, padding='same')
        self.c1_3 = layers.Conv2D(64, 1, padding='same')
        self.c2_3 = layers.Conv2D(96, 3, padding='same')
        self.c1_4 = layers.Conv2D(64, 1, padding='same')
        self.c2_4 = layers.Conv2D(96, 1, padding='same')
        self.c3_4 = layers.Conv2D(96, 3, padding='same')
        self.c4_4 = layers.Conv2D(96, 3, padding='same')
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
        self.c2_1 = layers.Conv2D(128, 1, padding='same')
        self.c1_2 = layers.Conv2D(192, 1, padding='same')
        self.c2_2 = layers.Conv2D(224, (1, 7), padding='same')
        self.c3_2 = layers.Conv2D(256, (1, 7), padding='same')
        self.c1_3 = layers.Conv2D(192, 1, padding='same')
        self.c2_3 = layers.Conv2D(192, (1, 7), padding='same')
        self.c3_3 = layers.Conv2D(224, (7, 1), padding='same')
        self.c4_3 = layers.Conv2D(244, (1, 7), padding='same')
        self.c5_3 = layers.Conv2D(256, (7, 1), padding='same')
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
        x3 = self.c4_3(inputs)
        x3 = self.c5_3(x3)
        return self.concat([x1, x2, x3])


class InceptionC(layers.Layer):
    def __init__(self, **kwargs):
        super(InceptionC, self).__init__(**kwargs)
        self.c1_1 = layers.AvgPool2D(3, 1)
        self.c2_1 = layers.Conv2D(256, 1, padding='same')
        self.c1_2 = layers.Conv2D(256, 1, padding='same')
        self.c1_3 = layers.Conv2D(384, 1, padding='same')
        self.c2_3 = layers.Conv2D(256, (1, 3), padding='same')
        self.c2_3p = layers.Conv2D(256, (3, 1), padding='same')
        self.c1_4 = layers.Conv2D(384, 1, padding='same')
        self.c2_4 = layers.Conv2D(448, (1, 3), padding='same')
        self.c3_4 = layers.Conv2D(512, (3, 1), padding='same')
        self.c4_4 = layers.Conv2D(256, (3, 1), padding='same')
        self.c4_4p = layers.Conv2D(256, (1, 3), padding='same')
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
        self.c1_2 = layers.Conv2D(n, 3, 2, padding='valid')
        self.c1_3 = layers.Conv2D(k, 1, padding='same')
        self.c2_3 = layers.Conv2D(l, 3, padding='same')
        self.c3_3 = layers.Conv2D(m, 3, 2, padding='valid')
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
        self.c1_2 = layers.Conv2D(192, 1, padding='same')
        self.c2_2 = layers.Conv2D(192, 3, 2, padding='valid')
        self.c1_3 = layers.Conv2D(256, 1, padding='same')
        self.c2_3 = layers.Conv2D(256, (1, 7), 2, padding='same')
        self.c3_3 = layers.Conv2D(320, (7, 1), padding='same')
        self.c4_3 = layers.Conv2D(320, 3, 2, padding='valid')
        self.concat = layers.Concatenate()

    def call(self, inputs):
        x1 = self.c1_1(inputs)
        x2 = self.c1_2(x1)
        x2 = self.c2_2(x2)
        x3 = self.c1_3(inputs)
        x3 = self.c2_3(x3)
        x3 = self.c3_3(x3)
        return self.concat([x1, x2, x3])
