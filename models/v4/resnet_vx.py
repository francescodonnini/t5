from keras import layers

class InceptionAVx(layers.Layer):
    def __init__(self, f23: int, f33: int, f: int,  **kwargs):
        super(InceptionAVx, self).__init__(**kwargs)
        self.c1_1 = layers.Conv2D(32, 1, padding='same')
        self.c1_2 = layers.Conv2D(32, 1, padding='same')
        self.c2_2 = layers.Conv2D(32, 3, padding='same')
        self.c1_3 = layers.Conv2D(32, 1, padding='same')
        self.c2_3 = layers.Conv2D(f23, 3, padding='same')
        self.c3_3 = layers.Conv2D(f33, 3, padding='same')
        self.concat = layers.Concatenate()
        self.conv = layers.Conv2D(f, 1, padding='same', activation='linear')
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
        return self.add([inputs, x])

    @staticmethod
    def v1(**kwargs):
        return InceptionAVx(32, 32, 256, **kwargs)

    @staticmethod
    def v2(**kwargs):
        return InceptionAVx(48, 64, 384, **kwargs)


class InceptionBVx(layers.Layer):
    def __init__(self, f11: int, f22: int, f32: int, f: int, **kwargs):
        super(InceptionBVx, self).__init__(**kwargs)
        self.c1_1 = layers.Conv2D(f11, 1, padding='same')
        self.c1_2 = layers.Conv2D(128, 1, padding='same')
        self.c2_2 = layers.Conv2D(f22, (1, 7), padding='same')
        self.c3_2 = layers.Conv2D(f32, (7, 1), padding='same')
        self.concat = layers.Concatenate()
        self.conv = layers.Conv2D(f, 1, padding='same', activation='linear')
        self.add = layers.Add()

    def call(self, inputs):
        x1 = self.c1_1(inputs)
        x2 = self.c1_2(inputs)
        x2 = self.c2_2(x2)
        x2 = self.c3_2(x2)
        x = self.concat([x1, x2])
        x = self.conv(x)
        return self.add([inputs, x])

    @staticmethod
    def v1(**kwargs):
        return InceptionBVx(128, 128, 128, 896, **kwargs)

    @staticmethod
    def v2(**kwargs):
        return InceptionBVx(192, 160, 192, 1154. **kwargs)


class InceptionCVx(layers.Layer):
    def __init__(self, f22: int, f32: int, f: int, **kwargs):
        super(InceptionCVx, self).__init__(**kwargs)
        self.c1_1 = layers.Conv2D(192, 1, padding='same')
        self.c1_2 = layers.Conv2D(192, 1, padding='same')
        self.c2_2 = layers.Conv2D(f22, (1, 3), padding='same')
        self.c3_2 = layers.Conv2D(f32, (3, 1), padding='same')
        self.concat = layers.Concatenate()
        self.conv = layers.Conv2D(f, 1, padding='same', activation='linear')
        self.add = layers.Add()

    def call(self, inputs):
        x1 = self.c1_1(inputs)
        x2 = self.c1_2(inputs)
        x2 = self.c2_2(x2)
        x2 = self.c3_2(x2)
        x = self.concat([x1, x2])
        x = self.conv(x)
        return self.add([inputs, x])

    @staticmethod
    def v1(**kwargs):
        return InceptionCVx(192, 192, 1792, **kwargs)

    @staticmethod
    def v2(**kwargs):
        return InceptionCVx(224, 256, 2048, **kwargs)


class ReductionBVx(layers.Layer):
    def __init__(self, f23: int, f24: int, f34: int, **kwargs):
        super(ReductionBVx, self).__init__(**kwargs)
        self.c1_1 = layers.MaxPool2D(3, 2, padding='valid')
        self.c1_2 = layers.Conv2D(256, 1, padding='same')
        self.c2_2 = layers.Conv2D(384, 3, 2, padding='valid')
        self.c1_3 = layers.Conv2D(256, 1, padding='same')
        self.c2_3 = layers.Conv2D(f23, 3, 2, padding='valid')
        self.c1_4 = layers.Conv2D(256, 1, padding='same')
        self.c2_4 = layers.Conv2D(f24, 3, padding='same')
        self.c3_4 = layers.Conv2D(f34, 3, 2, padding='valid')
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

    @staticmethod
    def v1(**kwargs):
        return ReductionBVx(256, 256, 256, **kwargs)

    @staticmethod
    def v2(**kwargs):
        return ReductionBVx(256, 288, 320, **kwargs)
