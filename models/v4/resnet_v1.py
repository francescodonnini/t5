from keras import layers


class StemV1(layers.Layer):
    def __init__(self, **kwargs):
        super(StemV1, self).__init__(**kwargs)
        self.c1 = layers.Conv2D(32, 3, strides=2, padding='valid')
        self.c2 = layers.Conv2D(32, 3, padding='valid')
        self.c3 = layers.Conv2D(64, 3, padding='same')
        self.c4 = layers.MaxPool2D(3, 2, padding='valid')
        self.c5 = layers.Conv2D(64, 3, padding='same')
        self.c6 = layers.MaxPool2D(3, 2, padding='valid')
        self.c7 = layers.Conv2D(80, 1, padding='same')
        self.c8 = layers.Conv2D(192, 3, padding='valid')
        self.c9 = layers.Conv2D(256, 3, 2, padding='valid')

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        x = self.c7(x)
        x = self.c8(x)
        return self.c9(x)
