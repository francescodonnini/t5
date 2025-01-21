from keras import activations, layers, models

class ResBlock(layers.Layer):
    def __init__(self, k: (int, int), n: int, s: (int, int), **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(n, k, s)
        self.bn1   = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(n, k, s)
        self.bn2   = layers.BatchNormalization()
        self.sum   = layers.Add()

    def __call__(self, x):
        z = self.conv1(x)
        z = self.bn1(z)
        z = activations.leaky_relu(z, 0.2)
        z = self.conv2(z)
        z = self.bn2(z)
        return self.sum(x, z)


class Resizer(layers.Layer):
    def __init__(self, target: (int, int), interpolation='bilinear', r: int=4, **kwargs):
        super().__init__(**kwargs)
        ## learnable layers
        self.conv1 = layers.Conv2D(16, (7, 7), activation=activations.leaky_relu)
        self.conv2 = layers.Conv2D(16, (1, 1), activation=activations.leaky_relu)
        self.bn1   = layers.BatchNormalization()
        self.r_blocks = [ResBlock((3, 3), 16, (1, 1)) for _ in range(r)]
        self.conv3 = layers.Conv2D(16, (3, 3), (1, 1))
        self.bn2   = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(3, (7, 7), (1, 1))
        ## non-learnable layers
        self.sum  = layers.Add()
        self.sampler = layers.UpSampling2D(target, interpolation=interpolation)

    def __call__(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.bn1(y)
        y = self.sampler(y)
        z = y
        for r_block in self.r_blocks:
            z = r_block(z)
        z = self.conv3(z)
        z = self.bn2(z)
        z = self.sum(y, z)
        z = self.conv4(z)
        x = self.sampler(x)
        return self.sum(x, z)

