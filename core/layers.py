from collections.abc import Iterable
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from keras import layers as ly
from keras import saving


class RandomApply(ly.Layer):
    def __init__(self, probability: float, layer: ly.Layer, **kwargs):
        super(RandomApply, self).__init__(**kwargs)
        self.probability = RandomApply.check_probability(probability)
        self.layer = RandomApply.check_layer(layer)

    @staticmethod
    def check_layer(layer: ly.Layer):
        if not isinstance(layer, ly.Layer):
            raise ValueError('layer must be a keras.Layer')
        return layer

    @staticmethod
    def check_probability(probability: float):
        if not isinstance(probability, float) or probability < 0 or probability > 1:
            raise ValueError('probability must be a float between 0 and 1')
        return probability

    @classmethod
    def from_config(cls, config):
        probability = saving.deserialize_keras_object(config.pop('probability'))
        layer = saving.deserialize_keras_object(config.pop('layer'))
        return cls(probability, layer, **config)

    def get_config(self):
        config = super().get_config()
        config.update({
            'probability': saving.serialize_keras_object(self.probability),
            'layer': saving.serialize_keras_object(self.layer)
        })
        return config

    def build(self, input_shape):
        super(RandomApply, self).build(input_shape)

    def call(self, inputs, training=None):
        if training:
            if np.random.uniform(0, 1) < self.probability:
                return self.layer(inputs)
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


class RandomCutout(ly.Layer):
    def __init__(self, size: Tuple[int, int]|int=16,  n_holes: int=1, name=None, **kwargs):
        super(RandomCutout, self).__init__(name=name, **kwargs)
        self.target_width, self.target_height = RandomCutout.check_target_size(size)
        if isinstance(n_holes, int) and n_holes > 0:
            self.n_holes = n_holes
        else:
            raise ValueError('n_holes must be a positive integer')

    @staticmethod
    def check_target_size(target_size: Tuple[int, int]|int) -> Tuple[int, int]:
        if isinstance(target_size, int):
            if target_size <= 0:
                raise ValueError('target_size must be either a single or a pair positive integers')
            return target_size, target_size
        elif isinstance(target_size, (int, int)):
            target_width, target_height = target_size
            if target_width <= 0 or target_height <= 0:
                raise ValueError('target_size must be either a single or a pair positive integers')
            return target_width, target_height
        else:
            raise ValueError('target_size must be a tuple (int, int) or int')

    @classmethod
    def from_config(cls, config):
        size = saving.deserialize_keras_object(config.pop('size'))
        n_holes = saving.deserialize_keras_object(config.pop('n_holes'))
        return cls(size, n_holes, **config)

    def get_config(self):
        config = super().get_config()
        config.update({
            'size': saving.serialize_keras_object((self.target_width, self.target_height)),
            'n_holes': saving.serialize_keras_object(self.n_holes)
        })
        return config

    def call(self, inputs, training=True):
        if training:
            return tf.map_fn(self.cutout, inputs)
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    @tf.function
    def cutout(self, img):
        w, h, _ = img.shape
        mask = np.ones((w, h, 1), np.float32)
        for _ in range(self.n_holes):
            x1, y1, x2, y2 = RandomCutout.random_hole(w, h, self.target_width, self.target_height)
            mask[x1:x2, y1:y2] = 0
        return img * mask

    @staticmethod
    def random_center(w: int, h: int) -> Tuple[int, int]:
        return np.random.randint(w), np.random.randint(h)

    @staticmethod
    def random_hole(w: int, h: int, target_w: int, target_h: int) -> Tuple[int, int, int, int]:
        c_x, c_y = RandomCutout.random_center(w, h)
        y1 = np.clip(c_y - target_h // 2, 0, h)
        y2 = np.clip(c_y + target_h // 2, 0, h)
        x1 = np.clip(c_x - target_w // 2, 0, w)
        x2 = np.clip(c_x + target_w // 2, 0, w)
        return x1, y1, x2, y2


class RandomSample(ly.Layer):
    def __init__(self, layer_list: List[ly.Layer], n: int, name=None, **kwargs):
        super(RandomSample, self).__init__(name=name, **kwargs)
        self.layer_list = layer_list
        self.n = n

    @classmethod
    def from_config(cls, config):
        n = saving.deserialize_keras_object(config.pop('n'))
        layer_list = saving.deserialize_keras_object(config.pop('layer_list'))
        return cls(n, layer_list, **config)

    def get_config(self):
        config = super().get_config()
        config.update({
            'n': saving.serialize_keras_object(self.n),
            'layer_list': saving.serialize_keras_object(self.layer_list)
        })
        return config

    def build(self, input_shape):
        super(RandomSample, self).build(input_shape)

    def call(self, inputs, training=None):
        if training:
            for i in self.samples():
                inputs = self.layer_list[i](inputs)
        return inputs

    def samples(self) -> Iterable[int]:
        s = list(range(len(self.layer_list)))
        for _ in range(2):
            np.random.shuffle(s)
        return list(s[:self.n])

    def random_sample(self) -> int:
        return np.random.randint(0, len(self.layer_list) - 1)

    def compute_output_shape(self, input_shape):
        return input_shape
