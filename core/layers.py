import numpy as np
import random

from collections.abc import Iterable
from keras import saving
from typing import List, Any, Tuple

from keras import layers

'''
RandomSwitch supplies the input x to one of the layers (randomly chosen) stored in choices
with probability p
'''
class RandomSwitch(layers.Layer):
    def __init__(self, choices: List[layers.Layer], probability: float, **kwargs):
        super(RandomSwitch, self).__init__(**kwargs)
        if isinstance(choices, Iterable) and all(RandomSwitch._is_keras_layer(i) for i in choices):
            self.choices = choices
        else:
            raise ValueError('choices must be a list or an iterable of tuples')
        if isinstance(probability, float) and 0 <= probability <= 1:
            self.probability = probability
        else:
            raise ValueError('probability must be a float comprised between 0 and 1')

    @classmethod
    def from_config(cls, config):
        probability = saving.deserialize_keras_object(config.pop('probability'))
        choices = saving.deserialize_keras_object(config.pop('choices'))
        return cls(choices, probability, **config)

    def get_config(self):
        config = super().get_config()
        config.update({
            'probability': saving.serialize_keras_object(self.probability),
            'choices': saving.serialize_keras_object(self.choices)
        })
        return config

    @staticmethod
    def _is_keras_layer(i: Any) -> bool:
        return isinstance(i, layers.Layer)

    def _random_choice(self):
        return random.choice(self.choices)

    def build(self, input_shape):
        super(RandomSwitch, self).build(input_shape)

    def call(self, inputs, training=None):
        if training:
            if random.uniform(0, 1) < self.probability:
                return self._random_choice().call(inputs)
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


class RandomCutout(layers.Layer):
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
        return np.asarray([self.cutout(i) for i in inputs])

    def compute_output_shape(self, input_shape):
        return input_shape

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
