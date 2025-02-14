import random

from collections.abc import Iterable
from keras import saving
from typing import List, Any

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
