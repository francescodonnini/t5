from typing import List

from keras import Layer, random
from scipy import ndimage


class RandomSwitch(Layer):
    def __init__(self, choices: List[Layer], probability, **kwargs):
        super(RandomSwitch, self).__init__(**kwargs)
        self.choices = choices
        self.probability = probability

    def _random_choice(self):
        return self.choices[random.randint(0, len(self.choices))]

    def __call__(self, x):
        if random.uniform((1, 1)) < self.probability:
            return self._random_choice()(x)
        return x
