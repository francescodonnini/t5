from keras import losses
import numpy as np
import numpy.typing as npt
import tensorflow as tf


def class_weights(y: npt.NDArray, beta: float=0.9999) -> npt.NDArray[np.float64]:
    def expected_volume(n: int) -> float:
        return (1 - beta ** n)/(1 - beta)

    pos = np.count_nonzero(y)
    if pos == 0:
        pos = 1
    neg = len(y) - pos
    if neg == 0:
        neg = 1
    return np.asarray([1/expected_volume(pos), 1/expected_volume(neg)])


def weighted_bce(t, y, w):
    t = tf.cast(t, dtype=tf.float32)
    weights = t * w[0] + (1. - t) * w[1]
    return tf.reduce_mean(weights * losses.binary_crossentropy(t, y))
