import numpy as np
import numpy.typing as npt


def class_weights(y: npt.NDArray, beta: float=0.9999) -> dict[int, float]:
    pos = np.count_nonzero(y)
    if pos == 0:
        pos = 1
    neg = len(y) - pos
    if neg == 0:
        neg = 1
    samples_per_class = np.maximum([neg, pos], 1)
    effective_num = 1. - np.power(beta, samples_per_class)
    weights = (1. - beta) / np.array(effective_num)
    weights /= np.sum(weights) * 2
    return { 0: float(weights[0]), 1: float(weights[1]) }
