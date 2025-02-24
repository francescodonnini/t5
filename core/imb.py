import numpy as np
import numpy.typing as npt


def class_weights(y: npt.NDArray, beta: float=0.9999) -> dict[int, float]:
    def expected_volume(n: int) -> float:
        return (1 - beta ** n)/(1 - beta)

    pos = np.count_nonzero(y)
    if pos == 0:
        pos = 1
    neg = len(y) - pos
    if neg == 0:
        neg = 1
    return { 0: 1/expected_volume(pos), 1: 1/expected_volume(neg) }
