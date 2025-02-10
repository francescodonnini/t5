import os
import pathlib
import random
from typing import List, Tuple, Callable
import numpy as np
import numpy.typing as npt
from PIL import Image
from keras import utils

def mklbl(i: Image) -> int:
    s = os.path.basename(i.filename)
    if s.startswith('NORMAL'):
        return 0
    elif s.startswith('VIRUS') or s.startswith('BACTERIA'):
        return 1
    else:
        raise ValueError(f'invalid string {s}')


def bicubic(i: Image, size: Tuple[int, int]) -> Image.Image:
    return i.resize(size, Image.Resampling.BICUBIC)


def bilinear(i: Image, size: Tuple[int, int]) -> Image.Image:
    return i.resize(size, Image.Resampling.BILINEAR)


def random_resample(i: Image, size: Tuple[int, int]) -> Image.Image:
    sampling = [Image.Resampling.NEAREST, Image.Resampling.BILINEAR, Image.Resampling.BICUBIC]
    return i.resize(size, random.choice(sampling))


def nearest(i: Image, size: Tuple[int, int]) -> Image.Image:
    return i.resize(size, Image.Resampling.NEAREST)


def load_data(
        data_path: str,
        resize: Tuple[int, int],
        resample: Callable[[Image.Image, Tuple[int, int]], Image.Image]=bilinear) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
    data_path = pathlib.Path(data_path)
    xs: List[npt.NDArray[np.uint8]] = []
    ys: List[int] = []
    for i in data_path.glob('**/*.jpeg'):
        i = Image.open(i)
        ys.append(mklbl(i))
        if resize is not None:
            i = resample(i, resize)
        xs.append(np.asarray(i, dtype=np.uint8).reshape(i.height, i.width, 1))
    return np.asarray(xs), utils.to_categorical(ys, num_classes=2)


def over_sampling(
        xs: npt.NDArray[np.uint8],
        ys: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
    P = len(list(filter(lambda y: y[1] > 0, ys)))
    N = int(abs(len(ys) - P))
    gap = int(abs(P - N))
    which = 0 if N < P else 1
    idx_min = list(filter(lambda j: ys[j][which] > 0, range(len(ys))))
    x_min = [xs[i] for i in idx_min]
    y_min = [ys[i] for i in idx_min]
    xs_extras = []
    ys_extras = []
    for _ in range(gap):
        i = random.randint(0, len(y_min)-1)
        xs_extras.append(x_min[i])
        ys_extras.append(y_min[i])
    for i in range(len(ys)):
        xs_extras.append(xs[i])
        ys_extras.append(ys[i])
    return np.asarray(xs_extras), np.asarray(ys_extras)
