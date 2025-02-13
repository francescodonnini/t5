import io
import os
import pathlib
import random
import zipfile as zf
from typing import List, Tuple, Callable, Iterable

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


def from_dir(
        data_path: str,
        resize: Tuple[int, int],
        resample: Callable[[Image.Image, Tuple[int, int]], Image.Image]=bilinear,
        file_format: str='.jpeg') -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
    images = [Image.open(p) for p in pathlib.Path(data_path).glob(f'**/*{file_format}')]
    return _prepare_data(images, resize, resample)


def from_zip(
        data_path: str,
        selector: Callable[[str], bool],
        resize: Tuple[int, int],
        resample: Callable[[Image.Image, Tuple[int, int]], Image.Image]=bilinear) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
    with zf.ZipFile(data_path) as z:
        images = []
        for file in filter(lambda f: selector(f), z.namelist()):
            i = Image.open(io.BytesIO(z.read(file)))
            i.filename = file
            images.append(i)
        return _prepare_data(images, resize, resample)


def _prepare_data(
        images: Iterable[Image.Image],
        resize,
        resample):
    xs: List[npt.NDArray[np.uint8]] = []
    ys: List[int] = []
    for i in images:
        ys.append(mklbl(i))
        if resize is not None:
            i = resample(i, resize)
        xs.append(np.asarray(i, dtype=np.uint8).reshape(i.height, i.width, 1))
    return np.asarray(xs), utils.to_categorical(ys, num_classes=2)

def over_sampling(
        xs: npt.NDArray[np.uint8],
        ys: npt.NDArray[np.float32],
        ratio: float=1.0) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
    if ratio > 1 or ratio < 0:
        raise ValueError('ratio must be between 0 and 1')
    positives = len(list(filter(lambda y: y[1] > 0, ys)))
    negatives = int(abs(len(ys) - positives))
    gap = int(int(abs(positives - negatives)) * ratio)
    which = 0 if negatives < positives else 1
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
