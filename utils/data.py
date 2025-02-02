import os
import pathlib
from typing import List, Tuple
import numpy as np
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


def mktensor(i: Image) -> np.ndarray[Tuple[int, int, int], np.uint8]:
    return np.asarray(i, dtype=np.uint8).reshape(i.height, i.width, 1)


def load_data(data_path: str, resize: Tuple[int, int]=None) -> Tuple[List[np.ndarray[Tuple[int, int, int], np.uint8]], List[int]]:
    data_dir = pathlib.Path(os.path.join(data_path, 'train'))
    xs: List[np.ndarray[Tuple[int, int, 1], np.uint8]] = []
    ys: List[int] = []
    for i in data_dir.glob('**/*.jpeg'):
        i = Image.open(i)
        ys.append(mklbl(i))
        if resize is not None:
            i = i.resize(resize)
        xs.append(mktensor(i))
    return np.asarray(xs), utils.to_categorical(np.asarray(ys), num_classes=2)
