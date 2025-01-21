import os
import pathlib
from typing import Callable, Tuple, List

import numpy as np
from PIL import Image


def center(dataset):
    mean = np.mean(list(map(lambda x: x[1], dataset)))
    return list(map(lambda x: (x[0], x[1] - mean), dataset))


def load_data(data_path: str) -> List[Tuple[int, np.array]]:
    def lbl_mapping(s: str) -> int:
        if s.startswith('NORMAL'):
            return 0
        elif s.startswith('VIRUS') or s.startswith('BACTERIA'):
            return 1
        else:
            raise ValueError(f'invalid string {s}')

    def lbl(i: Image) -> int:
        return lbl_mapping(os.path.basename(i.filename))

    def array(i: Image):
        return np.asarray(i, dtype=np.uint8) / 255

    data_dir = pathlib.Path(os.path.join(data_path, 'train'))
    images = data_dir.glob('**/*.jpeg')
    return list(map(lambda i: (lbl(i), array(i)), map(lambda i: Image.open(i), images)))
