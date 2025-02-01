import os
import pathlib
from typing import Callable, Tuple, List

import numpy as np
from PIL import Image


def center(dataset):
    mean = np.mean(list(map(lambda x: x[1], dataset)))
    return list(map(lambda x: (x[0], x[1] - mean), dataset))


from PIL import Image
from typing import List, Tuple
import numpy as np
import os
import pathlib

def load_data(data_path: str) -> Tuple[List[np.array], List[int]]:
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
        return np.asarray(i, dtype=np.uint8)

    data_dir = pathlib.Path(os.path.join(data_path, 'train'))
    xs = []
    ys = []
    for i in data_dir.glob('**/*.jpeg'):
        i = Image.open(i)
        xs.append(array(i))
        ys.append(lbl(i))
    return (xs, ys)

