import numpy as np
import random

def oversample(x_train, y_train, trans_fn):
    TOTAL = len(y_train)
    positives = list(filter(lambda i: i[1] == 1, enumerate(y_train)))
    P = len(positives)
    N = TOTAL - P
    print(f'total      = {TOTAL}')
    print(f'#positives = {P}')
    print(f'#negatives = {N}')
    minority = positives
    if P > N:
        minority = list(filter(lambda i: i[0] == 1, enumerate(y_train)))
    gap = int(abs(P - N))
    x_gap = []
    y_gap = []
    for _ in range(gap):
        i = random.choice(minority)
        fn = random.choice(trans_fn)
        x_gap.append(fn(x_train[i]))
        y_gap.append(fn(y_train[i]))
    return np.array(x_gap), np.array(y_gap)