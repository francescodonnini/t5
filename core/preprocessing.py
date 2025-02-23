import numpy as np
import tensorflow as tf


def mixup(sample_a, sample_b, alpha):
    img_a, y_a = sample_a
    img_b, y_b = sample_b
    lambd_ = np.random.beta(alpha, alpha)
    return lambd_ * img_a + (1 - lambd_) * img_b, lambd_ * y_a + (1 - lambd_) * y_b

def cutmix(sample_a, sample_b):
    img_a, y_a = sample_a
    img_b, y_b = sample_b
    (w, h, _) = img_a.shape
    lambd_ = np.random.uniform(0,1)
    x1, y1, w1, h1 = bbox(w, h, lambd_)
    img_c = remove_patch(img_a, x1, y1, w1, h1)
    patch_b = read_patch(img_b, x1, y1, w1, h1)
    y_a = tf.cast(y_a, tf.float32)
    y_b = tf.cast(y_b, tf.float32)
    l = tf.cast(1 - float(w1 * h1) / (w * h), dtype=tf.float32)
    y = l * y_a + (1 - l) * y_b
    return img_c + patch_b, y

def random_crop(w, h, l):
    rw = int(w * np.sqrt(1-l))
    rh = int(h * np.sqrt(1-l))
    rx = int(np.random.uniform(0, w))
    ry = int(np.random.uniform(0, h))
    return rx, ry, rw, rh

def bbox(w, h, l):
    rx, ry, rw, rh = random_crop(w, h, l)
    x1 = np.clip(rx - rw // 2, 0, w)
    x2 = np.clip(rx + rw // 2, 0, w)
    x2 -= x1
    if x2 == 0:
        x2 = 1
    y1 = np.clip(ry - rh // 2, 0, h)
    y2 = np.clip(ry + rh // 2, 0, h)
    y2 -= y1
    if y2 == 0:
        y2 = 1
    return x1, y1, x2, y2

def remove_patch(img, x1, y1, w1, h1):
    m = np.ones(img.shape)
    m[x1:w1, y1:h1] = 0
    return img * m

def read_patch(img, x1, y1, w1, h1):
    m = np.zeros(img.shape)
    m[x1:w1, y1:h1] = 1
    return img * m
