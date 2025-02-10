import matplotlib.pyplot as plt
import numpy as np


def visualize(original, augmented):
    original = np.array(original)
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original, cmap='gray')

    augmented = np.array(augmented)
    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(augmented, cmap='gray')


def show(i):
    i = np.array(i)
    plt.subplot(1, 2, 1)
    plt.title('image')
    plt.imshow(i, cmap='gray')