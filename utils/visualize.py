import matplotlib.pyplot as plt
import numpy as np


def visualize(original, augmented):
    original = 255 - np.array(original)
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original, cmap='gray')

    augmented = 255 - np.array(augmented)
    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(augmented, cmap='gray')
