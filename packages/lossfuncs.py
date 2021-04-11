'''
Created on 2021年2月15日

@author: 95108
'''
import numpy as np


def square(y, predy):
    return 0.5 * (y - predy) ** 2


def squared(y, predy):
    return predy - y


def cross_entropy(y, a):
    delta = 1e-7
    return - np.dot(y, np.log(a + delta))


def cross_entropy_d(y, a):
    delta = 1e-7
    return - np.dot(y, 1 / (a + delta))


def cross_entropy_mini_batch(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, t.size)

    batch_size = t.shape[0]
    return - np.sum(t * np.log(y + 1e-7)) / batch_size


if __name__ == "__main__":
    y = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    a = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    print(cross_entropy(y, a))
    print(cross_entropy_d(y, a))
