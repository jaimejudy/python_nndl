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


if __name__ == "__main__":
    y = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    a = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    print(cross_entropy(y, a))
    print(cross_entropy_d(y, a))
