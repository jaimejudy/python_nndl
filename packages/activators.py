'''
Created on 2021年2月13日

@author: 95108
'''
import numpy as np


def pureline(x):
    return x


def purelinep(x):
    return np.ones(x.shape)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidp(x):
    return sigmoid(x) * (1 - sigmoid(x))


def step(x):
    y = x > 0
    return y.astype(np.int)


def ReLU(x):
    return np.maximum(0, x)


def ReLUp(x):
    return step(x)


def softmax(x):
    # input : x为n维向量
    # output : 也为n维向量
    m = np.max(x)
    y = np.exp(x - m)
    sumy = np.sum(y)
    return y / sumy


def softmaxp(x):
    # n维向量分别对n维分量求导，得到n阶矩阵
    return np.diag(softmax(x)) - np.outer(softmax(x), softmax(x))


def test1():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    z = sigmoidp(x)
    plt.plot(x, y)
    plt.plot(x,z)
    plt.ylim(-0.1, 1.1)
    plt.show()


def test2():
    x = np.arange(-3.0, 3.0, 0.1)
    y = ReLU(x)
    plt.plot(x, y)
    plt.show()


def test3():
    x = np.array([1,2,3,4,5,6])
    print(softmaxp(x))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test3()
#     test1()
#     test2()