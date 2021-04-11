"""
编写一个两层的神经网络，用于训练
"""

import numpy as np
from packages.activators import *
import packages.lossfuncs as loss
from packages.tools import *


class Layer():
    def __init__(self, Ml, func, deriv, W, b):
        # Ml 全连接层的神经元个数
        # func 该层的激活函数
        # deriv 激活函数导数
        # W : 权重
        # b : 偏置
        self.M = Ml
        self.func = func
        self.deriv = deriv
        self.W = W
        self.b = b


class FNN:
    BATCH = 100
    RANDOM = 200
    ALL = 300

    def __init__(self, L, Mls, Funcs, Derivs, dimInput, Loss, LossDeriv):
        self.L = L
        Layers = []
        for i in range(L):
            b = np.zeros([Mls[i]])
            W = np.random.normal(size=[Mls[i], Mls[i - 1] if i > 0 else dimInput])
            Layers.append(Layer(Mls[i], Funcs[i], Derivs[i], W, b))
        self.Layers = Layers
        self.dimInput = dimInput
        self.Loss = Loss
        self.LossDeriv = LossDeriv
        self.learnRate = 1
        self.maxEpoch = 5000

    def forwardPropagation(self, inputData, y):
        # 正向计算神经网络，输出每一层的激活
        if len(inputData) != self.dimInput:
            raise ValueError("Dimension did not comply")
        outputs = [np.array(inputData)]
        zs = []
        a = inputData
        for i in range(self.L):
            z = np.matmul(self.Layers[i].W, a) + self.Layers[i].b
            a = self.Layers[i].func(z)
            zs.append(np.array(z, copy=True))
            outputs.append(np.array(a, copy=True))
        lossValue = self.Loss(y, outputs[-1])
        return zs, outputs, lossValue

    def predict(self, input_data):
        a = input_data
        for i in range(self.L):
            z = np.matmul(self.Layers[i].W, a) + self.Layers[i].b
            a = self.Layers[i].func(z)
        return a

    def backPropagation(self, y, outputs, zs):
        '''
        # 误差反向传播并更新权重
        '''
        delta = self.Layers[-1].deriv(zs[-1]) * self.LossDeriv(y, outputs[-1])
        deltas = [np.array(delta, copy=True)]
        i = self.L - 1
        while i > 0:
            delta = self.Layers[i - 1].deriv(zs[i - 1]) * np.matmul(self.Layers[i].W.T, deltas[-1])
            deltas.append(np.array(delta))
            i -= 1
        deltas.reverse()
        return deltas

    def gradient(self, inputData, y):
        # 更新一次权重
        zs, outputs, lossValue = self.forwardPropagation(inputData, y)
        deltas = self.backPropagation(y, outputs, zs)
        for i in range(self.L):
            print(np.outer(deltas[i], outputs[i]))
            # self.Layers[i].W -= self.learnRate * np.outer(deltas[i], outputs[i].T)
            # self.Layers[i].b -= self.learnRate * deltas[i]

    def setLearnRate(self, learnRate):
        self.learnRate = learnRate

    def train(self, mode, inputData, labelY, epsilon):
        m, n = inputData.shape
        if m != self.dimInput:
            raise ValueError("train error, dim not comply")
        epoch = 0
        losses = []
        while True:
            epoch += 1
            if (epoch + 1) % 100 == 0:
                print("have trained " + str(epoch + 1) + " epochs")
            dW, db = [], []
            lossTotal = 0
            for j in range(n):
                # 此处可以利用GPU并行计算
                zs, outputs, lossValue = self.forwardPropagation(inputData[:, j], labelY[j])
                deltas = self.backPropagation(labelY[j], outputs, zs)
                lossTotal += lossValue
                if not dW:
                    for i in range(self.L):
                        dW.append(- np.outer(deltas[i], outputs[i]))
                        db.append(- deltas[i])
                else:
                    for i in range(self.L):
                        dW[i] -= np.outer(deltas[i], outputs[i])
                        db[i] -= deltas[i]
            # 更新每层的W和b，先判断是否满足停止条件
            losses.append(lossTotal)
            if lossTotal / n < epsilon or epoch > self.maxEpoch:
                break
            for i in range(self.L):
                self.Layers[i].W += self.learnRate * dW[i] / n
                self.Layers[i].b += self.learnRate * db[i] / n
        return epoch, losses

    def setW(self, W):
        for i in range(self.L):
            self.Layers[i].W = W[i]

    def loss(self, input_data, Y):
        a = self.predict(input_data)
        return self.Loss(Y, a)


def testnumpy():
    a = np.random.uniform(size=[3, 3])
    print(a)
    b = np.random.uniform(size=[3, 1])
    print(b)
    print(np.matmul(a, b))


# def test():
#     L = 2
#     Mls = [10, 1]
#     Funcs = [sigmoid, sigmoid]
#     Derivs = [sigmoidp, sigmoidp]
#     dimInput = 1
#     Loss = loss.square
#     LossDeriv = loss.squared
#     network = FNN(L, Mls, Funcs, Derivs, dimInput, Loss, LossDeriv)
#
#     inputData, y = np.array([3]), 0
#     network.renew(inputData, y)
#     network.renew(inputData, y)
#     network.renew(inputData, y)
#     network.renew(inputData, y)
#     network.renew(inputData, y)
#
#     zs, outputs, _ = network.forwardPropagation(inputData, y)
#     print(outputs)


def testTrain():
    L = 2
    Mls = [20, 1]
    Funcs = [sigmoid, pureline]
    Derivs = [sigmoidp, purelinep]
    dimInput = 1
    Loss = loss.square
    LossDeriv = loss.squared
    network = FNN(L, Mls, Funcs, Derivs, dimInput, Loss, LossDeriv)
    network.setLearnRate(0.01)

    inputData = [[0.01 * i] for i in range(400)]
    Y = [np.sin(x[0]) for x in inputData]
    # print(Y)
    inputData = np.array(inputData).T
    epoch, losses = (network.train(None, inputData, Y, 1e-5))
    # zs, outputs, lossValue = network.forwardPropagation([1], np.sin(1))
    # print(outputs[-1], lossValue)
    # print(np.sin(1))
    print("epoch = ", epoch)

    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.show()

    predy = []
    for i in range(400):
        # print(inputData[:, i])
        predy.append(network.predict(inputData[:, i] + 0.005))
    plt.plot(predy)
    plt.plot(Y)
    plt.show()
    print(predy)


def test_gradient():
    np.random.seed(1)

    L = 3
    Mls = [3, 3, 1]
    Funcs = [sigmoid, sigmoid, sigmoid]
    Derivs = [sigmoidp, sigmoidp, sigmoidp]
    dimInput = 2
    Loss = loss.square
    LossDeriv = loss.squared
    network = FNN(L, Mls, Funcs, Derivs, dimInput, Loss, LossDeriv)
    network.setLearnRate(0.05)

    inputData = np.asarray([1, 2]).T
    y = 0
    # print(network.predict(inputData))

    def func(W):
        return network.loss(inputData, y)

    print("automatic")
    network.gradient(inputData, y)

    # 数值微分结果
    print("Numerical")
    print(numerical_gradient(func, network.Layers[0].W))
    print(numerical_gradient(func, network.Layers[1].W))


if __name__ == '__main__':
    # test_gradient()
    testTrain()
#     a = np.asarray([[1,2,3], [2,3,4]])
#     print(a[:, 1])
