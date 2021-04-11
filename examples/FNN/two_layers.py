import numpy as np
from packages.activators import *
from packages.lossfuncs import *
from packages.tools import numerical_gradient
from examples.CNN.utils import load_data
import matplotlib.pyplot as plt


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        # 初始化权重
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        z1 = np.dot(x, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        y = softmax(z2)

        return y

    def loss(self, x, y):
        y_predict = self.predict(x)
        return cross_entropy_mini_batch(y_predict, y)

    def accuracy(self, x, y):
        y_predict = self.predict(x)
        y_predict = np.argmax(y_predict, axis=1)
        t = np.argmax(y, axis=1)

        accuracy = np.sum(y_predict == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, y):
        loss_W = lambda W: self.loss(x, y)

        grads = {}
        grads['w1'] = numerical_gradient(loss_W, self.params['w1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['w2'] = numerical_gradient(loss_W, self.params['w2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads


def test():
    """a test for learning y = sin(x)"""
    # 获取数据
    x = np.arange(0, 5)
    print(x)

    # 超参数
    iter_nums = 10000
    train_size = x.shape[0]

    two_layer_net = TwoLayerNet(1, 10, 1)


def minst_train():
    x_train, t_train, x_test, t_test = load_data(flatten=True, onehot=True)

    train_loss_list = []

    # 超参数
    iter_nums = 100
    train_size = x_train.shape[0]
    batch_size = 100
    learn_rate = 0.1
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    for i in range(iter_nums):
        # 获取mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 计算梯度
        grad = network.gradient(x_batch, t_batch)

        for key in ('w1', 'b1', 'w2', 'b2'):
            network.params[key] -= learn_rate * grad[key]

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        if i % 10 == 0: print(str(i) + "mini-batches have been trained")

    plt.plot(train_loss_list)
    plt.show()


if __name__ == "__main__":
    minst_train()
