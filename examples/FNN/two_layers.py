import numpy as np
from packages.activators import *
from packages.lossfuncs import *
from packages.tools import numerical_gradient


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
        return cross_entropy(y, y_predict)

    def accuracy(self, x, y):
        y_predict = self.predict(x)
        y_predict = np.argmax(y_predict, axis=1)
        t = np.argmax(y, axis=1)

        accuracy = np.sum(y_predict == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, y):
        loss_W = lambda W : self.loss(x, y)

        grads = {}
        grads['w1'] = numerical_gradient(loss_W, self.params['w1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['w2'] = numerical_gradient(loss_W, self.params['w2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads


def test():
    two_layer_net = TwoLayerNet(1, 10, 1)


if __name__ == "__main__":
    test()
