from two_layers import TwoLayerNet
import numpy as np
from packages.activators import *
from packages.lossfuncs import *


class TwoLayerNetF(TwoLayerNet):
    def __init__(self, input_size, hidden_size, output_size, weight_int_std):
        super().__init__(input_size=input_size, hidden_size=hidden_size, output_size=output_size, weight_int_std=weight_int_std)
        self.params['act1'] = sigmoid
        self.params['act2'] = softmax
        self.params['loss'] = cross_entropy_mini_batch

    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        act1, act2 = self.params['act1'], self.params['act2']
        z1 = np.dot(x, w1) + b1
        a1 = act1(z1)
        z2 = np.dot(a1, w2) + b2
        y = act2(z2)
        return y

    def set_activators(self, activator_name, function):
        self.params[activator_name] = function

    def set_lossfunction(self, function):
        self.params['loss'] = function


if __name__ == "__main__":
    net = TwoLayerNetF(input_size=784, hidden_size=50, output_size=10)