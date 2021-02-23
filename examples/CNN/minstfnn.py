import examples.FNN.fnn_layers as fnn
import packages.activators as activators
import packages.lossfuncs as lossfuncs
from utils import *


def minst_learn():
    train_data, train_label, test_data, test_label = load_data(flatten=True)
    L = 2
    Mls = [100, 10]
    Funcs = [activators.sigmoid, activators.softmax]
    Derivs = [activators.sigmoidp, activators.softmaxp]
    dimInput = 784
    Loss = lossfuncs.cross_entropy
    LossDeriv = lossfuncs.cross_entropy_d
    network = fnn.FNN(L, Mls, Funcs, Derivs, dimInput, Loss, LossDeriv)


if __name__ == "__main__":
    minst_learn()
