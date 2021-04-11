import datasets.mnist.readimage as readimage
import numpy as np
import matplotlib.pyplot as plt


def load_data(flatten=False, onehot=False):
    train_data = readimage.read_training_data(flatten)
    train_labels = readimage.read_training_labels(onehot)

    test_data = readimage.read_testing_data(flatten)
    test_labels = readimage.read_testing_labels(onehot)

    # for i in range(2):
    #     plt.imshow(train_data[i], cmap='gray')
    #     plt.show()
    #     plt.pause(2)

    return train_data, train_labels, test_data, test_labels


if __name__ == '__main__':
    _, labels, _, _ = load_data(flatten=True, onehot=False)
    for i in range(10):
        print(labels[i])
    # readimage.read_testing_data()
