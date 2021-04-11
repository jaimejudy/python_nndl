import os
import struct
import sys

import matplotlib.pyplot as plt
import numpy as np

print(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))


def read_image(filename, flatten=False):
    f = open(filename, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, f, offset)
    print(magic_number, num_images, num_rows, num_cols)

    if flatten:
        num_cols = num_rows * num_cols
        num_rows = 1

    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    # print('offset', offset)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print("已解析" + str(i + 1) + '张')

        images[i] = np.array(struct.unpack_from(fmt_image, f, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)

    return images


def read_labels(filename, onehot=False):
    f = open(filename, 'rb').read()

    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, f, offset)
    print('魔术' + str(magic_number) + '标签数 ' + str(num_images))

    offset += struct.calcsize(fmt_header)
    fmt_images = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_images, f, offset)[0]
        offset += struct.calcsize(fmt_images)
    if onehot:
        v = np.zeros((num_images, 10))
        for i in range(num_images):
            v[i, int(labels[i])] = 1
        labels = v
    return labels


def read_training_data(flatten=False):
    return read_image(r'D:\workspace\pycharm\p03_nndl\datasets\mnist\train-images.idx3-ubyte', flatten)


def read_training_labels(onehot=False):
    labels = read_labels(r'D:\workspace\pycharm\p03_nndl\datasets\mnist\train-labels.idx1-ubyte', onehot)
    return labels


def read_testing_data(flatten=False):
    return read_image(r'D:\workspace\pycharm\p03_nndl\datasets\mnist\t10k-images.idx3-ubyte', flatten)


def read_testing_labels(onehot=False):
    return read_labels(r'D:\workspace\pycharm\p03_nndl\datasets\mnist\t10k-labels.idx1-ubyte', onehot)


def read_mnist(flatten=False, onehot=False):
    train_data = read_training_data(flatten)
    train_label = read_training_labels(onehot)
    test_data = read_testing_data(flatten)
    test_label = read_testing_labels(onehot)
    return train_data, train_label, test_data, test_label


def test_read_training_data():
    images = read_training_data()
    labels = read_training_labels()
    for i in range(10):
        print(labels[i])
        plt.imshow(images[i], cmap='gray')
        plt.show()
        plt.pause(2)


if __name__ == '__main__':
    test_read_training_data()
