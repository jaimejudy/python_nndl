from datasets.mnist import readimage


def main():
    train_data, train_label, test_data, test_label = readimage.read_mnist(flatten=True)


if __name__ == '__main__':
    main()