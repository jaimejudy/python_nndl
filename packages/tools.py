import numpy as np


def numerical_gradient_2d(func, x):
    h = 1e-4
    grad = np.zeros_like(x)
    m, n = x.shape
    for idx in range(m):
        for idy in range(n):
            tmp_val = x[idx, idy]
            x[idx, idy] += h
            fxh1 = func(x)

            x[idx, idy] -= 2 * h
            fxh2 = func(x)

            grad[idx, idy] = (fxh1 - fxh2) / (2 * h)
            x[idx, idy] = tmp_val

    return grad


def numerical_gradient(f, x):
    h = 1e-4

    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] += h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
        it.iternext()

    return grad


if __name__ == "__main__":
    x = np.asarray([[1, 2], [2, 3]])
    print(x.size)
    x = np.zeros(5)
    x = x.reshape(5, 1)
    print(x[0])
    print(x.shape)

    f = lambda y: y[0] ** 2 + y[1]
    # print(f(np.asarray([[1.0, 2.0], [3.0, 4.0]])))
    # print(numerical_gradient(f, np.asarray([[1.0, 2.0], [3.0, 4.0]])))
    print(np.dot(np.asarray([1,2,3]), np.asarray([1,1,1])))
