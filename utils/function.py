import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)


def sigmoid_dash(x):
    return x * (1 - x)


def cross_entropy(y, t):
    E = 0.0
    for i in range(len(t[0])):
        E = E - t[0][i] * np.log(y[0][i])
    return E.astype(np.float32)
