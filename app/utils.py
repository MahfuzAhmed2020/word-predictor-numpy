import numpy as np

def one_hot(index, size):
    vec = np.zeros(size)
    vec[index] = 1
    return vec


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)