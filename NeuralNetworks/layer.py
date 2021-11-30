import numpy as np

from .activation_functions import sigmoid


class Layer:

    def __init__(self, len_input: int, node_count: int, activation_fn=sigmoid, inst='gauss', name=None):
        self.name = name
        self.len_input = len_input
        self.node_count = node_count  # Must be >= 1
        weight_shape = [node_count, len_input]
        if inst == 'gauss':
            self.W = np.random.normal(size=weight_shape)
        elif inst == 'zero':
            self.W = np.zeros(weight_shape)
        else:
            self.W = np.empty(weight_shape)
        self.activation_fn = activation_fn

    def eval(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 1:
            x = x.reshape(1, x.shape[0])
        z = np.zeros([x.shape[0], self.node_count + 1])
        z[:, 0] = 1  # 0th index is reserved for the bias
        z[:, 1:] = sigmoid(np.dot(x, self.W.T))
        return z


class OutputLayer(Layer):

    def eval(self, x: np.ndarray) -> np.ndarray:
        return np.dot(self.W, x.T)
