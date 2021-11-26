import numpy as np
import pandas as pd

from .layer import Layer, OutputLayer


class NeuralNetwork:

    def __init__(self, cols: list):
        self.columns = ['MODEL_BIAS'] + cols
        self.hidden_layers = list()
        self.output_layer = None

    def add_hidden_layers(self, layer_list: list):
        self.hidden_layers.extend(layer_list)

    def add_output_layer(self, output_layer: OutputLayer):
        self.output_layer = output_layer

    def eval(self, X_df: pd.DataFrame) -> np.ndarray:
        return self.eval_nodes(X_df)[-1]

    def eval_nodes(self, X_df: pd.DataFrame) -> np.ndarray:
        X_df['MODEL_BIAS'] = 1
        Z = X_df[self.columns].values
        Z_list = list(Z)
        for layer in self.hidden_layers:
            Z_list.append(layer.eval(Z_list[-1]))
        Z_list.append(self.output_layer.eval(Z_list[-1]))
        return Z_list

    def calc_gradient(self, X: pd.DataFrame, y_star: float) -> np.ndarray:
        Z = self.eval_nodes(X)
        y = Z[-1]
        dL_dy = y - y_star
        layers = self.hidden_layers + [self.output_layer]
        layers.reverse()
        Z = Z[:-1]
        Z.reverse()
        i = 0
        dL_dw = list()
        for layer, z in zip(layers, Z):
            if len(z.shape) == 1:
                z = z.reshape(1, z.shape[0])
            inner_coef = self.determine_partial_deriv(layers, i)
            dL_dw.append(dL_dy * np.dot(inner_coef.T, z))
            i += 1
        dL_dw.reverse()
        print(dL_dw)
        return dL_dw

    # TODO: This is a very weak function, should be able to handle
    # any number of layers, not just <= 3
    def determine_partial_deriv(self, layers: list, i: int) -> float:
        if i == 0:
            return np.array([1])
        elif i == 1:
            coef = layers[i-1].W[1:]
            return coef.reshape(1, coef.shape[0])
        elif i == 2:
            coef = np.dot(layers[i-2].W[1:].T, layers[i-1].W[:, 1:])
            return coef.reshape(1, coef.shape[0])
