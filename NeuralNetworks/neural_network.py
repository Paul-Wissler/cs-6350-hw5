import numpy as np
import pandas as pd

from .layer import Layer, OutputLayer


class NeuralNetwork:

    def __init__(self, cols: list, rate=0.1, rate_damping=0.1, random_seed=True):
        self.random_seed = random_seed
        self.gamma_0 = rate
        self.damping = rate_damping
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
        Z_list = [Z]
        for layer in self.hidden_layers:
            Z_list.append(layer.eval(Z_list[-1]))
        Z_list.append(self.output_layer.eval(Z_list[-1]))
        return Z_list

    def train(self, X: pd.DataFrame, Y: pd.Series, epochs=10):
        t = 0
        for epoch in range(epochs):
            if self.random_seed:
                epoch_X = X.sample(frac=1)
            else:
                epoch_X = X.sample(frac=1, random_state=epoch)
            epoch_y = Y.iloc[epoch_X.index.to_list()]
            for i in range(len(epoch_X)):
                t += 1
                X_df = pd.DataFrame()
                X_df = X_df.append(epoch_X.iloc[i])
                dL_dw = self.calc_gradient(X_df, epoch_y.iloc[i])
                gamma_t = self.calc_gamma_t(t)
                for layer, delta_w in zip(self.hidden_layers, dL_dw):
                    layer.W -= delta_w * gamma_t
                self.output_layer.W -= dL_dw[-1] * gamma_t

    def calc_gamma_t(self, t: int) -> float:
        return self.gamma_0 / (1 + (self.gamma_0 / self.damping) * t)

    def test(self, X: pd.DataFrame, y: pd.Series) -> float:
        y_hat = np.round(self.eval(X)[0])
        s = y.to_numpy() == y_hat
        return np.sum(s) / len(s)

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
        return dL_dw

    # TODO: This is a very weak function, should be able to handle
    # any number of layers, not just <= 3
    def determine_partial_deriv(self, layers: list, i: int) -> float:
        if i == 0:
            return np.array([1])
        elif i == 1:
            if len(layers[i-1].W) == 1:
                coef = layers[i-1].W[0, 1:]
            else:
                coef = layers[i-1].W[1:]
            return coef.reshape(1, coef.shape[0])
        elif i == 2:
            if len(layers[i-2].W) == 1:
                left_layer = layers[i-2].W[0, 1:]
            else:
                left_layer = layers[i-2].W[1:]
            coef = np.dot(left_layer.T, layers[i-1].W[:, 1:])
            return coef.reshape(1, coef.shape[0])
        else:
            print(i)
