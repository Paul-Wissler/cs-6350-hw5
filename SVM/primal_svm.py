import pandas as pd
import numpy as np


class PrimalSvmModel:

    def __init__(self, X: pd.DataFrame, y: pd.Series, rate=0.1, rate_damping=0.1, epochs=10, 
            bias=0, random_seed=True, hyper_c=0):
        X['MODEL_BIAS'] = 1
        self.X = X.copy()
        self.y = y.copy()
        self.gamma_0 = rate
        self.a = rate_damping
        self.epochs = epochs
        self.bias = bias
        self.random_seed = random_seed
        self.weights = pd.Series([0 for _ in X.columns], index=X.columns)
        self.convergence_of_weights = pd.Series()
        self.C = hyper_c
        self.N = len(self.y)
        self.J = np.array([])
        self.create_model()

    def create_model(self):
        self.t = 0
        for epoch in range(self.epochs):
            print('EPOCH:', epoch)
            if self.random_seed:
                epoch_X = self.X.sample(frac=1)
            else:
                epoch_X = self.X.sample(frac=1, random_state=epoch)
            epoch_y = self.y.iloc[epoch_X.index.to_list()]
            self.calc_weights_per_epoch(epoch_X, epoch_y, 0)

    def calc_weights_per_epoch(self, X: pd.DataFrame, y: pd.Series, gamma_t: float):
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        for i, y_i in y.iteritems():
            self.t += 1
            gamma_t = self.calc_gamma_t(self.t)
            x_i = X.iloc[i].values
            is_error = y_i * np.dot(self.weights, x_i) <= 1
            w_0 = np.append(self.weights[:-1], 0)
            if is_error:
                self.weights -= gamma_t * w_0
                self.weights += gamma_t * self.C * self.N * y_i * x_i
            else:
                self.weights[:-1] = (1 - gamma_t) * self.weights[:-1]
            self.J = np.append(self.J, self.calc_J(y_i, x_i))

    def calc_J(self, y_i, x_i) -> float:
        # The objective function
        # J_t(w) = 1/2 * {w_0, w_0} + C * N * max(0, 1 - y_i * {w, x_i})
        # Note w_0 is weights except bias, but w includes bias
        hinge_loss = self.calc_hinge_loss(y_i, x_i)
        w_0 = self.weights.values[:-1]
        return 0.5 * np.dot(w_0, w_0) + self.C * self.N * hinge_loss

    def calc_gamma_t(self, t: int) -> float:
        return self.gamma_0 / (1 + (self.gamma_0 / self.a) * t)

    def calc_hinge_loss(self, y_i: float, x_i: np.ndarray):
        # hinge loss = max(0, 1 - y_i * {w, x_i})
        return np.max([0, 1 - y_i * np.dot(self.weights.values, x_i)])

    def test(self, X: pd.DataFrame, y: pd.Series) -> float:
        X['MODEL_BIAS'] = 1
        y_hat = self.evaluate(X)
        s = y.to_numpy() == y_hat
        return np.sum(s) / len(s)

    def evaluate(self, X: pd.DataFrame) -> np.ndarray:
        return np.sign(np.dot(X.to_numpy(), self.weights.to_numpy()))

    @staticmethod
    def compute_norm(x: np.array) -> float:
        return np.linalg.norm(x)