import numpy as np
import pandas as pd


class PerceptronModel:

    def __init__(self, X: pd.DataFrame, y: pd.Series, rate=0.1, epochs=10, bias=0, random_seed=True):
        X['MODEL_BIAS'] = -1
        self.X = X.copy()
        self.y = y.copy()
        self.rate = rate
        self.epochs = epochs
        self.bias = bias
        self.random_seed = random_seed
        self.weights = pd.Series([0 for _ in X.columns], index=X.columns)
        self.convergence_of_weights = pd.Series()
        self.create_model()

    def create_model(self):
        for epoch in range(self.epochs):
            if self.random_seed:
                epoch_X = self.X.sample(frac=1)
            else:
                epoch_X = self.X.sample(frac=1, random_state=epoch)
            epoch_y = self.y.iloc[epoch_X.index.to_list()]
            self.calc_weights_per_epoch(epoch_X, epoch_y)

    def calc_weights_per_epoch(self, X: pd.DataFrame, y: pd.Series):
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        for i, y_i in y.iteritems():
            w = self.weights.values.T
            x_i = X.iloc[i].values
            is_error = y_i * np.dot(w, x_i) <= 0
            if is_error:
                weight_change = self.rate * x_i * y_i
                self.convergence_of_weights = self.convergence_of_weights.append(
                    pd.Series(self.compute_norm(
                        (self.weights + weight_change).values - self.weights.values)
                    )
                ).reset_index(drop=True)
                self.weights += weight_change

    def test(self, X: pd.DataFrame, y: pd.Series) -> float:
        X['MODEL_BIAS'] = -1
        y_hat = self.evaluate(X)
        s = y.to_numpy() == y_hat
        return np.sum(s) / len(s)

    def evaluate(self, X: pd.DataFrame) -> np.ndarray:
        return np.sign(np.dot(X.to_numpy(), self.weights.to_numpy()))

    @staticmethod
    def compute_norm(x: np.array) -> float:
        return np.linalg.norm(x)
