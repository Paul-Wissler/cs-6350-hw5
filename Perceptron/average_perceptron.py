import numpy as np
import pandas as pd

from .perceptron import PerceptronModel


class AveragePerceptronModel(PerceptronModel):
    averaged_weights = pd.Series()

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
                self.weights = self.weights + weight_change
            else:
                if self.averaged_weights.empty:
                    self.averaged_weights = self.weights.copy()
                else:
                    self.averaged_weights += self.weights

    def test(self, X: pd.DataFrame, y: pd.Series) -> float:
        X['MODEL_BIAS'] = -1
        y_hat = self.evaluate(X)
        s = y.to_numpy() == y_hat
        return np.sum(s) / len(s)

    def evaluate(self, X: pd.DataFrame) -> np.ndarray:
        return np.sign(np.dot(X.to_numpy(), self.averaged_weights.to_numpy()))
