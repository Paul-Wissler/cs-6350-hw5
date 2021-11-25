import numpy as np
import pandas as pd

from .batch_gradient_descent import BatchGradientDescentModel


class StochasticGradientDescentModel(BatchGradientDescentModel):

    def create_model(self, X: pd.DataFrame, y: pd.Series, bias: float):
        w = pd.Series([0] * len(X.columns), index=X.columns)
        self.convergence_of_weights = pd.Series(
            [self.compute_cost(self.X.copy(), self.y.copy(), w)]
        ).reset_index(drop=True)
        w.loc['MODEL_BIAS'] = bias
        i = 0
        for _ in range(self.max_rounds):
            self.row = self.X.sample(1).copy()
            w = self.compute_new_weights(w)
            if self.convergence_of_weights.iloc[-1] < self.convergence_threshold:
                return w
        return w

    def compute_new_weights(self, weights: pd.Series) -> pd.Series:
        new_weights = weights.to_dict()
        for col in weights.keys():
            gradient = self.compute_gradient(pd.Series(new_weights))
            new_weights[col] = new_weights[col] - self.rate * gradient[col]
        new_weights = pd.Series(new_weights)
        new_weights.name = 'weights'
        self.convergence_of_weights = (
            self.convergence_of_weights.append(
                pd.Series([self.compute_norm(new_weights - weights)]), 
                ignore_index=True
            ).reset_index(drop=True)
        )
        self.cost_of_each_step = (
            self.cost_of_each_step.append(
                pd.Series([self.compute_cost(self.X.copy(), self.y.copy(), new_weights)]), 
                ignore_index=True
            ).reset_index(drop=True)
        )
        return new_weights

    def compute_dJ_dw_j(self, weights: pd.Series, col: str) -> float:
        X = self.row.copy()
        y = self.y[X.index].values[0]
        x_i_multiply_w = np.dot(X.to_numpy(), weights.to_numpy())
        error = y - x_i_multiply_w
        return -np.dot(error, X[col].to_numpy())
