import numpy as np
import pandas as pd


class BatchGradientDescentModel:

    def __init__(self, X: pd.DataFrame, y: pd.Series, rate=0.1, convergence_threshold=1e-6, max_rounds=100, bias=0):
        X['MODEL_BIAS'] = 1
        self.X = X.copy()
        self.y = y.copy()
        self.max_rounds = max_rounds
        self.rate = rate
        self.convergence_threshold = convergence_threshold
        self.cost_of_each_step = pd.Series()
        self.weights = self.create_model(X.copy(), y.copy(), bias)

    def create_model(self, X: pd.DataFrame, y: pd.Series, bias: float):
        w = pd.Series([0] * len(X.columns), index=X.columns)
        self.cost_of_each_step = pd.Series(
            [self.compute_cost(self.X.copy(), self.y.copy(), w)]
        ).reset_index(drop=True)
        self.convergence_of_weights = pd.Series()
        w.loc['MODEL_BIAS'] = bias
        i = 0
        while i <= self.max_rounds + 1:
            if i == self.max_rounds:
                print('WARNING: Model failed to converge')
                return w
            i += 1
            w = self.compute_new_weights(self.compute_gradient(w), w)
            if self.convergence_of_weights.iloc[-1] < self.convergence_threshold:
                return w
        return w

    def compute_new_weights(self, gradient: pd.Series, weights: pd.Series) -> pd.Series:
        new_weights = weights - self.rate * gradient
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

    def compute_gradient(self, weights: pd.Series) -> pd.Series:
        gradient = pd.Series(index=weights.index, name='gradient')
        for col, _ in gradient.iteritems():
            gradient[col] = self.compute_dJ_dw_j(weights.copy(), col)
        return gradient

    def compute_dJ_dw_j(self, weights: pd.Series, col: str) -> float:
        x_i_multiply_w = np.dot(self.X.to_numpy(), weights.to_numpy())
        error = self.y.to_numpy() - x_i_multiply_w
        return -np.dot(error, self.X[col].to_numpy())

    def compute_cost(self, X, y, weights: pd.Series) -> float:
        X['MODEL_BIAS'] = weights['MODEL_BIAS']
        x_i_multiply_w = np.dot(X.to_numpy(), weights.to_numpy())
        error = np.square(y.to_numpy() - x_i_multiply_w)
        return np.sum(0.5 * np.square(error))

    def compute_mean_error(self, weights: pd.Series) -> np.ndarray:
        e = self.compute_point_error(weights)
        return np.abs(np.mean(e))

    # TODO: Implement in other functions?
    def compute_point_error(self, weights: pd.Series) -> np.ndarray:
        x_i_multiply_w = np.dot(self.X.to_numpy(), weights.to_numpy())
        return self.y.to_numpy() - x_i_multiply_w

    @staticmethod
    def compute_norm(x) -> float:
        return np.linalg.norm(x)
