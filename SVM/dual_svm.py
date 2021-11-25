import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult

from .kernels import LinearKernel

linear_kernel = LinearKernel()


class DualSvmModel:

    def __init__(self, X: pd.DataFrame, y: pd.Series, hyper_c=0, kernel=linear_kernel):
        self.X = X.copy()
        self.y = y.copy()
        self.C = hyper_c
        self.kernel_cls = kernel
        self.gram = self.kernel_cls.calc_kernel(self.X.values, self.X.values)
        self.create_model()
        del self.X
        del self.y

    def create_model(self):
        alpha_optimization = self.calc_lagrangian_multipliers(self.X.values, self.y.values, self.gram)
        init_alpha = alpha_optimization.x
        self.sv_ix = self.determine_support_vectors(init_alpha)
        self.sv_X = self.X.values[self.sv_ix]
        self.sv_y = self.y.values[self.sv_ix]
        self.alpha = init_alpha[self.sv_ix]
        self.bias = self.calc_bias()

    def calc_lagrangian_multipliers(self, X: np.ndarray, y: np.ndarray, 
            gram: np.ndarray) -> OptimizeResult:

        def objective(alphas, X_input, y_input):
            # equation: 1/2 * sum_i ( sum_j ( y_i y_j a_i a_j K(x_i, x_j) ) ) - sum_i (a_i)
            y_dot_y = np.outer(y_input, y_input)  # Gives a matrix of dot products between all y_i's
            a_dot_a = np.outer(alphas, alphas)  # Gives a matrix of dot products between all alpha_i's
            to_sum = np.multiply(gram, np.multiply(y_dot_y, a_dot_a))
            return 0.5 * np.sum(np.sum(to_sum)) - np.sum(alphas)

        def constraint(alphas, _, y_input):
            return np.dot(alphas, y_input)

        args = (X, y)
        x0 = np.array([0] * len(y))  # Will guess 0 as default for all alphas
        bnds = [(0, self.C)] * len(y)  # Know 0 <= alpha <= C
        cons = ({'type': 'eq', 'fun': constraint, 'args': args})
        print('MINIMIZING, THIS MAY TAKE A WHILE . . .')
        return minimize(
            fun=objective,
            x0=x0,
            method='SLSQP',  # Will use at TM's suggestion
            bounds=bnds,
            constraints=cons,
            args=args,
            callback=lambda x: print('NEW MINIMIZE ITERATION\nALPHA MIN', min(x), '\nALPHA MAX', max(x), '\nALPHA MEDIAN', np.median(x))
        )

    def determine_support_vectors(self, alpha: np.ndarray) -> np.ndarray:
        # Support vectors come from indices where lagrange multiplier is > 0
        return np.where(alpha > 0)

    def calc_weight_vector(self) -> np.ndarray:
        # TODO: Cannot handle kernels
        return np.dot(self.sv_X.T, np.multiply(self.alpha, self.sv_y))

    def calc_bias(self) -> float:
        k = self.kernel_cls.calc_kernel(self.sv_X, self.sv_X)
        return np.mean(self.sv_y - np.dot(np.multiply(self.alpha, self.sv_y), k))

    def test(self, X: pd.DataFrame, y: pd.Series) -> float:
        y_hat = self.evaluate(X)
        s = y.to_numpy() == y_hat
        return np.sum(s) / len(s)

    def evaluate(self, X: pd.DataFrame) -> np.ndarray:
        k = self.kernel_cls.calc_kernel(self.sv_X, X.to_numpy())
        return np.sign(np.dot(np.multiply(self.alpha, self.sv_y), k) + self.bias)
