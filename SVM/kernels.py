import numpy as np


# The best way to implement this would be with an abstract base class to enforce
# that all Kernels require a calc_kernel method

class LinearKernel:

    def calc_kernel(self, X1: np.ndarray, X2: np.ndarray):
        return np.dot(X1, X2.T)  # Gives a matrix of dot products between all x_i's


class GaussianKernel:

    def __init__(self, gamma: float):
        self.gamma = gamma  # A hyperparamter determining Gaussian scaling

    def calc_kernel(self, X1: np.ndarray, X2: np.ndarray):
        # k(x_i, x_j) = exp(-||x_i - x_j||^2 / gamma)
        x_i_j_diff = self.create_diff_permutations_for_Xs(X1, X2)
        norm_x_i_j_diff = np.linalg.norm(x_i_j_diff, axis=1)
        sq_norm_x_i_j_diff = np.square(norm_x_i_j_diff)
        k = np.exp(-1 * sq_norm_x_i_j_diff / self.gamma)
        return k

    def create_diff_permutations_for_Xs(self, X1: np.ndarray, X2: np.ndarray):
        # Creates a matrix with permutations of differences between each X sample from
        # X1 and X2
        
        # Reshape X1 and X2 so they can be broadcast into a 3d matrix of dim
        # (n, d, m) where n is len(X1), m is len(X2), and d is the number of 
        # columns in each individual sample
        X1 = X1.reshape(X1.shape[0], X1.shape[1], 1)
        X2 = X2.reshape(X2.shape[0], X2.shape[1], 1)
        X1_X2 = np.broadcast(X1, X2.T)
        x_i_j_diff = np.empty(X1_X2.shape)
        x_i_j_diff.flat = [x1_i - x2_j for x1_i, x2_j in X1_X2]
        return x_i_j_diff
