import numpy as np
from function.abstract_function import AbstractFunction


class LeastSquareError(AbstractFunction):
    """Least-squares function f(b)=(1/2m)||y-Xb||_2^2 where m=size(y)."""

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.m = len(y)
        self.H = np.linalg.inv(self.X.T @ self.X / self.m)

    def function_definition(self, b):
        return np.linalg.norm(self.X @ b - self.y, 2) ** 2 / (2 * self.m)

    def gradient(self, b):
        return self.X.T @ (self.X @ b - self.y) / self.m

    def gradient_stochastic(self, b, sample_size):
        idx = np.random.choice(range(self.m), sample_size)
        X_idx = self.X[idx, :]
        y_idx = self.y[idx]
        return X_idx.T @ (X_idx @ b - y_idx) / (self.m * sample_size)

    def inverse_hessian(self, b):
        return self.H
