import numpy as np
from function.abstract_function import AbstractFunction


class LeastSquareError(AbstractFunction):
    """Least-squares function f(b)=(1/2m)||y-Xb||_2^2 where m=len(y)."""

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.m = y.size
        self.H = np.linalg.inv(X.T @ X / self.m)

    def function_definition(self, b):
        return np.linalg.norm(self.y - self.X @ b, 2) ** 2 / (2. * self.m)

    def gradient(self, b):
        return self.X.T @ (self.X @ b - self.y) / self.m

    def gradient_stochastic(self, b, sample_size):
        I = np.random.choice(range(self.m), sample_size)
        X_I = self.X[I, :]
        y_I = self.y[I]
        return X_I.T @ (X_I @ b - y_I) / (self.m * sample_size)

    def inverse_hessian(self, b):
        return self.H