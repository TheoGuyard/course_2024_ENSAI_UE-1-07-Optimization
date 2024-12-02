import numpy as np
from function.abstract_function import AbstractFunction


class Rosenbrock(AbstractFunction):
    """Rosenbrock function f(x1,x2) = (x1-1)**2 + 100*(x1**2-x2)**2."""

    def function_definition(self, point):
        return (point[0] - 1.) ** 2 + 100. * (point[0] ** 2. - point[1]) ** 2

    def gradient(self, x):
        g = np.zeros(2)
        g[0] = 400. * x[0] * (x[0] ** 2 - x[1]) + 2. * (x[0] - 1.)
        g[1] = 200. * (x[1] - x[0] ** 2)
        return g

    def inverse_hessian(self, x):
        H = np.zeros((2, 2))
        H[0, 0] = 2. + 400. * (x[0] ** 2 - x[1]) + 800. * x[0] ** 2
        H[1, 0] = -400. * x[0]
        H[0, 1] = H[1, 0]
        H[1, 1] = 200.
        return np.linalg.inv(H)
