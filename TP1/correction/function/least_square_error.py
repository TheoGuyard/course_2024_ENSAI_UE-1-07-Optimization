import numpy as np
from numpy.random.mtrand import choice
from function.abstract_function import AbstractFunction
from helper.reg_data import get_data


class LeastSquareError(AbstractFunction):
    def function_definition(self, point):
        # Need to change the shape of beta. Beta is a line vector, but we need and column vector
        beta_array = np.asarray(point).reshape(2, 1)
        X, y = get_data()
        result = np.mean(np.square((X * beta_array - y)))
        return result

    def gradient(self, point):
        beta_array = np.asarray(point).reshape(2, 1)
        X, y = get_data()
        m = len(X)
        Xt_Xb = X.T * X * beta_array
        Xy = X.T * y
        result = 2 / m * (Xt_Xb - Xy)
        return result.A1
