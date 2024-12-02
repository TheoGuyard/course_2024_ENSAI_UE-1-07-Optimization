import numpy as np
from numerical_methods.abstract_gradient_descent import AbstractGradientDescent


class GradientDescentNormalized(AbstractGradientDescent):

    def get_next(self, x, i):
        g = self.studied_function.gradient(x)
        x = x - (self.step_size / np.linalg.norm(g, 2)) * g
        return x, g
