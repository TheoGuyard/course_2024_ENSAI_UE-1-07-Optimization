import numpy as np
from numerical_methods.abstract_gradient_descent import AbstractGradientDescent


class GradientDescentMomentum(AbstractGradientDescent):

    def __init__(self, gamma, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def get_next(self, x, i):
        if i == 0:
            self.s = np.zeros(x.size)
        g       = self.studied_function.gradient(x)
        self.s  = self.gamma * self.s + (1. - self.gamma) * g
        x       = x - self.step_size * self.s
        return x, g