from numerical_methods.abstract_gradient_descent import AbstractGradientDescent
import numpy as np


class GradientDescentMomentum(AbstractGradientDescent):
    def __init__(self, gamma, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma  # momentum coefficient
        self.s = 0.         # momentum sum

    def get_next(self, x, i):
        # Re-initialize momentum sum at first iteration
        if i == 0.:
            self.s = 0.
        g = self.studied_function.gradient(x)
        self.s = self.gamma * self.s + (1. - self.gamma) * g
        x = x - self.step_size * self.s
        return x, g
