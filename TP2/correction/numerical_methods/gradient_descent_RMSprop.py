from numerical_methods.abstract_gradient_descent import AbstractGradientDescent
from numpy import sqrt


class GradientDescentRMSProp(AbstractGradientDescent):

    def __init__(self, gamma, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma  # momentum coefficient
        self.e = 1e-10      # denominator constant

    def get_next(self, x, i):
        # Re-initialize momentum sum at first iteration
        if i == 0.:
            self.s = 0.
        g = self.studied_function.gradient(x)
        self.s = self.gamma * self.s + (1 - self.gamma) * (g**2)
        x = x - self.step_size * g / (self.e + sqrt(self.s))
        return x, g