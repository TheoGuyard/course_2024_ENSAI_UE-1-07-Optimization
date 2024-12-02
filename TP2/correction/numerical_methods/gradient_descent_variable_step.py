from numerical_methods.abstract_gradient_descent import AbstractGradientDescent


class GradientDescentVariableStep(AbstractGradientDescent):
    def __init__(self, step_numer, step_denom, **kwargs):
        super().__init__(step_size=None, **kwargs)
        self.step_numer = step_numer
        self.step_denom = step_denom

    def get_next(self, x, i):
        g = self.studied_function.gradient(x)
        x = x - (self.step_numer / (1. + i * self.step_denom)) * g
        return x, g
