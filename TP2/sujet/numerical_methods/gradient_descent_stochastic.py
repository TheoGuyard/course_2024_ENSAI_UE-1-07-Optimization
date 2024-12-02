from numerical_methods.abstract_gradient_descent import AbstractGradientDescent


class GradientDescentStochastic(AbstractGradientDescent):

    def __init__(self, sample_size, **kwargs):
        super().__init__(**kwargs)
        self.sample_size = sample_size

    def get_next(self, x, i):
        g = self.studied_function.gradient_stochastic(x, self.sample_size)
        x = x - self.step_size * g
        return x, g
