from numerical_methods.abstract_gradient_descent import AbstractGradientDescent


class NewtonMethod(AbstractGradientDescent):
    def __init__(self, **kwargs):
        super().__init__(step_size=None, **kwargs)

    def get_next(self, x, i):
        g = self.studied_function.gradient(x)
        H = self.studied_function.inverse_hessian(x)
        x = x - H @ g
        return x, g
