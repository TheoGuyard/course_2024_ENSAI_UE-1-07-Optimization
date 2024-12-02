from numerical_methods.abstract_gradient_descent import AbstractGradientDescent


class GradientDescentConstantStep(AbstractGradientDescent):

    def get_next(self, x, i):
        g = self.studied_function.gradient(x)
        x = x - self.step_size * g
        return x, g
