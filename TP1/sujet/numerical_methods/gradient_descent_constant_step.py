import numpy as np

from numerical_methods.abstract_gradient_descent import AbstractGradientDescent


class GradientDescentConstantStep(AbstractGradientDescent):
    def __init__(
        self,
        studied_function,
        step_size,
        starting_point,
        max_iteration,
        ceil_norm_grad,
    ):
        super().__init__(
            studied_function,
            step_size,
            starting_point,
            max_iteration,
            ceil_norm_grad,
        )

    def get_next(self, current_point, iteration=None):
        grad = self.studied_function.gradient(current_point).ravel()
        x = current_point - self.step_size * grad
        return x, grad
