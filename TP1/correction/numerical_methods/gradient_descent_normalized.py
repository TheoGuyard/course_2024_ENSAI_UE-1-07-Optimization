from numerical_methods.gradient_descent_constant_step import (
    GradientDescentConstantStep,
)
import numpy as np


class GradientDescentNormalized(GradientDescentConstantStep):
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
        grad = self.studied_function.gradient(current_point)
        x = (
            current_point 
            - self.step_size / np.linalg.norm(grad)
            * grad
        )
        return x, grad
