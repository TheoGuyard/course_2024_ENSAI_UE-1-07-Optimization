from numerical_methods.gradient_descent_constant_step import (
    GradientDescentConstantStep,
)


class GradientDescentVariableStep(GradientDescentConstantStep):
    def __init__(
        self,
        studied_function,
        starting_point,
        max_iteration,
        ceil_norm_grad,
        step_numerator,
        step_denominator,
    ):
        self.studied_function = studied_function
        self.starting_point = starting_point
        self.max_iteration = max_iteration
        self.ceil_norm_grad = ceil_norm_grad
        self.step_numerator = step_numerator
        self.step_denominator = step_denominator

    def get_next(self, current_point, iteration=None):
        grad = self.studied_function.gradient(current_point)
        x = (
            current_point
            - (self.step_numerator / (1 + iteration * self.step_denominator))
            * grad
        )
        return x, grad
