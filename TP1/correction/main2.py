from numpy.random.mtrand import choice

from function.least_square_error import LeastSquareError
from numerical_methods.gradient_descent_constant_step import (
    GradientDescentConstantStep,
)

from helper import graph_3d

if __name__ == "__main__":
    least_square_error = LeastSquareError()
    graph_3d.display_3d_surface(least_square_error, -5, 5, -5, 5, 0.2)
    x0 = [1, 1]
    gradient_descent = GradientDescentConstantStep(
        studied_function=least_square_error,
        step_size=10**-3,
        starting_point=x0,
        max_iteration=1000,
        ceil_norm_grad=10**-10,
    )
    result = gradient_descent.descent()
    graph_3d.display_gradient_descent(
        least_square_error, result["x_list"], -5, 5, -5, 5, 0.4
    )
