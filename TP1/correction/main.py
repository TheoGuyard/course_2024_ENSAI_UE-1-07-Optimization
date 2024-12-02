from numerical_methods.gradient_descent_constant_step import (
    GradientDescentConstantStep,
)
from numerical_methods.gradient_descent_variable_step import (
    GradientDescentVariableStep,
)
from numerical_methods.gradient_descent_normalized import (
    GradientDescentNormalized,
)

from helper import graph_3d
from function.rosenbrock import Rosenbrock

if __name__ == "__main__":
    x0 = [-1, -1]
    rosenbrock = Rosenbrock()

    # Display the function
    if False:
        # graph_3d.contour(rosenbrock, -1.1, 2.2, -1.1, 2.5)
        graph_3d.display_3d_surface(rosenbrock, -1.1, 2.2, -1.1, 2.5, 0.1)

    ###########################################################
    # Gradient descent constant step
    ###########################################################
    if True:
        gradient_descent = GradientDescentConstantStep(
            studied_function=rosenbrock,
            step_size=10**-3,
            starting_point=x0,
            max_iteration=10000,
            ceil_norm_grad=10**-10,
        )
        result = gradient_descent.descent()
        graph_3d.display_gradient_descent(rosenbrock, result["x_list"])

    ###########################################################
    # Gradient descent variable step
    ###########################################################
    if False:
        gradient_descent = GradientDescentVariableStep(
            studied_function=rosenbrock,
            starting_point=x0,
            max_iteration=10000,
            ceil_norm_grad=10**-10,
            step_numerator=10**-3,
            step_denominator=10**-3,
        )
        result = gradient_descent.descent()
        graph_3d.display_gradient_descent(rosenbrock, result["x_list"])

    ###########################################################
    # Gradient descent normalized step
    ###########################################################
    if True:
        gradient_descent = GradientDescentNormalized(
            studied_function=rosenbrock,
            starting_point=x0,
            step_size=10**-3,
            max_iteration=10000,
            ceil_norm_grad=10**-10,
        )
        result = gradient_descent.descent()
        graph_3d.display_gradient_descent(
            rosenbrock, x_max=2, y_max=4, coord_descent=result["x_list"]
        )
        graph_3d.contour_and_gradient(
            rosenbrock, x_max=2, y_max=4, coord_descent=result["x_list"]
        )

        # gradient_descent = GradientDescentNormalized(
        #     studied_function=rosenbrock,
        #     starting_point=x0,
        #     step_size=10**-1,
        #     max_iteration=100,
        #     ceil_norm_grad=10**-10,
        # )
        # result = gradient_descent.descent()
        # graph_3d.contour_and_gradient(
        #     rosenbrock, x_max=2, y_max=4, coord_descent=result["x_list"]
        # )
