import numpy as np
from numerical_methods.gradient_descent_RMSprop import GradientDescentRMSProp
from numerical_methods.gradient_descent_constant_step import GradientDescentConstantStep
from numerical_methods.gradient_descent_momentum import GradientDescentMomentum
from numerical_methods.gradient_descent_variable_step import GradientDescentVariableStep
from numerical_methods.gradient_descent_normalized import GradientDescentNormalized
from numerical_methods.gradient_descent_sochastic import GradientDescentStochastic
from numerical_methods.newton_method import NewtonMethod
from function.least_square_error import LeastSquareError
from helper import graph_3d

if __name__ == "__main__":

    # Function
    np.random.seed(0)
    m = 100
    n = 100
    y = np.random.randn(m)
    X = np.random.randn(m, n)
    func = LeastSquareError(X, y)

    # Parameters
    x0 = np.zeros(n)
    step_size = 1e-3
    max_iteration = 10000
    ceil_norm_grad = 10**-10
    sample_size = int(np.round(m / 5))
    verbose = True
    gamma = 0.6

    ###########################################################
    # Display the function
    ###########################################################
    if False:
        graph_3d.display_3d_surface(func)

    ###########################################################
    # Gradient descent constant step
    ###########################################################
    if False:
        print("GradientDescentConstantStep")
        gradient_descent = GradientDescentConstantStep(
            studied_function=func,
            step_size=step_size,
            starting_point=x0,
            max_iteration=max_iteration,
            ceil_norm_grad=ceil_norm_grad,
            verbose=verbose,
        )
        result = gradient_descent.descent()
        # graph_3d.display_gradient_descent(func, result["x_list"])

    ###########################################################
    # Gradient descent variable step
    ###########################################################
    if False:
        print("GradientDescentVariableStep")
        gradient_descent = GradientDescentVariableStep(
            studied_function=func,
            starting_point=x0,
            max_iteration=max_iteration,
            ceil_norm_grad=ceil_norm_grad,
            step_numer=step_size,
            step_denom=0.001 * step_size,
            verbose=verbose,
        )
        result = gradient_descent.descent()
        # graph_3d.display_gradient_descent(func, result["x_list"])

    ###########################################################
    # Gradient descent normalized step
    ###########################################################
    if False:
        print("GradientDescentNormalized")
        gradient_descent = GradientDescentNormalized(
            studied_function=func,
            starting_point=x0,
            step_size=step_size,
            max_iteration=max_iteration,
            ceil_norm_grad=ceil_norm_grad,
            verbose=verbose,
        )
        result = gradient_descent.descent()
        # graph_3d.display_gradient_descent(func, result["x_list"])

    ###########################################################
    # Gradient descent stochastic
    ###########################################################
    if False:
        print("GradientDescentStochastic")
        gradient_descent = GradientDescentStochastic(
            studied_function=func,
            starting_point=x0,
            step_size=step_size,
            max_iteration=max_iteration,
            ceil_norm_grad=ceil_norm_grad,
            verbose=verbose,
            sample_size=sample_size,
        )
        result = gradient_descent.descent()
        # graph_3d.display_gradient_descent(func, result["x_list"])

    ###########################################################
    # Gradient descent with momentum
    ###########################################################
    if False:
        print("GradientDescentMomentum")
        gradient_descent = GradientDescentMomentum(
            studied_function=func,
            starting_point=x0,
            step_size=step_size,
            max_iteration=max_iteration,
            ceil_norm_grad=ceil_norm_grad,
            verbose=verbose,
            gamma=gamma,
        )
        result = gradient_descent.descent()
        # graph_3d.display_gradient_descent(func, result["x_list"])

    ###########################################################
    # Gradient descent RMSProp
    ###########################################################
    if False:
        print("GradientDescentRMSProp")
        gradient_descent = GradientDescentRMSProp(
            studied_function=func,
            starting_point=x0,
            step_size=step_size,
            max_iteration=max_iteration,
            ceil_norm_grad=ceil_norm_grad,
            verbose=verbose,
            gamma=gamma,
        )
        result = gradient_descent.descent()
        # graph_3d.display_gradient_descent(func, result["x_list"])

    ###########################################################
    # Newton's method
    ###########################################################
    if True:
        print("NewtonMethod")
        gradient_descent = NewtonMethod(
            studied_function=func,
            starting_point=x0,
            max_iteration=max_iteration,
            ceil_norm_grad=ceil_norm_grad,
            verbose=verbose,
        )
        result = gradient_descent.descent()
        # graph_3d.display_gradient_descent(func, result["x_list"])