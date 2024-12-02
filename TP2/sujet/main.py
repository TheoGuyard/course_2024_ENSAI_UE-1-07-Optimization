import numpy as np
from numerical_methods.gradient_descent_constant_step import GradientDescentConstantStep
from numerical_methods.gradient_descent_variable_step import GradientDescentVariableStep
from numerical_methods.gradient_descent_normalized import GradientDescentNormalized
from numerical_methods.gradient_descent_stochastic import GradientDescentStochastic
from numerical_methods.gradient_descent_momentum import GradientDescentMomentum
from numerical_methods.newton import Newton
from function.least_square_error import LeastSquareError

if __name__ == "__main__":
    
    np.random.seed(0)

    m = 100
    n = 100
    y = np.random.randn(m)
    X = np.random.randn(m, n)
    b = np.zeros(n)
    f = LeastSquareError(X, y)

    gd = Newton(
        studied_function    = f,
        starting_point      = b,
        max_iteration       = 1000,
        ceil_norm_grad      = 1e-10, 
        verbose             = True, 
    )
    gd.descent()
