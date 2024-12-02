import time
import numpy as np
from abc import ABC, abstractmethod


class AbstractGradientDescent(ABC):

    def __init__(
        self,
        studied_function,
        step_size,
        starting_point,
        max_iteration,
        ceil_norm_grad,
        verbose,
    ):
        self.studied_function = studied_function
        self.step_size = step_size
        self.starting_point = starting_point
        self.max_iteration = max_iteration
        self.ceil_norm_grad = ceil_norm_grad
        self.verbose = verbose

    def descent(self):
        x_list = []
        fx_list = []
        gx_list = []
        tx_list = []

        x = np.array(self.starting_point)
        t = time.time()
        fx = np.nan
        gx = np.nan
        tx = np.nan

        for i in range(self.max_iteration):

            # Compute the next point, also returns the gradient
            x, g = self.get_next(x, i)
            fx = self.studied_function.function_definition(x)
            gx = np.linalg.norm(g, 2)
            tx = time.time() - t

            # Append new values to the lists
            x_list.append(x)
            fx_list.append(fx)
            gx_list.append(gx)
            tx_list.append(tx)

            # Displays
            if self.verbose and i % 100 == 0:
                print("iter={:<4d}  t(x)={:.4f}  f(x)={:.2e}  g(x)={:.2e}".format(i, tx, fx, gx))

            # Check stopping criterion
            if gx < self.ceil_norm_grad:
                break

        # Displays at end of algorithm
        if self.verbose:
            print("iter={:<4d}  t(x)={:.4f}  f(x)={:.2e}  g(x)={:.2e}".format(i, tx, fx, gx))

        return {
            "x_list": np.array(x_list),
            "fx_list": fx_list,
            "gx_list": gx_list,
            "tx_list": tx_list,
        }

    @abstractmethod
    def get_next(self, x, i):
        pass
