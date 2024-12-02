from abc import ABC, abstractmethod
import numpy as np


class AbstractGradientDescent(ABC):
    """
    This class contain the basic algorithme for the gradient descent. But the
    get_next() method have to be implemented by the sub classes.
    Abstract class contains the abstract method (= unimplemented method). We
    can easily create a new class witch inherit of AbstractGradientDescent
    and write the missing code.
    Abstract class are useful to create abstract concept, such as a
    AbstractGradientDescent, and create a lot of real class witch inherit of it.
    Because all the child class have identical methode we can switch from one to
    another without any risk of breaking our code (kinding, there is still
    risk, but it is lower than without abstraction)
    """

    def __init__(
        self,
        studied_function,
        step_size,
        starting_point,
        max_iteration,
        ceil_norm_grad,
    ):
        """

        :param studied_function: the function you want to optimize
        :type studied_function: AbstractFunction
        :param step_size: the gradient descent step size
        :type step_size: float
        :param starting_point: where do the descent start
        :type starting_point: array
        :param max_iteration: how many iteration
        :type max_iteration: int
        :param ceil_norm_grad: the minimum size of the gradient. Make the
        descent stop if it as converged
        :type ceil_norm_grad: float
        """
        self.studied_function = studied_function
        self.step_size = step_size
        self.starting_point = starting_point
        self.max_iteration = max_iteration
        self.ceil_norm_grad = ceil_norm_grad

    def descent(self):
        """
        General algorithm for the descent
        """
        # Your code

        # return a dictionnay with all the informations
        return {
            "x_list": np.asarray(x_list),
            "f_list": fx_list,
            "gradient_norm_list": gradient_norm_list,
        }

    @abstractmethod
    def get_next(self, current_point, iteration=None, **kwargs):
        """
        How do I get a new point of the descent ?
        :param current_point: the current point
        :type current_point: np.array
        :param iteration: the current iteration
        :type iteration: int
        :return: the new point and the gradient vector
        :rtype: tuple(np.array,np.array)
        """
        pass
