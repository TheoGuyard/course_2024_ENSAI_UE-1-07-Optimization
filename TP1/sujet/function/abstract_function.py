from abc import ABC, abstractmethod


class AbstractFunction(ABC):

    @abstractmethod
    def function_definition(self, point):
        """
        How to compute the function
        :param point: the point coordinate in R^n
        :type point: np.array
        :return: the function result
        :rtype: float
        """

    @abstractmethod
    def gradient(self, point):
        """
        How to compute the gradient
        :param point: the point coordinate in R^n
        :type point: np.array
        :return: the gradient
        :rtype: np.array
        """
