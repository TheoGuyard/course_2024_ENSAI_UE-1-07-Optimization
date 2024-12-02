from function.abstract_function import AbstractFunction
import numpy as np


class Rosenbrock(AbstractFunction):
    def function_definition(self, point):
        """
        Retourne la valeur de x=(x1, x2) par la fonction de rosenbrock :
        f(x1,x2) = (x1-1)**2 + 100*(x1**2-x2)**2

        Args:
            x (list or array): la coordonnée du point

        Returns :
            float : la valeur numérique de rosenbrock(x)

        """
        return (point[0] - 1) ** 2 + 100 * (point[0] ** 2 - point[1]) ** 2

    def gradient(self, point):
        grad = np.zeros(len(point))
        grad[0] = 400 * point[0] * (point[0] ** 2 - point[1]) + 2 * (
            point[0] - 1
        )
        grad[1] = 200 * (point[1] - point[0] ** 2)
        return grad
