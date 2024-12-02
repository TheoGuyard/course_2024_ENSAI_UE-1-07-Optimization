from function.abstract_function import AbstractFunction


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
        # -> YOUR CODE

    def gradient(self, point):
        # -> YOUR CODE
        pass
