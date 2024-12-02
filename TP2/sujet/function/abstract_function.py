from abc import ABC, abstractmethod


class AbstractFunction(ABC):

    @abstractmethod
    def function_definition(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass

    @abstractmethod
    def inverse_hessian(self, x):
        pass
