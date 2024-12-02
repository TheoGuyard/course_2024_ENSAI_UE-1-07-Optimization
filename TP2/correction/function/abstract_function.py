from abc import ABC, abstractmethod


class AbstractFunction(ABC):

    @abstractmethod
    def function_definition(self, x):
        ...

    @abstractmethod
    def gradient(self, x):
        ...

    @abstractmethod
    def inverse_hessian(self, x):
        ...
