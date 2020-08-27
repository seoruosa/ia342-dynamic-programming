import abc
from dypro.dypro.randomVariable import RandomVariable


class StochasticProblem(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def realizableRandomValues(self, k:int) -> RandomVariable:
        """
        Generates all realizable values for each stage

        Args:
            k (int): stage

        Output:
            np.array: generator of realizable variables
        """
        pass

