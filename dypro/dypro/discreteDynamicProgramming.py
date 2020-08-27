import abc
import numpy as np


class DiscreteDynamicProgramming(metaclass=abc.ABCMeta):
    
    def __init__(self, numberOfStages:int, inf=np.inf):
        self.numberOfStages = numberOfStages
        self.inf = inf
    
    @abc.abstractmethod
    def state(self, k:int) -> np.array:
        """
        Generates all the feasible states (x_k in X_k) of a given stage, this should be a generator

        Args:
            k (int): stage

        Returns:
            np.array: generator of feasible states
        """
        pass

    @abc.abstractmethod
    def decision(self, k:int) -> np.array:
        """
        Generates all the feasible decisions (u_k in U_k) of a stage

        Args:
            k (int): stage

        Returns:
            np.array: generator of feasibles decisions
        """
        pass

    @abc.abstractmethod
    def elementaryCost(self, k:int, state:np.array, decision:np.array) -> float:
        """
        Calculate the elementary cost of current state on stage k and making a decision

        Args:
            k (int): current stage
            state (np.array): current state
            decision (np.array): decision made
        
        Returns:
            float: cost associated with state x_k
        """
        pass

    @abc.abstractmethod
    def finalStateCost(self, state:np.array) -> float:
        """
        Calculate the cost associated with the final state x_n

        Args:
            state (np.array): feasible state of stage n

        Returns:
            float: cost associated with the final state x_n
        """
        pass

    @abc.abstractmethod
    def transitionFunction(self, k:int, state:np.array, decision:np.array) -> np.array:
        """
        Given the current stage, state and decision calculates the state of next stage

        Args:
            k (int): stage
            state (np.array): feasible state of stage k
            decision (np.array): feasible decision on stage k

        Returns:
            np.array: state of stage k + 1
        """
        pass

    @abc.abstractmethod
    def initialState(self) -> np.array:
        """
        Generates the initial state, this could be 1 or more states

        Returns:
            np.array: initial state
        """
        pass