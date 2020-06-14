import abc
import logging
import numpy as np

logging.basicConfig(level=logging.ERROR)

class DiscreteDynamicProblem(metaclass=abc.ABCMeta):
    
    def __init__(self, numberOfStages:int, inf=np.inf, F=dict(), policy=dict()):
        self.__numberOfStages = numberOfStages
        self.__inf = inf
        self.__F = F # accumulated optimal cost
        self.__policy = policy # best decision on stage k, state u_k

    def inf(self):
        """
        Return the used infinite value
        """
        return self.__inf

    def F(self, k:int, state:np.array) -> float:
        """
        Returns the accumulated optimal cost of stage k and state state
        Args:
            k (int): stage
            state (np.array): current state

        Returns:
            float: the accumulated optimal cost
        """
        if (k, state) in self.__F:
            return self.__F[k, state]
        else:
            self.__F[k, state] = self.inf()
            return self.inf()
    
    def accOptimalCost(self) -> dict:
        """
        Returns the accumulated optimal cost for all states and stages
        """
        return self.__F
    
    def numberOfStages(self) -> int:
        """
        Return the number of stages
        """
        return self.__numberOfStages
    
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
    def transictionFunction(self, k:int, state:np.array, decision:np.array) -> np.array:
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

    def solve(self):
        """
        Solves the optimality recursive equation using a backward strategy
        """
        for xk in self.state(self.numberOfStages()):
            self.__F[self.numberOfStages(), xk] = self.finalStateCost(xk)
        
        for k in self.__stagesGenerator(): #n-1 to 0
            for xk in self.state(k):
                F_aux = self.inf()
                u_aux = None
                # Calculates the best decision on stage k and state uk
                for uk in self.decisionsSpace(k):
                    xk_next = self.transitionFunction(xk, uk, k)
                    F_aux_uk = self.elementaryCost(xk, uk, k) + self.F(k+1, xk_next)

                    if F_aux > F_aux_uk:
                        F_aux = F_aux_uk
                        u_aux = uk
                self.__F[k, xk] = F_aux
                self.__policy[k, xk] = u_aux

    def __stagesGenerator(self):
        """
        Generates integers from numberOfStages-1 to 0
        Yields:
            int: stage
        """
        stagesList= range(self.numberOfStages()-1, -1, -1)
        for stage in stagesList:
            yield stage


if __name__ == '__main__':
    dp = DiscreteDynamicProblem(3)

    for i in dp.stagesGenerator():
        print(i)