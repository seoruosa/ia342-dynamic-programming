import abc
import logging
import numpy as np

logging.basicConfig(level=logging.ERROR)

class StochasticDiscreteDynamicProblem(metaclass=abc.ABCMeta):
    
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
    
    def policy(self, k:int, state:np.array) -> np.array:
        """
        Return the best policy of stage k and given state

        Returns:
            np.array: decision
        """
        return self.__policy[k, state]
    
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
    def elementaryCost(self, k:int, state:np.array, decision:np.array, random_variable:np.array) -> float:
        """
        Calculate the elementary cost of current state on stage k and making a decision

        Args:
            k (int): current stage
            state (np.array): current state
            decision (np.array): decision made
            random_variable(np.array): random variable that is associated with a probability
        
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
    def transitionFunction(self, k:int, state:np.array, decision:np.array, random_variable:np.array) -> np.array:
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

    @abc.abstractmethod
    def probability(self, k:int, random_variable:np.array) -> float:
        """
        Returns a probability for random variable

        Args:
            k (int): [description]
            random_variable (np.array): [description]

        Returns:
            float: [description]
        """
        pass

    @abc.abstractmethod
    def realizableRandomValues(self, k:int) -> np.array:
        """
        Generates all realizable values for each stage

        Args:
            k (int): stage

        Yields:
            np.array: generator of realizable variables
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
                for uk in self.decision(k):
                    # Expected value of accumulated cost for decision uk and state xk
                    E_F_aux = 0 
                    for wk in self.realizableRandomValues(k):
                        xk_next = self.transitionFunction(k, xk, uk, wk)
                        E_F_aux = self.probability(k, wk) * (self.elementaryCost(k, xk, uk, wk) + self.F(k+1, xk_next)) + E_F_aux

                    if F_aux > E_F_aux:
                        F_aux = E_F_aux
                        u_aux = uk
                self.__F[k, xk] = F_aux
                self.__policy[k, xk] = u_aux
    
    # def optimalTrajectory(self, initialState:np.array):
    #     """
    #     Calculates the optimal trajectory given a initial state

    #     Args:
    #         initialState (np.array): initial state to calculate the optimal trajectory

    #     Returns:
    #         u_optimal: states of the optimal trajectory
    #         policy_optimal: decisions for the optimal trajectory
    #     """
    #     u_optimal = [initialState]
    #     policy_optimal = list()
    #     for k in range(self.numberOfStages()):
    #         policy_optimal.append(self.policy(k, u_optimal[-1]))
    #         u_optimal.append(self.transitionFunction(k, u_optimal[-1], policy_optimal[-1]))
        
    #     return (u_optimal, policy_optimal, )

    def __stagesGenerator(self):
        """
        Generates integers from numberOfStages-1 to 0
        Yields:
            int: stage
        """
        stagesList= range(self.numberOfStages()-1, -1, -1)
        for stage in stagesList:
            yield stage
        
class RandomVariable():
    def __init__(self, value:tuple, probability:float):
            self.value = value
            self.probability = probability
            if(probability>1 or probability<0):
                raise TypeError
    
    def getValue(self):
        return self.value
    
    def getProbability(self):
        return self.probability
    
    def __str__(self):
        return f"{self.getProbability()} {self.getValue()}"

def generatorFromLIst(itens:list):
    """
    Return a generator from the list
    """
    for i in itens:
        yield i

def generatorFromLists(a:list, b:list):
    """
    Return a generator of tuples of lists a and b
    """
    for i in a:
        for j in b:
            yield (i, j)

if __name__ == '__main__':
    for i in generatorFromLIst(list(range(5))):
        print(i)