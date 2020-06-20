from stochasticDiscreteDynamicProblem import StochasticDiscreteDynamicProblem, generatorFromLIst
from randomVariable import RandomVariable
import numpy as np

class SimpleStochasticDiscreteDynamicProblem(StochasticDiscreteDynamicProblem):
    def __init__(self, numberOfStages:int, feasibleStates, feasibleDecisions, productionCost, stockCost, initialState, 
                realizableRandomValues, finalStateCost, inf=100, F=dict(), policy=dict()):
        self.__feasibleStates = feasibleStates
        self.__feasibleDecisions = feasibleDecisions
        self.__productionCost = productionCost
        self.__stockCost = stockCost
        self.__initialState = initialState
        self.__realizableRandomValues = realizableRandomValues
        self.__finalStateCost = finalStateCost
        super().__init__(numberOfStages, inf, F, policy)

    def state(self, k:int) -> np.array:
        return self.__feasibleStates(k)

    def decision(self, k:int) -> np.array:
        return self.__feasibleDecisions(k)

    def elementaryCost(self, k:int, state:np.array, decision:np.array, random_variable:np.array) -> float:
        return self.__productionCost(k, decision) + self.__stockCost(k, state)

    def finalStateCost(self, state:np.array) -> float:
        return self.__finalStateCost(state)

    def transitionFunction(self, k:int, state:np.array, decision:np.array, random_variable:np.array) -> np.array:
        return state + decision - random_variable

    def initialState(self) -> np.array:
        return self.__initialState

    def realizableRandomValues(self, k:int) -> RandomVariable:
        return self.__realizableRandomValues(k)

if __name__ == '__main__':
    STAGES = 3

    # this cost works with the case that we cant meet the demand
    def stockCost(k, state):
        if state<0:
            return (-5*state)
        if state>4:
            return (state-4)*2 + 4
        else:
            return state

    productionCost = lambda k, decision: ([3, 5, 3][k])*decision
    
    stateSpace = lambda k: generatorFromLIst([0, 1, 2, 3, 4])
    productionSpace = lambda k: generatorFromLIst([0, 1, 2, 3])
    demandRandomValues = lambda k: ([
                                    RandomVariable((1,2,3), (0.2, 0.7, 0.1)),
                                    RandomVariable((3, 4, 5), (0.3, 0.6, 0.1)),
                                    RandomVariable((1, 2, 3), (0.5, 0.4, 0.1))
                                        ][k])

    finalStateCost = lambda state: 0
    initialState = 1

    a = SimpleStochasticDiscreteDynamicProblem(STAGES, stateSpace, productionSpace, productionCost, stockCost, initialState, demandRandomValues, finalStateCost)
    a.solve()

    print(f"All policy: {a.allPolicy()}")
    print(f"Expectancy of cost: {a.F(0, initialState)}")