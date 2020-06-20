from discreteDynamicProblem import DiscreteDynamicProblem
from discreteDynamicProblem import generatorFromLIst, generatorFromLists
import numpy as np

class DelayProducerPlanning(DiscreteDynamicProblem):
    def __init__(self, numberOfStages:int, initialState:np.array, demand, feasibleDecisions, feasibleStates, 
                finalStateCost, productionCost, currentStateCost, changeStateCost, inf=np.inf, F=dict(), policy=dict()):
        self.__initialState = initialState
        self.__demand = demand
        self.__feasibleDecisions = feasibleDecisions
        self.__feasibleStates = feasibleStates
        self.__finalStateCost = finalStateCost
        self.__productionCost = productionCost
        self.__currentStateCost = currentStateCost
        self.__changeStateCost = changeStateCost
        super().__init__(numberOfStages, inf=inf, F=F, policy=policy)
        
    def state(self, k:int) -> np.array:
        return self.__feasibleStates(k)

    def decision(self, k:int) -> np.array:
        return self.__feasibleDecisions(k)

    def elementaryCost(self, k:int, state:np.array, decision:np.array) -> float:
        return self.__productionCost(k, decision) + self.__currentStateCost(k, state) + self.__changeStateCost(k, state, decision)

    def finalStateCost(self, state:np.array) -> float:
        return self.__finalStateCost(state)

    def transitionFunction(self, k:int, state:np.array, decision:np.array) -> np.array:
        return (state[0] + decision - self.__demand(k), decision)

    def initialState(self) -> np.array:
        return self.__initialState
    

if __name__ == '__main__':
    
    STAGES = 3
    demandStage = lambda k: [2,4,1][k]
    stockCost = lambda k, state: state[0] if state[0]<=4 else np.inf
    changeProductionCost = lambda k, state, decision: max(decision - state[1], state[1] - decision)*2
    productionCost = lambda k, decision: ([3, 5, 3][k])*decision
    stateSpace = lambda k: generatorFromLists([0, 1, 2, 3, 4], [0, 1, 2, 3])
    productionSpace = lambda k: generatorFromLIst([0, 1, 2, 3])

    finalState = lambda state: 0
    initialState = (1, 1)
    
    dp = DelayProducerPlanning(STAGES, initialState, demandStage, productionSpace, stateSpace, finalState, productionCost, stockCost, changeProductionCost)
    dp.solve()
    u_optimal, policy_optimal = dp.optimalTrajectory(initialState)

    print(f"states of the optimal trajectory: {u_optimal}")
    print(f"Decisions for the optimal trajectory: {policy_optimal}")
    print(f"Cost of the optimal trajectory: {dp.F(0, initialState)}")