from dypro.dypro.discreteDynamicProblem import DiscreteDynamicProblem
from dypro.dypro.discreteDynamicProblem import generatorFromLIst
import numpy as np

class SimpleProductionProblem(DiscreteDynamicProblem):
    def __init__(self, numberOfStages:int, initialState:np.array, demand, feasibleDecisions, feasibleStates, 
                finalStateCost, productionCost, currentStateCost, inf=np.inf, F=dict(), policy=dict()):
        self.__initialState = initialState
        self.__demand = demand
        self.__feasibleDecisions = feasibleDecisions
        self.__feasibleStates = feasibleStates
        self.__finalStateCost = finalStateCost
        self.__productionCost = productionCost
        self.__currentStateCost = currentStateCost
        super().__init__(numberOfStages, inf=inf, F=F, policy=policy)
        
    def state(self, k:int) -> np.array:
        return self.__feasibleStates(k)

    def decision(self, k:int) -> np.array:
        return self.__feasibleDecisions(k)

    def elementaryCost(self, k:int, state:np.array, decision:np.array) -> float:
        return self.__productionCost(k, decision) + self.__currentStateCost(k, state)

    def finalStateCost(self, state:np.array) -> float:
        return self.__finalStateCost(state)

    def transitionFunction(self, k:int, state:np.array, decision:np.array) -> np.array:
        return state + decision - self.__demand(k)

    def initialState(self) -> np.array:
        return self.__initialState
    

if __name__ == '__main__':
    
    STAGES = 4
    demandStage = lambda k: [2,1,1,0][k]
    stockCost = lambda k, state: [0, 3, 7][state] if state<3 else np.inf
    productionCost = lambda k, decision: [10, 17, 20][decision]
    stateSpace = lambda k: generatorFromLIst([0, 1, 2])
    productionSpace = lambda k: generatorFromLIst([0, 1, 2])
    finalState = lambda state: 0 if state==1 else np.inf
    initialStock = 1
    
    dp = SimpleProductionProblem(STAGES, initialStock, demandStage, productionSpace, stateSpace, finalState, productionCost, stockCost)
    dp.solve()
    u_optimal, policy_optimal = dp.optimalTrajectory(initialStock)

    print(f"states of the optimal trajectory: {u_optimal}")
    print(f"Decisions for the optimal trajectory: {policy_optimal}")
    print(f"Cost of the optimal trajectory: {dp.F(0, initialStock)}")