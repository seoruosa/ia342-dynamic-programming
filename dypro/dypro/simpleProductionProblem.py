from discreteDynamicProblem import DiscreteDynamicProblem
from discreteDynamicProblem import generatorFromLIst
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
    
    def optimalTrajectory(self):
        u_optimal = [self.initialState()]
        policy_optimal = list()
        for k in range(self.numberOfStages()):
            policy_optimal.append(self.policy(k, u_optimal[-1]))
            u_optimal.append(self.transitionFunction(k, u_optimal[-1], policy_optimal[-1]))
        
        return (u_optimal, policy_optimal)

if __name__ == '__main__':
    
    STAGES = 4
    demandStage = lambda k: [2,1,1,0][k]
    stockCost = lambda k, state: [0, 3, 7][state] if state<3 else np.inf
    productionCost = lambda k, decision: [10, 17, 20][decision]
    stateSpace = lambda k: generatorFromLIst([0, 1, 2])
    productionSpace = lambda k: generatorFromLIst([0, 1, 2])
    finalState = lambda state: 1 if state==1 else np.inf
    initialStock = 1
    
    dp = SimpleProductionProblem(STAGES, initialStock, demandStage, productionSpace, stateSpace, finalState, productionCost, stockCost)
    dp.solve()
    u_optimal, policy_optimal = dp.optimalTrajectory()

    print(u_optimal)
    print(policy_optimal)
    # for i in range(5):
    #     for j in productionSpace(i):
    #         print(f"{i}-- {j}")