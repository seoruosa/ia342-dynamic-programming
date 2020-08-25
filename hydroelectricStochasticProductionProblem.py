from dypro.dypro.discreteDynamicProgramming import DiscreteDynamicProgramming
from dypro.dypro.hydroelectricProduction import HydroelectricProduction
from dypro.dypro.discreteDynamicProblem import DiscreteDynamicProblem
from dypro.dypro.stochasticProblem import StochasticProblem

from dypro.dypro.utils import limit, sampling, secondsOfMonth, nearestSample
from dypro.dypro.solve import solveStochastic
from dypro.dypro.randomVariable import RandomVariable

import numpy as np

from time import time

import concurrent.futures

class HydroelectricStochasticProductionProblem(HydroelectricProduction,  StochasticProblem, DiscreteDynamicProgramming):
    def __init__(self, numberOfStages:int, year:int, maxFlow, stateSampling=100, decisionSampling=100, inf=np.inf):

        HydroelectricProduction.__init__(self, efficiency=0.88, gravity=10, minReservatoryVolume=12800,
            maxReservatoryVolume=21200, minTurbineFlow=1400, maxTurbineFlow=7955, maxProductionCapacity=3230,
            uprightPolinomy=lambda x: (303.04+(0.0015519*x) - (0.17377e-7)*(x**2)),
            downstreamPolinomy=lambda x: (279.84 +(0.22130e-3)*x))

        DiscreteDynamicProgramming.__init__(self, numberOfStages=numberOfStages, inf=inf)

        self.__maxFlow = maxFlow
        self.__year = year
        self.__stateSampling = stateSampling
        self.__decisionSampling = decisionSampling

        flowDistribution = [
            [(6375, 0.1), (7741, 0.2), (9107, 0.4), (10473, 0.2), (11839, 0.1)], 
            [(5382, 0.1), (6535, 0.2), (7688, 0.4), (8841, 0.2), (9995, 0.1)],
            [(6550, 0.1), (7954, 0.2), (9358, 0.4), (10762, 0.2), (12165, 0.1)],
            [(4756, 0.1), (5775, 0.2), (6794, 0.4), (7814, 0.2), (8832, 0.1)],
            [(3012, 0.1), (3658, 0.2), (4303, 0.4), (4948, 0.2), (5594, 0.1)],
            [(2473, 0.1), (3003, 0.2), (3533, 0.4), (4063, 0.2), (4593, 0.1)],
            [(2007, 0.1), (2437, 0.2), (2867, 0.4), (3297, 0.2), (3727, 0.1)],
            [(1790, 0.1), (2173, 0.2), (2557, 0.4), (2941, 0.2), (3324, 0.1)],
            [(1520, 0.1), (1845, 0.2), (2171, 0.4), (2497, 0.2), (2822, 0.1)],
            [(1573, 0.1), (1910, 0.2), (2247, 0.4), (2584, 0.2), (2921, 0.1)],
            [(2462, 0.1), (2989, 0.2), (3517, 0.4), (4045, 0.2), (4572, 0.1)],
            [(2926, 0.1), (3553, 0.2), (4180, 0.4), (4807, 0.2), (5434, 0.1)]
            ]

        self.__realizableRandomValues = {k:RandomVariable([i[0] for i in el], [i[1] for i in el]) for k, el in enumerate(flowDistribution)}
    
    def realizableRandomValues(self, k:int) -> RandomVariable:
        return self.__realizableRandomValues[k]
    
    def monthlyPowerDemand(self, month:int) -> float:
        """Returns the expected demand of each month"""
        
        monthlyPowerDemandMap = {
            0: 2800,
            1: 2500,
            2: 2300,
            3: 2500,
            4: 2600,
            5: 2800,
            6: 2800,
            7: 2600,
            8: 2600,
            9: 2800,
            10: 3000,
            11: 3100 
            }
        
        if (month in monthlyPowerDemandMap):
            return monthlyPowerDemandMap.get(month)
        else:
            raise KeyError(f"dont exist month {month}")
    
    def thermalProductionCost(self, energy) -> float:
        return 64.8 * energy*energy*1e-6



    def state(self, k:int) -> np.array:
        return sampling(self.minReservatoryVolume(), self.maxReservatoryVolume(), self.__stateSampling)

    def decision(self, k:int) -> np.array:
        return sampling(self.minTurbineFlow(), self.__maxFlow, self.__decisionSampling)

    def elementaryCost(self, k:int, state:np.array, decision:np.array) -> float:
        demand = self.monthlyPowerDemand(k)
        generatedEnergy = self.generatedEnergy(state, decision)

        if generatedEnergy >= demand and generatedEnergy < self.maxProductionCapacity():
            cost = 0
        elif generatedEnergy>self.maxProductionCapacity():
            cost = self.thermalProductionCost(demand - self.maxProductionCapacity())
        else:
            cost = self.thermalProductionCost(demand - generatedEnergy)            
        
        return cost

    def finalStateCost(self, state:np.array) -> float:
        cost = 0.0247*(state-12800)*1e6

        return cost

    def transitionFunction(self, k:int, state:np.array, decision:np.array, randomValue:np.array) -> np.array:
        return state + (randomValue - decision)*secondsOfMonth(self.__year, k)*1e-6
    
    def solveInfeasibility(self, k:int, state:np.array, FMap:dict):
        if state<self.minReservatoryVolume() or state>self.maxReservatoryVolume():
            return (limit(state, self.minReservatoryVolume(), self.maxReservatoryVolume()), self.inf)
        
        else:
            sampledState = nearestSample(state, self.minReservatoryVolume(), self.maxReservatoryVolume(), self.__stateSampling)
            return (sampledState, FMap[(k, sampledState)])

    def initialState(self) -> np.array:
        return 15000
    
    def solve(self):
        return solveStochastic(self.state, self.numberOfStages, self.finalStateCost, self.decision, self.realizableRandomValues, self.solveInfeasibility,
        self.transitionFunction, self.elementaryCost, inf=self.inf)
    
    def costOfSolution(self, initialState, FMap):
        nearestVolume = lambda vol: nearestSample(vol, self.minReservatoryVolume(), self.maxReservatoryVolume(), self.__stateSampling)

        return (nearestVolume(initialState), FMap[0, nearestVolume(initialState)])
        
    

if __name__ == "__main__":

    start = time()
    h = HydroelectricStochasticProductionProblem(numberOfStages=12, year=2021, maxFlow=15000, stateSampling=10, decisionSampling=10, inf=100000)
    
    fmap, policy = h.solve()

    nearestInitial, cost = h.costOfSolution(h.initialState(), fmap)

    print(f"initialVolume: {nearestInitial}\nCost: {cost} M de reais\n")


    print(f"duration: {time() - start}")
    print("-------------------------------------------")

    # print(fmap)

    # print("\n\n\n\n")

    # print(policy)

        
