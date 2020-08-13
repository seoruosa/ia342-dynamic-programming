from dypro.dypro.discreteDynamicProgramming import DiscreteDynamicProgramming
from dypro.dypro.utils import limit, sampling, secondsOfMonth, nearestSample
from dypro.dypro.solve import solve, optimalTrajectory
import numpy as np

from time import time

import concurrent.futures

class HidroeletricProductionProblem(DiscreteDynamicProgramming):
    def __init__(self, numberOfStages:int, year:int, stateSampling=100, decisionSampling=100, inf=np.inf):
        super().__init__(numberOfStages=numberOfStages, inf=inf)

        self.__year = year
        self.__stateSampling = stateSampling
        self.__decisionSampling = decisionSampling

        self.__efficiency = 0.88
        self.__gravity = 10
        self.__minReservatoryVolume = 12800 #10^6 m^3
        self.__maxReservatoryVolume = 21200 #10^6 m^3
        self.__minTurbineFlow = 1400 #m^3/s
        self.__maxTurbineFlow = 7955 #m^3/s
        self.__maxFlow = 10000 #m^3/s
    
    def monthlyAvgFlow(self, month:int) -> float:
        """Returns the average flow rate[m^3/s] at each month"""
        monthlyAvgFlowMap = {
            0: 9107,
            1: 7688,
            2: 9358,
            3: 6794,
            4: 4303,
            5: 3533,
            6: 2867,
            7: 2557,
            8: 2171,
            9: 2247,
            10: 3517,
            11: 4180 
            }
        if (month in monthlyAvgFlowMap):
            return monthlyAvgFlowMap.get(month)
        else:
            raise KeyError(f"dont exist month {month}")
    
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

    def maximumProductionCapacity(self) -> float:
        return 3230.0
    
    def rho(self) -> float:
        return self.__efficiency * self.__gravity * 1e-3
    
    def uprightHeight(self, reservatoryVolume:float) -> float:
        limitedVolume = limit(reservatoryVolume, self.__minReservatoryVolume, self.__maxReservatoryVolume)
        
        return 303.04 + (0.0015519 * limitedVolume) - (0.17377e-7)*limitedVolume**2

    def downstreamHeight(self, turbinedFlow:float) -> float:
        return 279.84 + (0.22130e-3) * turbinedFlow
    
    def generatedEnergy(self, reservatoryVolume:float, turbinedFlow:float):
        flowThatGenerateEnergy = limit(turbinedFlow, self.__minTurbineFlow, self.__maxTurbineFlow)

        return (self.rho() * (self.uprightHeight(reservatoryVolume) - self.downstreamHeight(turbinedFlow)) * flowThatGenerateEnergy)

    # ---------------------------------------------------------------------------------------------------------------    

    def state(self, k:int) -> np.array:
        return sampling(self.__minReservatoryVolume, self.__maxReservatoryVolume, self.__stateSampling)

    def decision(self, k:int) -> np.array:
        return sampling(self.__minTurbineFlow, self.__maxFlow, self.__decisionSampling)

    def elementaryCost(self, k:int, state:np.array, decision:np.array) -> float:
        demand = self.monthlyPowerDemand(k)
        generatedEnergy = self.generatedEnergy(state, decision)

        if generatedEnergy >= demand and generatedEnergy < self.maximumProductionCapacity():
            cost = 0
        elif generatedEnergy>self.maximumProductionCapacity():
            cost = self.thermalProductionCost(demand - self.maximumProductionCapacity())
        else:
            cost = self.thermalProductionCost(demand - generatedEnergy)            
        
        return cost

    def finalStateCost(self, state:np.array) -> float:
        if state<15000:
            cost = self.inf
        else:
            cost = 0

        return cost

    def transitionFunction(self, k:int, state:np.array, decision:np.array) -> np.array:
        return state + (self.monthlyAvgFlow(k) - decision)*secondsOfMonth(self.__year, k)*1e-6
    
    def solveInfeasibility(self, k:int, state:np.array, FMap:dict):
        if state<self.__minReservatoryVolume or state>self.__maxReservatoryVolume:
            return (limit(state, self.__minReservatoryVolume, self.__maxReservatoryVolume), self.inf)
        
        else:
            sampledState = nearestSample(state, self.__minReservatoryVolume, self.__maxReservatoryVolume, self.__stateSampling)
            return (sampledState, FMap[(k, sampledState)])

    def initialState(self) -> np.array:
        return 15000
    
    def solve(self):
        return solve(self.state, self.numberOfStages, self.finalStateCost, self.decision, self.solveInfeasibility,
        self.transitionFunction, self.elementaryCost, inf=self.inf)
    
    def optimalTrajectoryHidro(self, policy, initialState):
        nearestVolume = lambda vol: nearestSample(vol, self.__minReservatoryVolume, self.__maxReservatoryVolume, self.__stateSampling)
        transitionWithNearest = lambda k, state, decision: nearestVolume(self.transitionFunction(k, state, decision))

        return optimalTrajectory(nearestVolume(initialState), self.numberOfStages, policy, transitionWithNearest)
    
    def costOfSolution(self, initialState, FMap):
        nearestVolume = lambda vol: nearestSample(vol, self.__minReservatoryVolume, self.__maxReservatoryVolume, self.__stateSampling)

        return (nearestVolume(initialState), FMap[0, nearestVolume(initialState)])


def run(pairOfSamplings):

    _stateSampling, _decisionSampling = pairOfSamplings
    start = time()

    hidro = HidroeletricProductionProblem(numberOfStages=11, year=2021,  stateSampling=_stateSampling, decisionSampling=_decisionSampling, inf=10000)
    fmap, policy = hidro.solve()
    u_optimal, policy_optimal = hidro.optimalTrajectoryHidro(policy, hidro.initialState())

    nearestInitial, cost = hidro.costOfSolution(hidro.initialState(), fmap)
    print(f"initialVolume: {nearestInitial}\nCost: {cost} M de reais\n")

    # print(f"\npolicy:\n{policy}")
    print(f"u_optimal: {u_optimal}")
    print(f"policy_optimal: {policy_optimal}")

    print(f"duration: {time() - start} \nsampling: state:{_stateSampling} x decision: {_decisionSampling}\n\n")
    print("-------------------------------------------")

if __name__ == '__main__':
    
    # samplings = [10, 100, 200, 400, 500, 750, 1000, 2000, 5000]
    # samplingPairs = [(a, b) for a in samplings for b in samplings]

    # with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:

    #     executor.map(run, samplingPairs)
    run((100, 100))