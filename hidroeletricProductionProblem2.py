from dypro.dypro.utils import limit, sampling, secondsOfMonth, nearestSample
from dypro.dypro.solve import solve, optimalTrajectory
import numpy as np
import logging
from time import time

import concurrent.futures

logging.basicConfig(level=logging.DEBUG)

class HidroeletricAguaVermelha:
    def __init__(self, stateSampling=100, decisionSampling=100, inf=np.inf):
        self.__efficiency = 0.88
        self.__gravity = 10
        self.__minReservatoryVolume = 4400 #10^6 m^3
        self.__maxReservatoryVolume = 11000 #10^6 m^3
        self.__minTurbineFlow = 475 #m^3/s
        self.__maxTurbineFlow = 2710 #m^3/s
        self.__maxFlow = 5000 #m^3/s
        self.__year = 2021
        self.__numberOfStages = 11
        self.__stateSampling = stateSampling
        self.__decisionSampling = decisionSampling
        self.__inf = inf
    
    def monthlyAvgFlow(self, month:int) -> float:
        """Returns the average flow rate[m^3/s] at each month"""
        monthlyAvgFlowMap = {
            0: 3899,
            1: 3202,
            2: 2953,
            3: 2600,
            4: 1671,
            5: 1271,
            6: 1045,
            7: 972,
            8: 792,
            9: 790,
            10: 1139,
            11: 1589
            }
        if (month in monthlyAvgFlowMap):
            return monthlyAvgFlowMap.get(month)
        else:
            raise KeyError(f"dont exist month {month}")      

    def maximumProductionCapacity(self) -> float:
        return 3230.0
    
    def rho(self) -> float:
        return self.__efficiency * self.__gravity * 1e-3
    
    def uprightHeight(self, reservatoryVolume:float) -> float:
        limitedVolume = limit(reservatoryVolume, self.__minReservatoryVolume, self.__maxReservatoryVolume)
        
        return 355.53 + (0.0036268 * limitedVolume) - (0.10090e-7)*limitedVolume**2

    def downstreamHeight(self, turbinedFlow:float) -> float:
        return 319.91 + (0.15882e-2) * turbinedFlow
    
    def generatedEnergy(self, reservatoryVolume:float, turbinedFlow:float):
        flowThatGenerateEnergy = limit(turbinedFlow, self.__minTurbineFlow, self.__maxTurbineFlow)

        _generatedEnergy = min(self.maximumProductionCapacity(), self.rho() * (self.uprightHeight(reservatoryVolume) - self.downstreamHeight(turbinedFlow)) * flowThatGenerateEnergy)

        return _generatedEnergy
    
    def state(self, k:int) -> np.array:
        return sampling(self.__minReservatoryVolume, self.__maxReservatoryVolume, self.__stateSampling)
    
    def decision(self, k:int) -> np.array:
        return sampling(self.__minTurbineFlow, self.__maxFlow, self.__decisionSampling)
    
    def solveInfeasibility(self, k:int, state:np.array):
        """[summary]

        Args:
            k (int): [description]
            state (np.array): [description]
            FMap (dict): [description]

        Returns:
            [type]: (next state, isInfeasible)
        """
        isInfeasible = state<self.__minReservatoryVolume or state>self.__maxReservatoryVolume
      
        if isInfeasible:
            feasibleState = limit(state, self.__minReservatoryVolume, self.__maxReservatoryVolume)
        
        else:
            feasibleState = nearestSample(state, self.__minReservatoryVolume, self.__maxReservatoryVolume, self.__stateSampling)
            
        return (feasibleState, isInfeasible)


class HidroeletricIlhaSolteira:
    def __init__(self, stateSampling=100, decisionSampling=100, inf=np.inf):
        self.__efficiency = 0.89
        self.__gravity = 10
        self.__minReservatoryVolume = 12800 #10^6 m^3
        self.__maxReservatoryVolume = 21200 #10^6 m^3
        self.__minTurbineFlow = 1400 #m^3/s
        self.__maxTurbineFlow = 7955 #m^3/s
        self.__maxFlow = 10000 #m^3/s
        self.__year = 2021
        self.__numberOfStages = 11
        self.__stateSampling = stateSampling
        self.__decisionSampling = decisionSampling
        self.__inf = inf
    
    def monthlyAvgFlow(self, month:int) -> float:
        """Returns the average flow rate[m^3/s] at each month"""
        monthlyAvgFlowMap = {
            0: 5208,
            1: 4486,
            2: 6405,
            3: 4194,
            4: 2632,
            5: 2262,
            6: 1822,
            7: 1585,
            8: 1379,
            9: 1457,
            10: 2378,
            11: 2591
            }
        if (month in monthlyAvgFlowMap):
            return monthlyAvgFlowMap.get(month)
        else:
            raise KeyError(f"dont exist month {month}")      

    def maximumProductionCapacity(self) -> float:
        return 1380.0
    
    def rho(self) -> float:
        return self.__efficiency * self.__gravity * 1e-3
    
    def uprightHeight(self, reservatoryVolume:float) -> float:
        limitedVolume = limit(reservatoryVolume, self.__minReservatoryVolume, self.__maxReservatoryVolume)
        
        return 303.04 + (0.0015519 * limitedVolume) - (0.17377e-7)*limitedVolume**2

    def downstreamHeight(self, turbinedFlow:float) -> float:
        return 279.84 + (0.22130e-3) * turbinedFlow
    
    def generatedEnergy(self, reservatoryVolume:float, turbinedFlow:float):
        flowThatGenerateEnergy = limit(turbinedFlow, self.__minTurbineFlow, self.__maxTurbineFlow)
        
        _generatedEnergy = min(self.maximumProductionCapacity(), self.rho() * (self.uprightHeight(reservatoryVolume) - self.downstreamHeight(turbinedFlow)) * flowThatGenerateEnergy)

        return _generatedEnergy
    
    def state(self, k:int) -> np.array:
        return sampling(self.__minReservatoryVolume, self.__maxReservatoryVolume, self.__stateSampling)
    
    def decision(self, k:int) -> np.array:
        return sampling(self.__minTurbineFlow, self.__maxFlow, self.__decisionSampling)
    
    def solveInfeasibility(self, k:int, state:np.array):
        """[summary]

        Args:
            k (int): [description]
            state (np.array): [description]
            FMap (dict): [description]

        Returns:
            [type]: (next state, wasInfeasible)
        """
        wasInfeasible = state<self.__minReservatoryVolume or state>self.__maxReservatoryVolume
      
        if wasInfeasible:
            feasibleState = limit(state, self.__minReservatoryVolume, self.__maxReservatoryVolume)
        
        else:
            feasibleState = nearestSample(state, self.__minReservatoryVolume, self.__maxReservatoryVolume, self.__stateSampling)
            
        return (feasibleState, wasInfeasible)

class HidroeletricProductionProblem2:
    def __init__(self, stateSampling=100, decisionSampling=100, inf=np.inf):
        self.__aguaVermelha = HidroeletricAguaVermelha(stateSampling, decisionSampling, inf)
        self.__ilhaSolteira = HidroeletricIlhaSolteira(stateSampling, decisionSampling, inf)
        self.stateSampling = stateSampling
        self.decisionSampling = decisionSampling
        self.__numberOfStages = 11
        self.__inf = inf
        self.__year = 2021
    
    def monthlyPowerDemand(self, month:int) -> float:
        """Returns the expected demand of each month"""
        
        monthlyPowerDemandMap = {
            0: 3600,
            1: 3300,
            2: 3000,
            3: 3200,
            4: 3400,
            5: 3600,
            6: 3700,
            7: 3300,
            8: 3400,
            9: 3600,
            10: 3900,
            11: 4000 
            }
        
        if (month in monthlyPowerDemandMap):
            return monthlyPowerDemandMap.get(month)
        else:
            raise KeyError(f"dont exist month {month}")
    
    def thermalProductionCost(self, energy) -> float:
        return 64.8 * energy*energy*1e-6
    
    def generatedEnergy(self, state, decision):

        gen_agua_vermelha = self.__aguaVermelha.generatedEnergy(state[0], decision[0])
        gen_ilha_solteira = self.__ilhaSolteira.generatedEnergy(state[1], decision[1])
        
        return gen_agua_vermelha + gen_ilha_solteira

    def maximumProductionCapacity(self):
        return self.__aguaVermelha.maximumProductionCapacity() + self.__ilhaSolteira.maximumProductionCapacity()
    
    # ---------------------------------------------------------------------------------------------------------------    

    def state(self, k:int) -> np.array:
        samples_agua_vermelha = list(self.__aguaVermelha.state(k))
        samples_ilha_solteira = list(self.__ilhaSolteira.state(k))

        return [(a, b) for a in samples_agua_vermelha for b in samples_ilha_solteira]

    def decision(self, k:int) -> np.array:
        samples_agua_vermelha = list(self.__aguaVermelha.decision(k))
        samples_ilha_solteira = list(self.__ilhaSolteira.decision(k))

        return [(a, b) for a in samples_agua_vermelha for b in samples_ilha_solteira]

    def elementaryCost(self, k:int, state:np.array, decision:np.array) -> float:
        demand = self.monthlyPowerDemand(k)
        generatedEnergy = self.generatedEnergy(state, decision)
        
        cost = self.thermalProductionCost(demand - self.maximumProductionCapacity())
                
        return cost

    def finalStateCost(self, state:np.array) -> float:
        agua_vermelha_vol = state[0]
        ilha_solteira_vol = state[1]

        if agua_vermelha_vol<8000 or ilha_solteira_vol<15000:
            cost = self.__inf
        else:
            cost = 0

        return cost

    def transitionFunction(self, k:int, state:np.array, decision:np.array) -> np.array:
        aguaVermelhaState = state[0] + (self.__aguaVermelha.monthlyAvgFlow(k) - decision[0])*secondsOfMonth(self.__year, k)*1e-6
        ilhaSolteiraState = state[1] + (self.__ilhaSolteira.monthlyAvgFlow(k) - decision[1])*secondsOfMonth(self.__year, k)*1e-6


        return (aguaVermelhaState, ilhaSolteiraState)
    
    def solveInfeasibility(self, k:int, state:np.array, FMap:dict):
        aguaVermelhaState, aguaVermelhaWasInfeasible = self.__aguaVermelha.solveInfeasibility(k, state[0])
        ilhaSolteiraState, ilhaSolteiraWasInfeasible = self.__ilhaSolteira.solveInfeasibility(k, state[1])

        if aguaVermelhaWasInfeasible or ilhaSolteiraWasInfeasible:
            return ((aguaVermelhaState, ilhaSolteiraState), self.__inf)
        
        else:
            return ((aguaVermelhaState, ilhaSolteiraState), FMap[(k, (aguaVermelhaState, ilhaSolteiraState))])

    def initialState(self) -> np.array:
        return (8000, 15000)
    
    def solveHidro(self):
        return solve(self.state, self.__numberOfStages, self.finalStateCost, self.decision, self.solveInfeasibility,
        self.transitionFunction, self.elementaryCost, inf=self.__inf)
    
    def optimalTrajectoryHidro(self, policy, initialState):
        nearestVolume = lambda vol: nearestSample(vol, self.__minReservatoryVolume, self.__maxReservatoryVolume, self.__stateSampling)
        transitionWithNearest = lambda k, state, decision: nearestVolume(self.transitionFunction(k, state, decision))

        return optimalTrajectory(nearestVolume(initialState), self.__numberOfStages, policy, transitionWithNearest)
    
    def costOfSolution(self, initialState, FMap):
        nearestVolume = lambda vol: nearestSample(vol, self.__minReservatoryVolume, self.__maxReservatoryVolume, self.__stateSampling)

        return (nearestVolume(initialState), FMap[0, nearestVolume(initialState)])

if __name__ == "__main__":
    h = HidroeletricProductionProblem2(1000, 1000, 100000)

    Fmap, policy = h.solveHidro()

    print(Fmap)
    print("\n\n")
    print(policy)