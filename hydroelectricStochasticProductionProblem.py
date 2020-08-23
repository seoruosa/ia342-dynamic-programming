from dypro.dypro.discreteDynamicProgramming import DiscreteDynamicProgramming
from dypro.dypro.hydroelectricProduction import HydroelectricProduction
from dypro.dypro.discreteDynamicProblem import DiscreteDynamicProblem
from dypro.dypro.stochasticProblem import StochasticProblem

from dypro.dypro.utils import limit, sampling, secondsOfMonth, nearestSample
from dypro.dypro.solve import solve, optimalTrajectory

import numpy as np

from time import time

import concurrent.futures

class HydroelectricStochasticProductionProblem(HydroelectricProduction,  StochasticProblem):
    def __init__(self, numberOfStages:int, year:int, maxFlow, stateSampling=100, decisionSampling=100, inf=np.inf):

        HydroelectricProduction.__init__(self, efficiency=0.88, gravity=10, minReservatoryVolume=12800,
            maxReservatoryVolume=21200, minTurbineFlow=1400, maxTurbineFlow=7955, maxProductionCapacity=3230,
            uprightPolinomy=lambda x: (303.04+(0.0015519*x) - (0.17377e-7)*(x**2)),
            downstreamPolinomy=lambda x: (279.84 +(0.22130e-3)*x))

        self.__maxFlow = maxFlow
        self.__year = year
        self.__stateSampling = stateSampling
        self.__decisionSampling = decisionSampling
    

if __name__ == "__main__":
    
    # h = HydroelectricStochasticProductionProblem(10, 2021, 10000, 1000, 1000)

    # print(h)

    a = [
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
    
    for i in a:
        print(len(i))
