from dypro.dypro.discreteDynamicProgramming import DiscreteDynamicProgramming
from dypro.dypro.hydroelectricProduction import HydroelectricProduction

from dypro.dypro.utils import limit, sampling, secondsOfMonth, nearestSample
from dypro.dypro.solve import solve, optimalTrajectory

class HydroelectricAguaVermelha(HydroelectricProduction):
    def __init__(self):
        super().__init__(efficiency=0.88, 
                         gravity=10, 
                         minReservatoryVolume=4400, 
                         maxReservatoryVolume=11000, 
                         minTurbineFlow=475, 
                         maxTurbineFlow=2710, 
                         maxProductionCapacity=3230, 
                         uprightPolinomy=lambda x: 355.53 + (0.0036268*x) - (0.10090e-7)*(x**2),
                         downstreamPolinomy=lambda x: 319.91 + (0.15882e-2)*x
                        )

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

class HydroelectricIlhaSolteira(HydroelectricProduction):
    def __init__(self):
        super().__init__(efficiency=0.89,
                         gravity=10,
                         minReservatoryVolume=12800,
                         maxReservatoryVolume=21200,
                         minTurbineFlow=1400,
                         maxTurbineFlow=7955,
                         maxProductionCapacity=1380,
                         uprightPolinomy=lambda x: 303.04 + (0.0015519*x) - (0.17377e-7)*(x),
                         downstreamPolinomy=lambda x: 279.84 + (0.22130e-3)*x
                        )

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