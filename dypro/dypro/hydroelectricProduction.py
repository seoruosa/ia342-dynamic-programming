from dypro.dypro.utils import limit

class HydroelectricProduction:
    def __init__(self, efficiency, gravity, minReservatoryVolume, maxReservatoryVolume, 
                    minTurbineFlow, maxTurbineFlow, maxProductionCapacity, 
                    uprightPolinomy, downstreamPolinomy):
        self.__efficiency = efficiency
        self.__gravity = gravity
        self.__minReservatoryVolume = minReservatoryVolume #10^6 m^3
        self.__maxReservatoryVolume = maxReservatoryVolume #10^6 m^3
        self.__minTurbineFlow = minTurbineFlow #m^3/s
        self.__maxTurbineFlow = maxTurbineFlow #m^3/s
        self.__maxProductionCapacity = maxProductionCapacity
        self.__uprightPolinomy = uprightPolinomy
        self.__downstreamPolinomy = downstreamPolinomy

    def maxProductionCapacity(self) -> float:
        return self.__maxProductionCapacity
    
    def rho(self) -> float:
        return self.__efficiency * self.__gravity * 1e-3
    
    def uprightHeight(self, reservatoryVolume:float) -> float:
        limitedVolume = limit(reservatoryVolume, self.__minReservatoryVolume, self.__maxReservatoryVolume)
        
        return self.__uprightPolinomy(limitedVolume)

    def downstreamHeight(self, turbinedFlow:float) -> float:
        return self.__downstreamPolinomy(turbinedFlow)
    
    def generatedEnergy(self, reservatoryVolume:float, turbinedFlow:float):
        flowThatGenerateEnergy = limit(turbinedFlow, self.__minTurbineFlow, self.__maxTurbineFlow)

        return (self.rho() * (self.uprightHeight(reservatoryVolume) - self.downstreamHeight(turbinedFlow)) * flowThatGenerateEnergy)
    
    def minReservatoryVolume(self):
        return self.__minReservatoryVolume
    
    def maxReservatoryVolume(self):
        return self.__maxReservatoryVolume
    
    def minTurbineFlow(self):
        return self.__minTurbineFlow
    
    def maxTurbineFlow(self):
        return self.__maxTurbineFlow
    

    