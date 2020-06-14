import unittest

from .discreteDynamicProblem import *

class DynamicProblemTestCase(unittest.TestCase):
    
    class DP(DiscreteDynamicProblem):
            def state(self, k:int) -> np.array:
                pass

            def decision(self, k:int) -> np.array:
                pass

            def elementaryCost(self, k:int, state:np.array, decision:np.array) -> float:
                pass

            def finalStateCost(self, state:np.array) -> float:
                pass

            def transictionFunction(self, k:int, state:np.array, decision:np.array) -> np.array:
                pass

            def initialState(self) -> np.array:
                pass
    
    def setUp(self):
        pass
    
    def test_configured_numberOfStages_and_inf(self):
        numberOfStages = 5
        inf = 100
        dp = self.DP(numberOfStages, inf)
        self.assertEqual(dp.numberOfStages(), numberOfStages)
        self.assertEqual(dp.inf(), inf)
    
    def test_configure_numberOfStages_and_not_configured_inf(self):
        import numpy as np
        numberOfStages = 5
        dp = self.DP(numberOfStages)

        self.assertEqual(dp.numberOfStages(), numberOfStages)
        self.assertEqual(dp.inf(), np.inf)

    def test_accOptimalCost_without_input(self):
        dp = self.DP(5)

        self.assertEqual(dp.accOptimalCost(), dict())
    
    def test_accOptimalCost_with_input(self):
        dictAcc = {'a':1}
        dp = self.DP(5, F=dictAcc)

        self.assertEqual(dp.accOptimalCost(), dictAcc)

    