import unittest

from ..utils import sampling

class UtilsTestCase(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_0_to_5_every_1(self):
        result = [0, 1, 2, 3, 4, 5]

        self.assertListEqual(result, list(sampling(0,5,1)))
    
    def test_0_to_2_every_0_5(self):
        result = [0, 0.5, 1, 1.5, 2]

        self.assertListEqual(result, list(sampling(0,2,0.5)))
    
    def test_0_to_1_every_0_3(self):
        result = [0, 0.3, 0.6, 0.9, 1]
        test = list(sampling(0, 1, 0.3))

        self.assertAlmostEqualForList(result, test)
    
    def test_2_variables(self):
        result = [(0,0), (0, 1), (1, 0), (1,1)]
        x = list(sampling(0, 1, 1))
        y = list(sampling(0, 1, 1))
        test = [(a, b) for a in x for b in y]

        self.assertAlmostEqualForList(result, test)


    def assertAlmostEqualForList(self, correct:list, test:list, places=4) -> bool:
        self.assertEqual(len(correct), len(test))

        for c, t in zip(correct, test):
            self.assertAlmostEqual(c, t, places)