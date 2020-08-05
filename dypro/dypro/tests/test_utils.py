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