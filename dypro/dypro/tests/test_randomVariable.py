import unittest

from ..randomVariable import RandomVariable
from ..randomVariable import ValueProbability

class RandomVariableTestCase(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_value_probability_init_valid(self):
        value = 1
        prob = 0.5
        a = ValueProbability(value, prob)

        self.assertEqual(value, a.getValue())
        self.assertEqual(prob, a.getProbability())

    def test_value_probability_init_array_valid(self):
        value = (0, 1)
        prob = 0.7
        a = ValueProbability(value, prob)

        self.assertEqual(value, a.getValue())
        self.assertEqual(prob, a.getProbability())

    def test_value_probability_init_object_valid(self):
        value = object()
        prob = 0.5
        a = ValueProbability(value, prob)

        self.assertEqual(value, a.getValue())
        self.assertEqual(prob, a.getProbability())

    def test_value_probability_init_invalid(self):
        value = 1
        prob = 1.5
        with self.assertRaises(TypeError):
         a = ValueProbability(value, prob)
    
    def test_random_variable_init_valid(self):
        values = [1]
        probs = [1]
        a = RandomVariable(values, probs)
        b = ValueProbability(values[0], probs[0])

        self.assertEqual(a.randomValue()[0].getValue(), b.getValue())
        self.assertEqual(a.randomValue()[0].getProbability(), b.getProbability())
    
    def test_random_variable_init_valid_sum_almost_equal_1(self):
        values = (1,2,3)
        probs = (0.2, 0.7, 0.1)
        
        a = RandomVariable(values, probs)
        
    
    def test_random_variable_init_valid_multiple_values(self):
        values = [1, 2]
        probs = [0.5, 0.5]
        a = RandomVariable(values, probs)
        b1 = ValueProbability(values[0], probs[0])
        b2 = ValueProbability(values[1], probs[1])

        self.assertEqual(a.randomValue()[0].getValue(), b1.getValue())
        self.assertEqual(a.randomValue()[0].getProbability(), b1.getProbability())

        self.assertEqual(a.randomValue()[1].getValue(), b2.getValue())
        self.assertEqual(a.randomValue()[1].getProbability(), b2.getProbability())
    
    def test_random_variable_init_invalid_probs_sum(self):
        values = [1, 2]
        probs = [0.5, 0.6]
        with self.assertRaisesRegex(Exception, "Sum of probabilities is not 1"):
            a = RandomVariable(values, probs)

    def test_random_variable_init_invalid_prob(self):
        values = [1]
        probs = [1.2]
        with self.assertRaisesRegex(Exception, "Sum of probabilities is not 1"):
            a = RandomVariable(values, probs)
    
    def test_random_variable_init_invalid_number_of_elements(self):
        values = [1, 2]
        probs = [1]
        with self.assertRaisesRegex(Exception, "Number of values is not equal probabilities"):
            a = RandomVariable(values, probs)
    
