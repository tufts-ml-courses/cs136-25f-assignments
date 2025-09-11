'''
Usage
-----
Execute as a script to run all unittests in verbose mode
'''

import unittest
import numpy as np

# Load methods we want to test
from div_doctest import division, division_vector

class TestDivision(unittest.TestCase):

    def test_returns_correct_values(self):
        ''' Check that division method returns expected values
        '''
        for a in [-5.0, -2.0, -0.1, .337, 4.56]:
            for b in [-3.0, -0.2, 0.8, 7.8]:
                method_ans = 1.0 # TODO FIXME, call division method
                ideal_ans = a / b # Assume this is ideal
                self.assertTrue(False) # TODO FIXME, assertEqual both answers

    def test_raise_error_when_div_by_zero(self):
        ''' Check that division method fails when divisor is zero
        '''
        self.assertTrue(False) # TODO FIXME assertRaises verify div by 0 is bad


class TestDivisionVector(unittest.TestCase):

    def test_returns_correct_values(self):
        ''' Check that division_vector returns expected values
        '''
        a_N = np.array([1.0, 2.0, 3.0])
        b_N = np.array([4.0, 8.0, 12.0])
        method_ans = 0.0 # TODO FIXME, call division_vector
        ideal_ans = 0.0 # TODO FIXME, use numpy to compute ideal directly
        self.assertTrue(False) # TODO FIXME, make sure both methods agree
        # Hint: see use numpy's allclose to verify answers

    def test_raise_warning_when_div_by_zero(self):
        ''' Check that division_vector warns you when any divisor is zero
        '''
        self.assertTrue(False) # TODO FIXME, use assertWarns
        

if __name__ == '__main__':
    unittest.main(verbosity=2)
