import numpy as np
import unittest
from impl.assignsex import assign_sex

class AssignSexTestCase(unittest.TestCase):
    def test_assign_sex(self):
        n = int(1e5)
        pmale = 0.44
        sexes = assign_sex(pmale, n)
        
        assert np.all((sexes >= 1) & (sexes <= 2)), "Sexes array contains elements outside the range [1, 2]"
        
        expected = np.array([pmale, 1 - pmale]) * n
        actual, _ = np.histogram(sexes, bins=[0.5, 1.5, 2.5])

        np.testing.assert_allclose(actual, expected, rtol=0.05)
