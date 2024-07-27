import numpy as np
import unittest
from pypopgenbe.impl.assignsex import assign_sex


class TestAssignSex(unittest.TestCase):

    def test_assign_sex(self):
        n = int(1e5)
        pmale = 0.44
        sexes = np.array([assign_sex(pmale) for _ in range(n)])

        # Check that all values are either 1 or 2
        self.assertTrue(np.all(sexes >= 1) and np.all(sexes <= 2))

        # Check the distribution
        expected = n * np.array([pmale, 1 - pmale])
        actual = np.array([np.sum(sexes == 1), np.sum(sexes == 2)])

        np.testing.assert_allclose(actual, expected, rtol=0.05)
