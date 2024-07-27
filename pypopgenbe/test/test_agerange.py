import numpy as np
from scipy.stats import uniform, kstest
import unittest
from pypopgenbe.impl.assignage import assign_age
from pypopgenbe.impl.enum import PopulationType

class TestAssignAge(unittest.TestCase):

    def test_age_range(self):
        age_range = (16., 65.)
        ages = np.array([assign_age(PopulationType.HighVariation, age_range)
                        for _ in range(100000)])

        self.assertTrue(
            np.all(ages >= age_range[0] - 0.5) and np.all(ages < age_range[1] + 0.5))

        x = np.arange(age_range[0] - 0.5, age_range[1] - 0.5, 0.01)
        ucdf = uniform.cdf(x, age_range[0], age_range[1] - age_range[0])
        _, p = kstest(ages, lambda t: np.interp(t, x, ucdf))

        self.assertTrue(p < 0.01)
