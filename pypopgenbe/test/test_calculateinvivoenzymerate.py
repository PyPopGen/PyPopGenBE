import unittest
import numpy as np
from pypopgenbe.impl.calculateinvivoenzymerate import calculate_in_vivo_enzyme_rate


class TestCalculateInVivoEnzymeRate(unittest.TestCase):

    def test_calculate_in_vivo_enzyme_rate(self):
        in_vitro_enzyme_rate = np.array([1, 2])
        mppgl = np.array([10, 50, 2.5])
        liver_mass = np.array([4, 3, 2])

        expected_output = np.array([[40, 80], [150, 300], [5, 10]])
        result = calculate_in_vivo_enzyme_rate(
            in_vitro_enzyme_rate, mppgl, liver_mass)

        np.testing.assert_array_almost_equal(result, expected_output)
