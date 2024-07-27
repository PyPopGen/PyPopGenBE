import unittest
import numpy as np
from pypopgenbe.impl.assignethnicity import assign_ethnicity


class TestAssignEthnicity(unittest.TestCase):
    def test_assign_ethnicity(self):
        n = int(1e4)
        ethnicities = np.zeros(n, dtype=int)
        ethnicity_breaks = [0.2, 0.5, 0.8, 1.0]

        for i in range(n):
            ethnicities[i] = assign_ethnicity(np.array(ethnicity_breaks))

        neth = len(ethnicity_breaks)

        self.assertTrue(np.all((ethnicities >= 1) & (
            ethnicities <= neth)), "Ethnicities are out of expected range.")

        expected = n * np.diff([0] + ethnicity_breaks)
        actual = np.histogram(ethnicities, bins=np.arange(1, neth+2))[0]

        np.testing.assert_allclose(
            actual, expected, rtol=0.05, err_msg="The actual ethnic distribution does not match the expected distribution within tolerance.")
