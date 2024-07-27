import numpy as np
import unittest
from pypopgenbe.impl.invertindicies import invert_indices


class TestInvertIndices(unittest.TestCase):

    def test_invert_indices(self):
        n = 10
        self.assertTrue(np.array_equal(invert_indices(
            n, np.array(range(0, 10))), np.zeros(0)))
        self.assertTrue(np.array_equal(
            invert_indices(n, np.array([])), np.arange(0, 10)))
        self.assertTrue(np.array_equal(invert_indices(
            n, np.array([3, 5, 6, 8])), np.array([0, 1, 2, 4, 7, 9])))
        self.assertTrue(np.array_equal(invert_indices(
            n, np.array([6, 8, 3, 5])), np.array([0, 1, 2, 4, 7, 9])))
