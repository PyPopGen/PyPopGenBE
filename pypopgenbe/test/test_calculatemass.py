import unittest
import numpy as np
from pypopgenbe.impl.calculatemass import calculate_mass


class TestCalculateMass(unittest.TestCase):
    def test_calculate_mass(self):
        assert np.isclose(calculate_mass(20, 176), 20 * (1.76 ** 2), rtol=0.15)
        assert np.isclose(calculate_mass(25, 120), 25 * (1.2 ** 2), rtol=0.15)

    def test_calculate_mass_exceptions(self):
        with self.assertRaises(TypeError):
            calculate_mass()  # type: ignore
        with self.assertRaises(TypeError):
            calculate_mass(20)  # type: ignore
        with self.assertRaises(TypeError):
            calculate_mass(20, '176')  # type: ignore
        # with self.assertRaises(ValueError):
        #     calculate_mass(-20, 176)
        # with self.assertRaises(ValueError):
        #     calculate_mass(20, -176)
