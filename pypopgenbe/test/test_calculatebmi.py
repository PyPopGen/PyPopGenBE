import numpy as np
import unittest
from pypopgenbe.impl.calculatebmi import calculate_bmi


class TestCalculateBmi(unittest.TestCase):

    def test_calculate_bmi(self):
        np.isclose(calculate_bmi(72, 176), 72 / 1.76**2, rtol=0.15)
        np.isclose(calculate_bmi(40, 120), 40 / 1.2**2, rtol=0.15)

    def test_calculate_bmi_invalid_inputs(self):
        with self.assertRaises(ValueError):
            calculate_bmi(-72, 176)
        with self.assertRaises(ValueError):
            calculate_bmi(72, -176)
        with self.assertRaises(ValueError):
            calculate_bmi(0, 176)
        with self.assertRaises(ValueError):
            calculate_bmi(72, 0)
        with self.assertRaises(TypeError):
            calculate_bmi()  # type: ignore
        with self.assertRaises(TypeError):
            calculate_bmi(72)  # type: ignore
        with self.assertRaises(TypeError):
            calculate_bmi(72, 176, 30)  # type: ignore
