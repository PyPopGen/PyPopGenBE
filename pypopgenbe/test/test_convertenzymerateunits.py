import unittest
import numpy as np

from pypopgenbe.impl.convertenzymerateunits import convert_enzyme_rate_units
from pypopgenbe.impl.enum import EnzymeRateCLintUnits, EnzymeRateVmaxUnits


class TestConvertEnzymeRateUnits(unittest.TestCase):

    def test_micro_litres_per_minute(self):
        x = np.arange(1, 11)
        expected = np.arange(1, 11)
        result = convert_enzyme_rate_units(x, EnzymeRateCLintUnits.MicroLitresPerMinute)
        np.testing.assert_almost_equal(result, expected)

    def test_milli_litres_per_hour(self):
        x = np.arange(1, 11)
        expected = 0.06 * np.arange(1, 11)
        result = convert_enzyme_rate_units(x, EnzymeRateCLintUnits.MilliLitresPerHour)
        np.testing.assert_almost_equal(result, expected)

    def test_litres_per_hour(self):
        x = np.arange(1, 11)
        expected = 6e-5 * np.arange(1, 11)
        result = convert_enzyme_rate_units(x, EnzymeRateCLintUnits.LitresPerHour)
        np.testing.assert_almost_equal(result, expected)

    def test_pico_mols_per_minute(self):
        x = np.arange(1, 11)
        expected = np.arange(1, 11)
        result = convert_enzyme_rate_units(x, EnzymeRateVmaxUnits.PicoMolsPerMinute)
        np.testing.assert_almost_equal(result, expected)

    def test_micro_mols_per_hour(self):
        x = np.arange(1, 11)
        expected = 0.06 * np.arange(1, 11)
        result = convert_enzyme_rate_units(x, EnzymeRateVmaxUnits.MicroMolsPerHour)
        np.testing.assert_almost_equal(result, expected)

    def test_milli_mols_per_hour(self):
        x = np.arange(1, 11)
        expected = 6e-5 * np.arange(1, 11)
        result = convert_enzyme_rate_units(x, EnzymeRateVmaxUnits.MilliMolsPerHour)
        np.testing.assert_almost_equal(result, expected)

    def test_pico_grams_per_minute(self):
        x = np.arange(1, 11)
        weight = 5
        expected = np.arange(1, 11) * weight
        result = convert_enzyme_rate_units(x, EnzymeRateVmaxUnits.PicoGramsPerMinute, weight)
        np.testing.assert_almost_equal(result, expected)

    def test_micro_grams_per_hour(self):
        x = np.arange(1, 11)
        weight = 5
        expected = 0.06 * np.arange(1, 11) * weight
        result = convert_enzyme_rate_units(x, EnzymeRateVmaxUnits.MicroGramsPerHour, weight)
        np.testing.assert_almost_equal(result, expected)

    def test_milli_grams_per_hour(self):
        x = np.arange(1, 11)
        weight = 5
        expected = 6e-5 * np.arange(1, 11) * weight
        result = convert_enzyme_rate_units(x, EnzymeRateVmaxUnits.MilliGramsPerHour, weight)
        np.testing.assert_almost_equal(result, expected)
