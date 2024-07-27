import numpy as np
from typing import Dict, cast
import unittest
from pypopgenbe.impl.addstochasticvariation import add_stochastic_variation


def test_asv(mean: float, coeff_of_var: float):
    n = int(1e6)
    mean_in = np.full(n, mean)
    coeff_of_var_in = np.full(n, coeff_of_var)

    dist = {'IsNormal': np.full(n, True), 'IsLognormal': np.full(n, False)}
    test_asv_inner(mean_in.copy(), coeff_of_var_in, dist)

    dist = {'IsNormal': np.full(n, False), 'IsLognormal': np.full(n, True)}
    test_asv_inner(mean_in, coeff_of_var_in, dist)


def test_asv_inner(mean_in: np.ndarray, coeff_of_var_in: np.ndarray, dist: Dict[str, np.ndarray]):
    a = mean_in[0]
    add_stochastic_variation(mean_in, coeff_of_var_in, dist)
    assert np.isclose(np.mean(mean_in), a, rtol=0.15), f"Mean {
        np.mean(mean_in)} differs from {a}"
    assert np.isclose(coeff_of_var(mean_in), coeff_of_var_in[0], rtol=0.15), f"Coeff of var {
        coeff_of_var(mean_in)} differs from {coeff_of_var_in[0]}"


def coeff_of_var(x: np.ndarray) -> float:
    return cast(float, np.std(x) / np.mean(x))


class TestAddStochasticVariation(unittest.TestCase):
    def test_add_stochastic_variation(self):
        test_asv(3., 4.)
        test_asv(0.1, 5.)
        test_asv(1., 1.)
        # test_asv(2, 50)  # Uncomment if needed for testing large coefficients of variation
