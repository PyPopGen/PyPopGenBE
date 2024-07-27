import numpy as np
from pypopgenbe.impl.normrnd0 import normrnd0


def vary_in_vitro_enzyme_rates(in_vitro_enzyme_rates: np.ndarray, in_vitro_enzyme_rate_coeffs_of_var: np.ndarray):
    """
    Adds normally distributed variation to the in-vitro enzyme rates.

    Parameters:
    in_vitro_enzyme_rates (np.ndarray): The in-vitro enzyme rates.
    in_vitro_enzyme_rate_coeffs_of_var (np.ndarray): The coefficients of variation for the in-vitro enzyme rates.

    Returns:
    None.
    """
    nan_coeffs_of_var = np.isnan(in_vitro_enzyme_rate_coeffs_of_var)
    if np.all(nan_coeffs_of_var):
        return

    to_vary = ~nan_coeffs_of_var & (in_vitro_enzyme_rate_coeffs_of_var != 0)

    in_vitro_enzyme_rates[to_vary] = normrnd0(
        in_vitro_enzyme_rates[to_vary],
        in_vitro_enzyme_rate_coeffs_of_var[to_vary]
    )
