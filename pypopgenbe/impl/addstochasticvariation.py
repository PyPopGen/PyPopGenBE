import numpy as np
from typing import Dict
from pypopgenbe.impl.normrnd0 import normrnd0
from pypopgenbe.impl.lognrnd0 import lognrnd0


def add_stochastic_variation(target: np.ndarray, coeff_of_var: np.ndarray, dist: Dict[str, np.ndarray]):
    """
    Adds stochastic variation to organ masses/flows.

    Parameters:
    target (np.ndarray): Target organ masses/flows.
    coeff_of_var (np.ndarray): Coefficient of variation for the organ masses/flows.
    dist (Dict[str, np.ndarray]): Distribution information containing 'IsNormal' and 'IsLognormal' fields.

    Returns:
    None
    """
    # Retrieve boolean arrays for normal and lognormal distributions
    is_normal = dist['IsNormal']
    is_lognormal = dist['IsLognormal']

    # Apply normal distribution variation
    target[is_normal] = normrnd0(target[is_normal], coeff_of_var[is_normal])

    # Apply lognormal distribution variation
    target[is_lognormal] = lognrnd0(
        target[is_lognormal], coeff_of_var[is_lognormal])
