import numpy as np
from typing import Tuple
from pypopgenbe.impl.enum import PopulationType

def assign_target_height(
    population_type: PopulationType,
    range_vals: Tuple[float, float],
    mean: float,
    coeff_of_var: float
) -> float:
    """
    Assigns a random target height.

    Parameters:
    population_type (str): Either 'Realistic' or 'HighVariation'.
    range_vals (Tuple[float, float]): Numeric vector of length two, giving the lower and upper bounds of the target height.
    mean (float): The mean height of the population to sample from.
    coeff_of_var (float): The coefficient of variation of population heights.

    Returns:
    float: Assigned target height.
    """

    if population_type == PopulationType.Realistic:
        target = np.random.normal(
            mean,
            mean * coeff_of_var
        )
    elif population_type == PopulationType.HighVariation:
        target = np.random.uniform(
            range_vals[0],
            range_vals[1]
        )

    return target
