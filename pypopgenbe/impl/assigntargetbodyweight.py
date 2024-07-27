from typing import Tuple
from pypopgenbe.impl.truncatednormrnd import truncated_norm_rnd
from pypopgenbe.impl.decreasingsquarelyrnd import decreasing_squarely_rnd
from pypopgenbe.impl.enum import PopulationType

def assign_target_body_weight(
    population_type: PopulationType,
    range_vals: Tuple[float, float],
    mean: float,
    coeff_of_var: float
) -> float:
    """
    Assigns a random target body weight.

    Parameters:
    population_type (str): Either 'Realistic' or 'HighVariation'.
    range_vals (Tuple[float, float]): Numeric vector of length two, giving the lower and upper bounds of the target weight.
    mean (float): The mean weight of the population to sample from.
    coeff_of_var (float): The coefficient of variation of population weights.

    Returns:
    float: Target body weight.
    """

    if population_type == PopulationType.Realistic:
        target = truncated_norm_rnd(
            mean,
            mean * coeff_of_var,
            range_vals[0],
            range_vals[1]
        )
    elif population_type == PopulationType.HighVariation:
        target = decreasing_squarely_rnd(
            range_vals[0],
            range_vals[1]
        )

    return target
