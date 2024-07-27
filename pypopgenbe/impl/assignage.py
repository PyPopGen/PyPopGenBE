import numpy as np
from typing import Tuple
from pypopgenbe.impl.agernd import age_rnd
from pypopgenbe.impl.enum import PopulationType

def assign_age(population_type: PopulationType, age_range: Tuple[float, float]) -> float:
    """
    Assigns an age.

    Parameters:
    population_type: str
        Either 'Realistic' or 'HighVariation'.
    age_range: Tuple[int, int]
        Lower and upper values of the age range.

    Returns:
    float
        Assigned age.

    Raises:
    ValueError
        If population_type is not supported or input validation fails.

    Examples:
    >>> r = assign_age('Realistic', (16, 80))
    >>> hv = assign_age('HighVariation', (16, 80))
    """

    def unifrnd(low: float, high: float) -> float:
        # Generate a uniform distribution of ages
        return np.random.uniform(low, high)

    # Input validation
    if population_type not in [PopulationType.Realistic, PopulationType.HighVariation]:
        raise ValueError("The 'population_type' is not supported.")

    if any(age <= 0 for age in age_range):
        raise ValueError("'age_range' values must be positive.")

    # Function selection based on population type
    if population_type == PopulationType.Realistic:
        fn = age_rnd
    elif population_type == PopulationType.HighVariation:
        fn = unifrnd

    # Generate ages
    pop = fn(age_range[0], age_range[1])
    age = np.maximum(np.sqrt(np.finfo(float).eps), pop)

    return age
