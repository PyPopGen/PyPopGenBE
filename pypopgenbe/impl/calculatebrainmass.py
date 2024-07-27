import numpy as np


def calculate_brain_mass(age_years: float, sex_name: str) -> float:
    """
    Calculates average brain mass in humans.

    Parameters:
    age_years (float): Age in years.
    sex_name (str): Sex of the individual ('Male' or 'Female').

    Returns:
    float: Average brain mass.
    """
    # Equations from Bosgra et al, 2012
    if sex_name == 'Male':
        brain = 0.405 * ((3.68 - 2.68 * np.exp(-age_years / 0.89))
                         * np.exp(-age_years / 629))
    else:  # Female
        brain = 0.373 * ((3.68 - 2.68 * np.exp(-age_years / 0.89))
                         * np.exp(-age_years / 629))

    return brain
