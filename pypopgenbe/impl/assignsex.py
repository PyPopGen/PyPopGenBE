import numpy as np


def assign_sex(prob_of_male: float) -> int:
    """
    Assigns a sex based on the probability of being male.

    Parameters:
    prob_of_male (float): Probability of returning male.

    Returns:
    int: Assigned sex (1=male, 2=female).
    """
    return 1 + (np.random.rand() > prob_of_male)
