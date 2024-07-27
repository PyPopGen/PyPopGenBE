import numpy as np
from typing import cast


def assign_ethnicity(ethnicity_breaks: np.ndarray) -> int:
    """
    Assigns an ethnicity based on the provided break points.

    Parameters:
    ethnicity_breaks (ndarray): A list of break points that determine the probability ranges for each ethnicity.

    Returns:
    int: The assigned ethnicity.
    """
    random_value = np.random.rand()
    ethnicity = np.searchsorted(ethnicity_breaks, random_value) + 1
    return cast(int, ethnicity)
