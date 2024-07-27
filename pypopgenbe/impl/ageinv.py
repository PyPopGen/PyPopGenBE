import numpy as np


def age_inv(p: float) -> float:
    """
    Inverse CDF for human ages.

    Parameters
    ----------
    p : float
        Input probability values.

    Returns
    -------
    x : float
        Inverse cumulative distribution function values for the input probability.
    """

    if p == 0.:
        return 0.

    mid_age = 45.
    max_age = 100.

    if p == 1.:
        return max_age

    h = 1 / (mid_age + 0.5 * (max_age - mid_age))
    m = h * mid_age

    if p <= m:
        return p / h

    return max_age - np.sqrt(7975. * (1. - p))
