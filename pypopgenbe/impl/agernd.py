import numpy as np
from pypopgenbe.impl.agecdf import age_cdf
from pypopgenbe.impl.ageinv import age_inv


def age_rnd(lower: float, upper: float) -> float:
    """
    Randomly generate an age.

    Parameters
    ----------
    lower : float
        Lower bound for age.
    upper : float
        Upper bound for age.

    Returns
    -------
    r : float
        Randomly generated age.
    """
    bounds = age_cdf(np.array([lower, upper]))
    quantile = np.random.uniform(bounds[0], bounds[1])
    r = age_inv(quantile)
    return r
