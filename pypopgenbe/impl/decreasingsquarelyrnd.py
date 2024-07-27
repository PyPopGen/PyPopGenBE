import numpy as np
from pypopgenbe.impl.decreasingsquarelyinv import decreasing_squarely_inv


def decreasing_squarely_rnd(lower: float, upper: float) -> float:
    """
    Generate random number sampled from the "decreasing squarely" distribution.

    :param lower: Lower bound of the distribution
    :param upper: Upper bound of the distribution
    :return: Random sample from the distribution
    """

    quantile = np.random.random_sample()
    r = decreasing_squarely_inv(quantile, lower, upper)

    return r
