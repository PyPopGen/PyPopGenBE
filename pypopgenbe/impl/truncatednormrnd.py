from typing import cast
from scipy.stats import norm, uniform


def truncated_norm_rnd(mu: float, sigma: float,
                       lower: float, upper: float) -> float:
    """
    Generate random numbers from a truncated normal distribution.

    Parameters:
    mu (float): Mean of the normal distribution.
    sigma (float): Standard deviation of the normal distribution.
    lower (float): Lower truncation bound.
    upper (float): Upper truncation bound.

    Returns:
    float: Random number from the truncated normal distribution.
    """

    # Calculate the CDF bounds
    bounds = norm.cdf([lower, upper], mu, sigma)

    # Sample uniformly within the CDF bounds and transform back to normal
    quantile = uniform.rvs(bounds[0], bounds[1] - bounds[0])
    r = norm.ppf(quantile, mu, sigma)

    return cast(float, r)
