import numpy as np


def age_cdf(x: np.ndarray) -> np.ndarray:
    """
    CDF for human ages.

    This function was determined by taking a LOESS smooth of the UK 2011
    census data, then making the model even cruder (so it applies more
    generally than to the UK at the present time).  Essentially, the
    population is evenly spread up to the age of 45, after which it declines
    in a linear fashion down to zero at age 100.  This should reasonable for
    most developed countries.

    Parameters
    ----------
    x : np.ndarray or list of float or float
        Input age values.

    Returns
    -------
    p : np.ndarray
        Cumulative distribution function values for the input ages.
    """
    x = np.asarray(x)
    mid_age = 45
    max_age = 100

    h = 1 / (mid_age + 0.5 * (max_age - mid_age))

    p = np.full_like(x, np.nan, dtype=float)

    p[x <= 0] = 0
    p[x >= max_age] = 1

    young = (x > 0) & (x <= mid_age)
    p[young] = h * x[young]

    old = (x > mid_age) & (x <= max_age)
    p[old] = h / (max_age - mid_age) * np.polyval([-0.5, 100, -1012.5], x[old])

    return p
