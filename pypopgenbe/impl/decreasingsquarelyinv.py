import numpy as np


def decreasing_squarely_inv(p: float, lower: float, upper: float) -> float:
    """
    Calculate the inverse cumulative distribution function for the "decreasing squarely" distribution.

    :param p: Cumulative probability at which to evaluate the inverse CDF
    :param lower: Lower bound of the distribution
    :param upper: Upper bound of the distribution
    :return: Quantile corresponding to the given cumulative probability
    """
    if p == 0.:
        return lower
    if p == 1.:
        return upper

    lower3 = lower ** 3.
    const_of_integration = lower3 + 3. * \
        lower ** 2. * upper + 3. * lower * upper ** 2.
    sum_of_bounds = lower + upper
    diff_cubes = upper ** 3. - lower3

    coeff3 = 1.
    coeff2 = -3. * sum_of_bounds
    coeff1 = 3. * sum_of_bounds ** 2.
    coeff0 = -const_of_integration - diff_cubes * p

    cubic = [coeff3, coeff2, coeff1, coeff0]
    roots = np.roots(cubic)
    real_roots = roots[np.isreal(roots)].real

    return real_roots[0]  # Assuming a single real root
