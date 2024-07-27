import numpy as np
from typing import Union


def lognrnd0(mean: Union[float, np.ndarray],
             coeffOfVar: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Generates lognormally distributed random numbers based on the mean
    and coefficient of variation.

    Parameters:
    mean (float or np.ndarray): The mean of the distribution.
    coeffOfVar (float or np.ndarray): The coefficient of variation.
    sizeOut (tuple): The shape of the output array. Default is (1, 1).

    Returns:
    float or np.ndarray: Lognormally distributed random number(s).
    """
    sigma = np.sqrt(np.log(coeffOfVar**2. + 1.))
    mu = np.log(mean) - sigma**2. / 2.
    samples = np.exp(np.random.randn(*mu.shape) * sigma + mu)

    if isinstance(samples, float):
        return samples

    if len(samples) == 1:
        return samples[0]

    return samples
