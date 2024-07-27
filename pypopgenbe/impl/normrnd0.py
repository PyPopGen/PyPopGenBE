import numpy as np
from typing import Union, cast


def normrnd0(mean: Union[float, np.ndarray],
             coeffOfVar: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Generates normally distributed random numbers based on the mean
    and coefficient of variation.

    Parameters:
    mean (float or np.ndarray): The mean of the distribution.
    coeffOfVar (float or np.ndarray): The coefficient of variation.
    sizeOut (tuple): The shape of the output array. Default is (1, 1).

    Returns:
    np.ndarray: Array of normally distributed random numbers.
    """
    if isinstance(mean, float):
        return (np.random.randn() * coeffOfVar + 1.) * mean

    mean = cast(np.ndarray, mean)
    return (np.random.randn(*mean.shape) * coeffOfVar + 1.) * mean
