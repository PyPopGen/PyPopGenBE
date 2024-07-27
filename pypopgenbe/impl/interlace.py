import numpy as np


def interlace(*args: np.ndarray) -> np.ndarray:
    """
    Interlaces vectors.

    Takes any number of vectors with the same length and interlaces them, 
    returning a column vector.

    Examples:
    interlace(np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9]))

    Returns:
    np.ndarray: A column vector with interlaced values from input vectors.
    """
    # Check that all inputs are vectors and have the same length
    lengths = [len(arg) for arg in args]
    if not all([arg.ndim == 1 for arg in args]) or len(set(lengths)) != 1:
        raise ValueError('The inputs are not vectors of the same length')

    # Interlace the vectors
    y = np.vstack([to_row_vector(arg) for arg in args]).T.flatten()

    return y.reshape(-1, 1)


def to_row_vector(x: np.ndarray) -> np.ndarray:
    """
    Converts a vector to a row vector if it is a column vector.

    Parameters:
    x (np.ndarray): Input vector.

    Returns:
    np.ndarray: Row vector.
    """
    return x if x.ndim == 1 else x.flatten()
