import numpy as np
import numpy.typing as npt


def invert_indices(n: int, index_to_remove: npt.NDArray[np.integer]) -> np.ndarray:
    """
    Creates an index for an array based upon what isn't passed in.

    Parameters:
    n (int): The length of the array.
    index_to_remove (List[int]): The indices to remove.

    Returns:
    np.ndarray: The indices to keep.

    Example:
    invert_indices(10, [3, 5, 6, 8])  # returns [0, 1, 2, 4, 7, 9]
    """
    return np.setdiff1d(np.arange(0, n), index_to_remove)
