from typing import List
import numpy as np
import numpy.typing as npt


def indices_of_organs(organs_of_interest: List[str], all_organs: List[str]) -> npt.NDArray[np.int_]:
    """
    Returns indices of organs.

    Parameters:
    organs_of_interest (List[str]): List of organs of interest.
    all_organs (List[str]): List of all possible organs.

    Returns:
    np.ndarray: Indices of the organs of interest in the list of all organs.
    """
    indices = [all_organs.index(organ)
               for organ in organs_of_interest if organ in all_organs]
    return np.array(indices)
