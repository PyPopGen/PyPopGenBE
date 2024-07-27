from typing import Tuple
import numpy as np


def create_ethnicity_breaks(probs_of_ethnicities: Tuple[float, float, float]) -> np.ndarray:
    """
    Creates the breaks in the ethnicity vector used by AssignEthnicity.

    Args:
    - probs_of_ethnicities: Tuple of probabilities of particular ethnicities.

    Returns:
    - ethnicity_breaks: Array of cumulative sums, suitable for use with AssignEthnicity.
    """
    # Validate inputs (similar to MATLAB validateattributes)
    assert isinstance(probs_of_ethnicities,
                      tuple), "probs_of_ethnicities must be a tuple"
    assert len(
        probs_of_ethnicities) == 3, "probs_of_ethnicities must have exactly 3 elements"
    assert all(isinstance(prob, (int, float)) and prob >= 0 for prob in probs_of_ethnicities), \
        "All probabilities must be non-negative numbers"

    # Calculate cumulative sums
    ethnicity_breaks = np.cumsum(probs_of_ethnicities)

    # Append a small epsilon value to ensure the last break is slightly greater than 1
    ethnicity_breaks = np.append(ethnicity_breaks, 1.0 + np.finfo(float).eps)

    return ethnicity_breaks
