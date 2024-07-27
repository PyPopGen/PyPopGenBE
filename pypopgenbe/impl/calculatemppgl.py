import numpy as np


def calculate_mppgl(age: np.ndarray, include_variation: bool = True) -> np.ndarray:
    """
    Calculates microsomal protein per gram of liver (MPPGL), dependent upon age.

    NOTE: The formula was taken from Barter et al 2007, Fig 4.
    http://www.ncbi.nlm.nih.gov/pubmed/17266522
    The standard deviation in the variability was calculated in 
    "<project root>\\Enzymology\\Calculate variability for
    MPPGL.r".
    Variability is normally distributed (but the value is forced to be
    positive).

    Parameters:
    age (np.ndarray): The age(s) to calculate MPPGL for.
    include_variation (bool): Whether to include variation in the calculation.

    Returns:
    np.ndarray: The calculated MPPGL values.
    """

    if include_variation:
        variation = np.random.randn(*age.shape) * 0.178
    else:
        variation = 0.

    mppgl = np.maximum(np.sqrt(np.finfo(float).eps), 10. **
                       (-0.3 * np.log10(age) + 2.04 + variation))

    return mppgl
