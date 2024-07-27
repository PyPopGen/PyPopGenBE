import numpy as np
from scipy.stats import gmean


def generate_stats(x: np.ndarray) -> dict:
    """
    Generates some summary stats.

    Parameters:
    x (np.ndarray): Input array of shape (n_rows, n_cols).

    Returns:
    dict: Dictionary containing summary statistics:
          - 'Mean': Arithmetic mean along each column.
          - 'StdDev': Standard deviation along each column.
          - 'GeoMean': Geometric mean along each column.
          - 'GeoStdDev': Geometric standard deviation along each column.
          - 'P2pt5': 2.5th percentile along each column.
          - 'P5': 5th percentile along each column.
          - 'Median': Median along each column.
          - 'P95': 95th percentile along each column.
          - 'P97pt5': 97.5th percentile along each column.
    """
    if len(x.shape) == 1:
        n_rows = x.shape[0]
        n_cols = 1
    else:
        n_rows, n_cols = x.shape

    if n_rows == 0:
        nan = np.full((1, n_cols), np.nan)
        stats = {
            'Mean': nan,
            'StdDev': nan,
            'GeoMean': nan,
            'GeoStdDev': nan,
            'P2pt5': nan,
            'P5': nan,
            'Median': nan,
            'P95': nan,
            'P97pt5': nan
        }
    elif n_rows == 1:
        zero = np.zeros((1, n_cols))
        one = np.ones((1, n_cols))
        stats = {
            'Mean': x,
            'StdDev': zero,
            'GeoMean': x,
            'GeoStdDev': one,
            'P2pt5': x,
            'P5': x,
            'Median': x,
            'P95': x,
            'P97pt5': x
        }
    else:
        mean = np.mean(x, axis=0)
        std_dev = np.std(x, axis=0, ddof=1)
        geo_mean = gmean(x, axis=0)
        geo_std_dev = np.exp(np.std(np.log(x), axis=0))
        prc = np.percentile(x, [2.5, 5, 50, 95, 97.5], axis=0)

        stats = {
            'Mean': mean,
            'StdDev': std_dev,
            'GeoMean': geo_mean,
            'GeoStdDev': geo_std_dev,
            'P2pt5': prc[0],
            'P5': prc[1],
            'Median': prc[2],
            'P95': prc[3],
            'P97pt5': prc[4]
        }

    return stats
