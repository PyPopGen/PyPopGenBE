import numpy as np


def calculate_in_vivo_enzyme_rate(in_vitro_enzyme_rate: np.ndarray,
                                  mppgl: np.ndarray,
                                  liver_mass: np.ndarray) -> np.ndarray:
    """
    Calculates the in-vivo enzyme rate.

    Parameters:
    in_vitro_enzyme_rate: np.ndarray
        In-vitro enzyme rate in pmol/min/g (Vmax) or mL/min/g (CLint)
    mppgl: np.ndarray
        MPPGL in mg/g
    liver_mass: np.ndarray
        Liver mass in kg

    Returns:
    np.ndarray
        In-vivo enzyme rate in pmol/min or mL/min
    """

    if in_vitro_enzyme_rate is None or len(in_vitro_enzyme_rate) == 0:
        return np.empty((len(mppgl), 0))

    x, y = np.meshgrid(in_vitro_enzyme_rate, np.multiply(mppgl, liver_mass))
    in_vivo_enzyme_rate = np.multiply(x, y)

    return in_vivo_enzyme_rate
