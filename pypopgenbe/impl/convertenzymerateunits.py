import numpy as np
from typing import Union, Optional
from pypopgenbe.impl.enum import EnzymeRateCLintUnits, EnzymeRateVmaxUnits


def convert_enzyme_rate_units(x: np.ndarray, new_units: Union[EnzymeRateVmaxUnits, EnzymeRateCLintUnits], molecular_weight: Optional[float] = None) -> np.ndarray:
    """
    Converts enzyme rate values from one unit to another.

    Parameters:
    x (np.ndarray): The enzyme rate values to convert.
    new_units (str): The new units for the enzyme rate values.
    molecular_weight (float, optional): The molecular weight necessary for conversion to units involving grams. Defaults to NaN.

    Returns:
    np.ndarray: The converted enzyme rate values.
    """

    def mols_to_grams(x: np.ndarray, molecular_weight: float) -> np.ndarray:
        return x * molecular_weight

    def x_per_minute_to_one_thousand_x_per_hour(x: np.ndarray) -> np.ndarray:
        return 0.06 * x

    def x_per_minute_to_one_million_x_per_hour(x: np.ndarray) -> np.ndarray:
        return 6e-5 * x

    if new_units in {EnzymeRateVmaxUnits.PicoMolsPerMinute, EnzymeRateCLintUnits.MicroLitresPerMinute}:
        return x
    elif new_units in {EnzymeRateVmaxUnits.MicroMolsPerHour, EnzymeRateCLintUnits.MilliLitresPerHour}:
        return x_per_minute_to_one_thousand_x_per_hour(x)
    elif new_units in {EnzymeRateVmaxUnits.MilliMolsPerHour, EnzymeRateCLintUnits.LitresPerHour}:
        return x_per_minute_to_one_million_x_per_hour(x)
    
    if molecular_weight is None:
        raise ValueError(f"New unit {new_units} requires value for molecular mass")

    if new_units == EnzymeRateVmaxUnits.PicoGramsPerMinute:
        return mols_to_grams(x, molecular_weight)
    elif new_units == EnzymeRateVmaxUnits.MicroGramsPerHour:
        return mols_to_grams(x_per_minute_to_one_thousand_x_per_hour(x), molecular_weight)
    elif new_units == EnzymeRateVmaxUnits.MilliGramsPerHour:
        return mols_to_grams(x_per_minute_to_one_million_x_per_hour(x), molecular_weight)
    else:
        raise ValueError(f"Unknown unit: {new_units}")
