import numpy as np
from pypopgenbe.impl.enum import FlowUnits


def convert_flow_units(x: np.ndarray, new_units: FlowUnits) -> np.ndarray:
    """
    Converts flow values from 'MilliLitresPerMinute' to 'MilliLitresPerMinute' or 'LitresPerHour'.

    Parameters:
    x (float): Flow value to convert.
    new_units (str): Target units to convert to. Can be 'MilliLitresPerMinute' or 'LitresPerHour'.

    Returns:
    float: Converted flow value.

    Raises:
    ValueError: If an unknown unit is provided.
    """
    if new_units == FlowUnits.MilliLitresPerMinute:
        return x
    elif new_units == FlowUnits.LitresPerHour:
        return to_litres_per_hour(x)


def to_litres_per_hour(x: np.ndarray) -> np.ndarray:
    """
    Converts flow from MilliLitresPerMinute to LitresPerHour.

    Parameters:
    x (float): Flow value in MilliLitresPerMinute.

    Returns:
    float: Converted flow value in LitresPerHour.
    """
    return 0.06 * x
