import pandas as pd
import numpy as np
from typing import List, Dict, Union, cast
from pathlib import Path


def find_str(array: List[str], pattern: str) -> List[bool]:
    return [pattern in element for element in array]


def read_coeffs_from_file(path: Path, ethnicities: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    # Read the CSV file
    data = pd.read_csv(path)

    # Find the row and column indices
    col_headers = list(data.columns)
    _ = [i for i, col in enumerate(col_headers) if 'female' in col] + \
        [i for i, col in enumerate(col_headers) if 'male' in col]
    _ = {ethnicity: [i for i, col in enumerate(
        col_headers) if ethnicity in col] for i, ethnicity in enumerate(ethnicities)}

    # Initialize the coefficients dictionary
    coeffs: dict[str, dict[str, Union[None, np.ndarray]]] = {
        'female': {ethnicity.replace(" ", ""): None for ethnicity in ethnicities},
        'male': {ethnicity.replace(" ", ""): None for ethnicity in ethnicities}
    }

    # Assign data to the coefficients dictionary
    for i in range(len(col_headers)):
        for gender in ['female', 'male']:
            for ethnicity in ethnicities:
                if gender in col_headers[i] and ethnicity in col_headers[i]:
                    coeffs[gender][ethnicity.replace(" ", "")] = np.asarray(
                        data.iloc[:, i].values)

    return cast(Dict[str, Dict[str, np.ndarray]], coeffs)
