import numpy as np
from typing import Union, Dict, Tuple
from pypopgenbe.impl.calculatebrainmass import calculate_brain_mass
from pypopgenbe.impl.calculateskinmass import calculate_skin_mass
from pypopgenbe.impl.calculatemusclemassadjustmentfactor import calculate_muscle_mass_adjustment_factor
from pypopgenbe.impl.calculatebonemassadjustmentfactor import calculate_bone_mass_adjustment_factor


def calculate_target_organ_mass(
    age: float,
    sex_name: str,
    ethnicity_name: str,
    mean_body_weight: float,
    mean_height: float,
    target_body_weight: float,
    target_height: float,
    organ_masses: np.ndarray,
    index: Dict[str, Union[int, list]]
) -> Tuple[float, np.ndarray]:
    # Calculate scaled height using the allometric scaling factor
    scaled_height = (target_height / mean_height) ** 0.75

    # Allometric scaling of all compartments
    target_organ_mass = scaled_height * organ_masses * mean_body_weight

    # Overwrite specific organ masses with calculated values
    target_organ_mass[index["Brain"]] = calculate_brain_mass(age, sex_name)

    target_organ_mass[index["Muscle"]
                      ] *= calculate_muscle_mass_adjustment_factor(age, sex_name)

    target_organ_mass[index["Bone"]
                      ] *= calculate_bone_mass_adjustment_factor(age, sex_name, ethnicity_name)

    target_organ_mass[index["Skin"]] = calculate_skin_mass(
        target_body_weight, sex_name)

    # Calculate adipose mass as the remaining mass
    target_organ_mass[index["Adipose"]] = target_body_weight - \
        (np.sum(target_organ_mass) - target_organ_mass[index["Adipose"]])

    return scaled_height, target_organ_mass
