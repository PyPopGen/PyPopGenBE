import numpy as np
from typing import cast
from scipy.interpolate import interp1d

CONSTS_decline_ages = np.array([25, 35, 45, 55, 65, 75, 85])
CONSTS_male_bone_density = np.array(
    [1.203, 1.214, 1.200, 1.186, 1.161, 1.14, 1.118])
CONSTS_male_bone_density_black = np.array(
    [1.297, 1.294, 1.258, 1.244, 1.245, 1.18, 1.177])
CONSTS_male_bone_density_non_black_hispanic = np.array(
    [1.17, 1.156, 1.149, 1.146, 1.135, 1.118, 1.074])
CONSTS_female_bone_density = np.array(
    [1.104, 1.129, 1.118, 1.090, 1.032, 0.982, 0.917])
CONSTS_female_bone_density_black = np.array(
    [1.191, 1.194, 1.191, 1.133, 1.093, 1.023, 0.978])
CONSTS_female_bone_density_non_black_hispanic = np.array(
    [1.104, 1.115, 1.104, 1.104, 0.989, 0.939, 0.884])

male_bone_density = CONSTS_male_bone_density / CONSTS_male_bone_density[0]
male_bone_density_black = CONSTS_male_bone_density_black / \
    CONSTS_male_bone_density_black[0]
male_bone_density_non_black_hispanic = CONSTS_male_bone_density_non_black_hispanic / \
    CONSTS_male_bone_density_non_black_hispanic[0]
female_bone_density = CONSTS_female_bone_density / \
    CONSTS_female_bone_density[0]
female_bone_density_black = CONSTS_female_bone_density_black / \
    CONSTS_female_bone_density_black[0]
female_bone_density_non_black_hispanic = CONSTS_female_bone_density_non_black_hispanic / \
    CONSTS_female_bone_density_non_black_hispanic[0]


def calc_adjustment_factor(p: float) -> float:
    bone_mineral_fraction = 0.44
    return 1 + p * bone_mineral_fraction


def calculate_bone_mass_adjustment_factor(age_years: float, sex_name: str, ethnicity_name: str) -> float:

    is_black = ethnicity_name.lower() == 'black'
    is_non_black_hispanic = ethnicity_name.lower() == 'nonblackhispanic'

    if sex_name.lower() == 'male':
        if age_years >= 20 and age_years <= 24:
            age_adjustment_factor_young = calc_adjustment_factor(
                0.05 - 0.01 * (24 - age_years))
        elif age_years >= 24:
            age_adjustment_factor_young = calc_adjustment_factor(0.05)
        else:
            age_adjustment_factor_young = 1.

        if is_black:
            ethnicity_adjustment_factor = calc_adjustment_factor(0.08)
        elif is_non_black_hispanic:
            ethnicity_adjustment_factor = calc_adjustment_factor(-0.03)
        else:
            ethnicity_adjustment_factor = 1.

        if is_black and age_years >= 25.:
            age_adjustment_factor_old = calc_adjustment_factor(
                interp1d(CONSTS_decline_ages, male_bone_density_black,
                         fill_value=cast(float, "extrapolate"))(age_years) - 1.
            )
        elif is_non_black_hispanic and age_years >= 25.:
            age_adjustment_factor_old = calc_adjustment_factor(
                interp1d(CONSTS_decline_ages, male_bone_density_non_black_hispanic,
                         fill_value=cast(float, "extrapolate"))(age_years) - 1.
            )
        elif age_years >= 25.:
            age_adjustment_factor_old = calc_adjustment_factor(
                interp1d(CONSTS_decline_ages, male_bone_density,
                         fill_value=cast(float, "extrapolate"))(age_years) - 1.
            )
        else:
            age_adjustment_factor_old = 1.

    else:  # Female
        if age_years >= 18 and age_years <= 22:
            age_adjustment_factor_young = calc_adjustment_factor(
                0.05 - 0.01 * (22 - age_years))
        elif age_years >= 22:
            age_adjustment_factor_young = calc_adjustment_factor(0.05)
        else:
            age_adjustment_factor_young = 1.

        if is_black:
            ethnicity_adjustment_factor = calc_adjustment_factor(0.055)
        elif is_non_black_hispanic:
            ethnicity_adjustment_factor = calc_adjustment_factor(-0.01)
        else:
            ethnicity_adjustment_factor = 1.

        if is_black and age_years >= 25.:
            age_adjustment_factor_old = calc_adjustment_factor(
                interp1d(CONSTS_decline_ages, female_bone_density_black,
                         fill_value=cast(float, "extrapolate"))(age_years) - 1.
            )
        elif is_non_black_hispanic and age_years >= 25.:
            age_adjustment_factor_old = calc_adjustment_factor(
                interp1d(CONSTS_decline_ages, female_bone_density_non_black_hispanic,
                         fill_value=cast(float, "extrapolate"))(age_years) - 1.
            )
        elif age_years >= 25.:
            age_adjustment_factor_old = calc_adjustment_factor(
                interp1d(CONSTS_decline_ages, female_bone_density,
                         fill_value=cast(float, "extrapolate"))(age_years) - 1.
            )
        else:
            age_adjustment_factor_old = 1.

    bone_mass_adjustment_factor = age_adjustment_factor_young * \
        age_adjustment_factor_old * ethnicity_adjustment_factor

    return bone_mass_adjustment_factor
