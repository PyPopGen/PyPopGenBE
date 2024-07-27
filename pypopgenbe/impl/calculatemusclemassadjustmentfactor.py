male_coeff = -0.006181912
female_coeff = -0.005522342


def calculate_muscle_mass_adjustment_factor(age_years: float, sex: str) -> float:
    """
    Adjusts muscle mass by age and sex.

    Based upon Janssen et al. 2000.  Skeletal muscle mass and distribution
    in 468 men and women aged 18-88 yr.  Journal of Applied Physiology 89:
    81-88.

    Parameters:
        age_years (float): Ages of individual.
        sex (str): Sex of individual ('Male' or 'Female').

    Returns:
        float: Adjustment factor for muscle mass.
    """

    old = age_years > 45

    if not old:
        return 1.

    male = sex.lower() == 'male'

    if male:
        return 1. + (male_coeff * (age_years - 45))

    return 1. + (female_coeff * (age_years - 45))
