def calculate_surface_area_costeff(body_mass_kg: float) -> float:
    """
    Calculate the surface area using the Costeff (1966) formula.

    :param body_mass_kg: Body mass in kilograms
    :return: Surface area in square meters
    """
    return (4. * body_mass_kg + 7.) / (body_mass_kg + 90.)


def calculate_skin_mass(body_mass_kg: float, sex_name: str) -> float:
    """
    Calculates the mass of skin for a person.

    The skin mass thickness is from ICRP 2003, section 10.3.
    A weighted average of the skin mass thickness for head and trunk,
    upper arms and legs and lower arms and legs is taken.

    :param body_mass_kg: Body mass in kilograms
    :param sex_name: Sex of the person ('Male' or 'Female')
    :return: Mass of skin in kilograms
    """
    if sex_name == 'Male':
        surface_thickness = 0.42 * 0.225 + 0.29 * 0.135 + 0.29 * 0.140
    else:
        surface_thickness = 0.42 * 0.180 + 0.29 * 0.110 + 0.29 * 0.115

    skin_surface_area_m2 = calculate_surface_area_costeff(body_mass_kg)
    skin_mass_kg = skin_surface_area_m2 * surface_thickness * 10.

    return skin_mass_kg
