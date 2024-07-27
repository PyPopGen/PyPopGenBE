def calculate_mass(bmi: float, height: float) -> float:
    """
    Calculates body mass.

    Parameters:
    bmi (float): Body Mass Index in kg/m^2
    height (float): Height in cm

    Returns:
    float: Body mass in kg
    """
    mass = bmi * (0.01 * height) ** 2
    return mass
