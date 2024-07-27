def calculate_bmi(mass: float, height: float) -> float:
    """
    Calculates body mass index.

    :param mass: Mass in kilograms.
    :param height: Height in centimeters.
    :return: Body mass index in kg/m^2.
    """

    if mass <= 0 or height <= 0:
        raise ValueError("Mass and height must be positive values.")

    bmi = mass / (0.01 * height) ** 2.
    return bmi
