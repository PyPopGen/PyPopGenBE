import numpy as np

def assign_sex(prob_of_male, n=1, debug=False):
    """
    Assigns a sex.
    
    SEX = assign_sex(prob_of_male) assigns a sex, with a PROB_OF_MALE
    probability of returning male.
    
    SEX = assign_sex(prob_of_male, n) assigns N sexes, each with a PROB_OF_MALE
    probability of returning male.
    
    Key: 1=male 2=female

    Parameters:
    prob_of_male (float): Probability of being male.
    n (int): Number of sexes to assign. Default is 1.
    debug (bool): Enable debug mode for argument validation. Default is False.
    
    Returns:
    numpy.ndarray: Array of assigned sexes (1=male, 2=female).
    """
    
    if debug:
        if not isinstance(prob_of_male, (int, float)) or not (0 <= prob_of_male <= 1):
            raise ValueError("prob_of_male must be a non-negative scalar between 0 and 1")
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive scalar integer")
    
    sexes = 1 + (np.random.rand(n) > prob_of_male)
    return sexes

# Example usage:
# sexes = assign_sex(0.5, 10)
# print(sexes)
