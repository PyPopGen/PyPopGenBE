from typing import Tuple, Dict, Optional, Union
from pypopgenbe.impl.enum import EnzymeRateCLintUnits, EnzymeRateVmaxUnits, PopulationType, Dataset, FlowUnits


def collate_inputs(
    population_size: int, 
    dataset: Dataset, 
    population_type: PopulationType, 
    seed: int,
    age_range: Tuple[float, float], 
    bmi_range: Tuple[float, float], 
    height_range: Tuple[float, float],
    prob_of_male: float, 
    probs_of_ethnicities: Optional[Tuple[float, float, float]],
    flow_units: FlowUnits, 
    enzyme_rate_units: Union[EnzymeRateVmaxUnits, EnzymeRateCLintUnits, None]
) -> Dict:
    """
    Collates the inputs to generatepop() into a dictionary suitable for writing to XML.

    Args:
        population_size (int): Size of the population.
        dataset_name (str): Name of the dataset.
        population_type (str): Type of the population.
        seed (int): Seed for random number generation.
        age_range (Tuple[float, float]): Age range.
        bmi_range (Tuple[float, float]): BMI range.
        height_range (Tuple[float, float]): Height range.
        prob_of_male (float): Probability of being male.
        probs_of_ethnicities (Tuple[float, float, float]): Probabilities of ethnicities.
        flow_units (str): Units for flow.
        enzyme_rate_units (str): Units for enzyme rate.

    Returns:
        Dict: Collated inputs structured in a dictionary.
    """
    inputs = {
        'Population': {
            'Size': population_size,
            'Dataset': dataset.name,
            'Type': population_type.name,
            'Seed': seed,
        },
        'Filter': {
            'Age': age_range,
            'BMI': bmi_range,
            'Height': height_range,
        },
        'Probability': {
            'Male': prob_of_male,
            'Ethnicity': probs_of_ethnicities,
        },
        'Units': {
            'Flow': flow_units.name,
            'EnzymeRate': ('None' if enzyme_rate_units is None else enzyme_rate_units.value)
        }
    }

    return inputs
