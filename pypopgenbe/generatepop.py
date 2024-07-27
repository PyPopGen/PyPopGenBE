from typing import Union, List, Optional, Dict, Any, Tuple, cast, Callable
import numpy as np
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import Polynomial
from pathlib import Path
import pickle
import time

from pypopgenbe.impl.createethnicitybreaks import create_ethnicity_breaks
from pypopgenbe.impl.assignsex import assign_sex
from pypopgenbe.impl.assignage import assign_age
from pypopgenbe.impl.assignethnicity import assign_ethnicity
from pypopgenbe.impl.calculatemass import calculate_mass
from pypopgenbe.impl.assigntargetheight import assign_target_height
from pypopgenbe.impl.assigntargetbodyweight import assign_target_body_weight
from pypopgenbe.impl.calculatetargetorganmass import calculate_target_organ_mass
from pypopgenbe.impl.addstochasticvariation import add_stochastic_variation
from pypopgenbe.impl.lognrnd0 import lognrnd0
from pypopgenbe.impl.calculatebmi import calculate_bmi
from pypopgenbe.impl.aggregateorgans import aggregate_organs
from pypopgenbe.impl.convertflowunits import convert_flow_units
from pypopgenbe.impl.calculatemppgl import calculate_mppgl
from pypopgenbe.impl.varyinvitroenzymerates import vary_in_vitro_enzyme_rates
from pypopgenbe.impl.calculateinvivoenzymerate import calculate_in_vivo_enzyme_rate
from pypopgenbe.impl.convertenzymerateunits import convert_enzyme_rate_units
from pypopgenbe.impl.invertindicies import invert_indices
from pypopgenbe.impl.generatestats import generate_stats
from pypopgenbe.impl.collateinputs import collate_inputs
from pypopgenbe.impl.enum import EnzymeRateCLintUnits, EnzymeRateParameter, EnzymeRateVmaxUnits, Dataset, FlowUnits, PopulationType

THIS_DIR = Path(__file__).parent


def generate_pop(
    population_size: int,
    dataset_name: str,
    age_range: Tuple[float, float],
    bmi_range: Tuple[float, float],
    height_range: Tuple[float, float],
    prob_of_male: float,
    probs_of_ethnicities: Optional[Tuple[float, float, float]],
    is_slowly_perfused_tissue_discrete: Union[bool, List[bool]] = False,
    is_richly_perfused_tissue_discrete: Union[bool, List[bool]] = False,
    enzyme_rate_parameter: Optional[Union[str, EnzymeRateParameter]] = None,
    enzyme_names: Optional[List[str]] = None,
    in_vitro_enzyme_rates: Optional[List[float]] = None,
    in_vitro_enzyme_rate_coeffs_of_var: Optional[List[float]] = None,
    flow_units: Union[str, FlowUnits] = FlowUnits.MilliLitresPerMinute,
    enzyme_rate_units: Optional[Union[EnzymeRateVmaxUnits, EnzymeRateCLintUnits, str]] = None,
    molecular_weight: Optional[float] = None,
    seed: Optional[int] = None,
    population_type: Union[PopulationType, str] = PopulationType.Realistic,
    callback: Optional[Callable[[int, int], bool]] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    """
    Generates a population of virtual individuals with data on organ masses and flows and some enzyme abundances.

    Parameters
    ----------
    population_size : int
        The number of individuals in the population.
    dataset_name : str
        The name of the dataset to retrieve height and weight probabilities from. Supported options are 'P3M', 'ICRP', 'HSE', and 'NDNS'.
    age_range : Union[None, Tuple[float, float]]
        The population's lower and upper age limits in years.
    bmi_range : Union[None, Tuple[float, float]]
        The population's lower and upper limits for BMIs in kg/m^2.
    height_range : Union[None, Tuple[float, float]]
        The population's lower and upper limits for heights in cm.
    prob_of_male : float
        The desired proportion of the population being male.
    probs_of_ethnicities : optional[Union[None, Tuple[float, float, float]]]
        The desired proportion of the population being White/Black/Non-Black Hispanic. Values should sum to less than or equal to 1.
    is_slowly_perfused_tissue_discrete : Union[bool, List[bool]], optional
        Indicators for whether 'Adipose', 'Bone', 'Muscle', and 'Skin' should be considered discretely or aggregated. Can be a scalar to apply to all. Default is False.
    is_richly_perfused_tissue_discrete : Union[bool, List[bool]], optional
        Indicators for whether 'Brain', 'Gonads', 'Heart', 'Kidneys', 'Large intestine', 'Liver', 'Pancreas', 'Small intestine', 'Spleen', and 'Stomach' should be considered discretely or aggregated. Can be a scalar to apply to all. Default is False.
    enzyme_rate_parameter : str, optional
        The name of the parameter for enzyme rate, either Vmax (in pmol/min/mg) or CLint (in uL/min/mg)        
    enzyme_names : Union[None, List[str]], optional
        The names of the enzymes to calculate in-vivo rates for. Default is None.
    in_vitro_enzyme_rates : Union[None, List[float]], optional
        The in-vitro values for the enzyme rate parameter. Default is None.
    in_vitro_enzyme_rate_coeffs_of_var : Union[None, List[float]], optional
        Coefficients of variation for the in-vitro enzyme rates. Default is None.
    flow_units : str, optional
        The units that organ flows should be displayed in; either 'MilliLitresPerMinute' or 'LitresPerHour'.
    enzyme_rate_units : str, optional
        The units that the in-vivo enzyme rates should be displayed in.
    molecular_weight : float, optional
        The molecular weight of the substance being metabolised if enzyme_rate_parameter is 'Vmax' and enzyme_rate_units involves grams.
    seed : int, optional
        The seed for the random number generator. If None or less than zero, a seed is generated from the system time. Default is None.
    population_type : str, optional
        The type of population desired. Either 'Realistic' or 'HighVariation'. Default is 'Realistic'.
    callback : Optional[Callable[[int, int], bool]]
        Function called during generation. Parameters are number generated and number discard. Return value indicates whether or not to continue generation.

    Returns
    -------
    population : dict
        The generated population details in a structured format.
    number_of_individuals_discarded : int
        The number of individuals discarded due to out-of-range values or negative tissue masses.
    """

    if population_size < 1:
        return None, None

    with open(THIS_DIR / 'popgenconsts.pkl', 'rb') as f:
        CONSTS: Dict[str, Any] = pickle.load(f)
    
    dataset = Dataset(dataset_name) # will raise if invalid

    lower_age, upper_age = age_range
    if lower_age > upper_age:
        raise ValueError(f"Invalid age range: {lower_age} to {upper_age}")

    lower_bmi, upper_bmi = bmi_range
    if lower_bmi > upper_bmi:
        raise ValueError(f"Invalid BMI range: {lower_bmi} to {upper_bmi}")

    lower_height, upper_height = height_range
    if lower_height > upper_height:
        raise ValueError(f"Invalid height range: {lower_height} to {upper_height}")
    
    if not (0. <= prob_of_male <= 1.):
        raise ValueError(f"Probability of male out of range: {prob_of_male}")
    
    if dataset == Dataset.P3M:
        if lower_age < 0. or upper_age > 80.:
            raise ValueError("Valid age range is 0 to 80 for P3M dataset")
    elif dataset == Dataset.ICRP:
        if lower_age < 0. or upper_age > 80.:
            raise ValueError("Valid age range is 0 to 80 for ICRP dataset")
    elif dataset == Dataset.HSE:
        if lower_age < 0. or upper_age > 70.:
            raise ValueError("Valid age range is 0 to 70 for HSE dataset")
    else: # NDNS
        if lower_age < 1.25 or upper_age > 5.:
            raise ValueError("Valid age range is 1.25 to 5 for NDNS dataset")
        if probs_of_ethnicities is not None:
            raise ValueError("Ethnicity is not considered for NDNS dataset")

    if dataset != Dataset.NDNS and probs_of_ethnicities is None:
        raise ValueError(f"Ethnicities probabilities are required for {dataset_name} dataset")
    
    if probs_of_ethnicities is not None:
        if any(np.array(probs_of_ethnicities) < 0.) or sum(probs_of_ethnicities) > 1.:
            raise ValueError(f"Probabilities of ethnicities are out of range: {probs_of_ethnicities}")

    if isinstance(is_slowly_perfused_tissue_discrete, bool):
        is_slowly_perfused_tissue_discrete = [
            is_slowly_perfused_tissue_discrete] * CONSTS['NUMBER_OF_TISSUES']['SlowlyPerfused']
    else:
        if len(is_slowly_perfused_tissue_discrete) != CONSTS['NUMBER_OF_TISSUES']['SlowlyPerfused']:
            raise ValueError(f"Expecting list of {CONSTS['NUMBER_OF_TISSUES']['SlowlyPerfused']} booleans for slowly perfused discrete tissues")
    is_slowly_perfused_tissue_discrete = cast(List[bool], is_slowly_perfused_tissue_discrete)

    if isinstance(is_richly_perfused_tissue_discrete, bool):
        is_richly_perfused_tissue_discrete = [
            is_richly_perfused_tissue_discrete] * CONSTS['NUMBER_OF_TISSUES']['RichlyPerfused']
    else:
        if len(is_richly_perfused_tissue_discrete) != CONSTS['NUMBER_OF_TISSUES']['RichlyPerfused']:
            raise ValueError(f"Expecting list of {CONSTS['NUMBER_OF_TISSUES']['RichlyPerfused']} booleans for richly perfused discrete tissues")
    is_richly_perfused_tissue_discrete = cast(List[bool], is_richly_perfused_tissue_discrete)
    
    if isinstance(enzyme_rate_parameter, str):
        enzyme_rate_parameter = EnzymeRateParameter(enzyme_rate_parameter)

    if isinstance(enzyme_rate_units, str):
        if enzyme_rate_parameter == EnzymeRateParameter.CLint:
            enzyme_rate_units = EnzymeRateCLintUnits(enzyme_rate_units)
        elif enzyme_rate_parameter == EnzymeRateParameter.Vmax:
            enzyme_rate_units = EnzymeRateVmaxUnits(enzyme_rate_units)
        else:
            raise ValueError(f"Enzyme rate units specified ({enzyme_rate_units}) without specifying Vmax or CLint")
    elif enzyme_rate_parameter is not None:
        raise ValueError("Enzyme rate parameter provided without specifying unit")
    
    if isinstance(enzyme_rate_units, EnzymeRateCLintUnits) and enzyme_rate_parameter != EnzymeRateParameter.CLint:
        raise ValueError("Unexpected CLint enzyme rate unit specified")
    elif isinstance(enzyme_rate_units, EnzymeRateVmaxUnits) and enzyme_rate_parameter != EnzymeRateParameter.Vmax:
        raise ValueError("Unexpected Vmax enzyme rate unit specified")
    elif enzyme_rate_units is None and enzyme_rate_parameter is not None:
        raise ValueError("Missing enzyme rate units")
        
    if enzyme_rate_units is not None:
        if enzyme_rate_units.requires_RMM and molecular_weight is None:
            raise ValueError(f"Enzyme rate unit ({enzyme_rate_units}) requires a molecular mass value")
        elif not enzyme_rate_units.requires_RMM and molecular_weight is not None:
            raise ValueError(f"Enzyme rate unit ({enzyme_rate_units}) does not require a molecular mass value")

    if enzyme_names is None:
        enzyme_names = []
    if in_vitro_enzyme_rates is None:
        in_vitro_enzyme_rates = [1.0] * len(enzyme_names)
    if in_vitro_enzyme_rate_coeffs_of_var is None:
        in_vitro_enzyme_rate_coeffs_of_var = [np.nan] * len(enzyme_names)

    if len(enzyme_names) != len(in_vitro_enzyme_rates):
        raise ValueError("Expecting equal number of enzyme names and rates")

    if len(enzyme_names) != len(in_vitro_enzyme_rate_coeffs_of_var):
        raise ValueError("Expecting equal number of enzyme names and coefficients of variation")
    
    if len(enzyme_names) > 0 and enzyme_rate_units is None:
        raise ValueError("Missing enzyme rate units")

    if isinstance(flow_units, str):
        flow_units = FlowUnits(flow_units)

    if seed is None or seed < 0:
        seed = int(time.time())

    if isinstance(population_type, str):
        population_type = PopulationType(population_type)

    if callback is None:
        callback = lambda _,__:  False

    return _generate_pop(
        population_size,
        dataset,
        age_range,
        bmi_range,
        height_range,
        prob_of_male,
        probs_of_ethnicities,
        is_slowly_perfused_tissue_discrete,
        is_richly_perfused_tissue_discrete,
        enzyme_names,
        in_vitro_enzyme_rates,
        in_vitro_enzyme_rate_coeffs_of_var,
        flow_units,
        enzyme_rate_units,
        molecular_weight,
        seed,
        population_type,
        callback,
        CONSTS
    )


def _generate_pop(
    population_size: int,
    dataset: Dataset,
    age_range: Tuple[float, float],
    bmi_range: Tuple[float, float],
    height_range: Tuple[float, float],
    prob_of_male: float,
    probs_of_ethnicities: Optional[Tuple[float, float, float]],
    is_slowly_perfused_tissue_discrete: List[bool],
    is_richly_perfused_tissue_discrete: List[bool],
    enzyme_names: List[str],
    in_vitro_enzyme_rates: List[float],
    in_vitro_enzyme_rate_coeffs_of_var: List[float],
    flow_units: FlowUnits,
    enzyme_rate_units: Optional[Union[EnzymeRateVmaxUnits, EnzymeRateCLintUnits]],
    molecular_weight: Optional[float],
    seed: int,
    population_type: PopulationType,
    callback: Callable[[int, int], bool],
    CONSTS: Optional[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:

    # Ignore diagnostics arising from adipose calcs and columns of zeros for liver total mass and lung bronchial mass:
    # lognrnd0.py:19: RuntimeWarning: invalid value encountered in log, mu = np.log(mean) - sigma**2 / 2
    # generatestats.py:60: RuntimeWarning: divide by zero encountered in log, geo_std_dev = np.exp(np.std(np.log(x), axis=0))
    # Is this a good idea? ¯\_(ツ)_/¯
    np.seterr(invalid='ignore', divide='ignore')

    if CONSTS is None:
        with open(THIS_DIR / 'popgenconsts.pkl', 'rb') as f:
            CONSTS = cast(Dict[str, Any], pickle.load(f))

    np.random.seed(seed)

    callback_interval = min(50, max(population_size // 10, 2))

    # Generate parameters for each individual
    number_of_individuals_discarded = 0
    index_of_person = 0
    personal_details = np.zeros((population_size, 6))
    number_of_individuals_discarded = 0
    tissues = np.zeros(
        (population_size, CONSTS["NUMBER_OF_TISSUES"]["Extended"], 2)
    )
    sexes = np.zeros(population_size)
    ages = np.zeros(population_size)

    if probs_of_ethnicities is None:
        ethnicity_breaks = cast(np.ndarray, None)
    else:
        ethnicity_breaks = create_ethnicity_breaks(probs_of_ethnicities)

    key_ethnicity = {
        v: k
        for k, v in CONSTS["KEY"]["Ethnicity"][dataset.name].items()
    }

    key_sex = {
        v: k
        for k, v in CONSTS["KEY"]["Sex"].items()
    }

    while (index_of_person < population_size):

        # Assign personal details
        sex = assign_sex(prob_of_male)
        sex_name = key_sex[sex]
        sex_name_lc = sex_name.lower()

        age = assign_age(population_type, age_range)

        if age > 16 and sex_name == 'Female':
            age_class = 'Adult'
        elif age > 16 and sex_name == 'Male':
            age_class = 'Adult'
        else:
            age_class = 'Child'

        if dataset == Dataset.NDNS:
            ethnicity = 1
        else:
            ethnicity = assign_ethnicity(ethnicity_breaks)

        ethnicity_name = key_ethnicity[ethnicity]
        ethnicity_name_lc = ethnicity_name.lower()

        # Modified by kmcnally 27/06/13. Changes to mean body weight to ensure 
        # that excess mass after full height is ascribed to the adipose tissue.
        # Assign mean BodyWeight, Height and BMI
        if dataset == Dataset.ICRP:
            bw = CONSTS["BodyWeight"][dataset.name]
            h = CONSTS["Height"][dataset.name]
            mean_body_weight = interp1d(
                bw["Ages"],
                bw["Values"][:, sex-1],
                fill_value=cast(float, "extrapolate")
            )(age)
            mean_body_weight_at_maturity = interp1d(
                bw["Ages"],
                bw["Values"][:, sex-1],
                fill_value=cast(float, "extrapolate")
            )(min(age, 20 if sex_name == 'Male' else 16))
            mean_height = interp1d(
                h["Ages"],
                h["Values"][:, sex-1],
                fill_value=cast(float, "extrapolate")
            )(age)
        elif dataset == Dataset.NDNS:
            bw = CONSTS["BodyWeight"][dataset.name]
            h = CONSTS["Height"][dataset.name]
            mean_body_weight = Polynomial(
                bw[sex_name_lc][ethnicity_name_lc][::-1]
            )(age)
            mean_body_weight_at_maturity = Polynomial(
                bw[sex_name_lc][ethnicity_name_lc][::-1]
            )(min(age, 20 if sex_name == 'Male' else 16))
            mean_height = Polynomial(
                h[sex_name_lc][ethnicity_name_lc][::-1]
            )(age)
        elif dataset == Dataset.HSE:
            bw = CONSTS["BodyWeight"]["HSE"][age_class]
            h = CONSTS["Height"]["HSE"][age_class]
            mean_body_weight = Polynomial(
                bw[sex_name_lc][ethnicity_name_lc][::-1]
            )(age)
            mean_body_weight_at_maturity = Polynomial(
                bw[sex_name_lc][ethnicity_name_lc][::-1]
            )(min(age, 20 if sex_name == 'Male' else 16))
            mean_height = Polynomial(
                h[sex_name_lc][ethnicity_name_lc][::-1]
            )(age)
        else:
            bw = CONSTS["BodyWeight"]["P3M"][age_class]
            h = CONSTS["Height"]["P3M"][age_class]
            mean_body_weight = Polynomial(
                bw[sex_name_lc][ethnicity_name_lc][::-1]
            )(age)
            mean_body_weight_at_maturity = Polynomial(
                bw[sex_name_lc][ethnicity_name_lc][::-1]
            )(min(age, 20 if sex_name == 'Male' else 16))
            mean_height = Polynomial(
                h[sex_name_lc][ethnicity_name_lc][::-1]
            )(age)

        # from ICRP Ref. Man. P139
        mean_cardiac_output = interp1d(
            CONSTS["AgeGroups"]["ICRP"],
            CONSTS["CardiacOutput"]["ICRP"][:, sex-1]
        )(age)

        # Scaling of individual Mass
        body_weight_range = (
            calculate_mass(bmi_range[0], height_range[0]),
            calculate_mass(bmi_range[1], height_range[1])
        )
        target_height = assign_target_height(
            population_type,
            height_range,
            mean_height,
            CONSTS["COEFF_OF_VAR"]["Height"][sex-1]
        )
        target_body_weight = assign_target_body_weight(
            population_type,
            body_weight_range,
            mean_body_weight,
            CONSTS["COEFF_OF_VAR"]["BodyWeight"][sex-1]
        )

        scaled_height, target_organ_mass = calculate_target_organ_mass(
            age,
            sex_name,
            ethnicity_name,
            mean_body_weight_at_maturity,
            mean_height,
            target_body_weight,
            target_height,
            CONSTS["ORGAN"]["Mass"]["Mean"][sex-1],
            CONSTS["INDEX"]
        )

        # Add stochastic variation and check whether they lie within target pop
        add_stochastic_variation(
            target_organ_mass,
            CONSTS["ORGAN"]["Mass"]["CoeffOfVar"][sex-1],
            CONSTS["DISTRIBUTION"]["Mass"][sex-1]
        )

        # The mean adipose is wrong after adding stocastic variation. Recalculate.
        # Ensure mean adipose is greater than zero
        adipose_mean = target_body_weight - \
            (np.sum(target_organ_mass) -
             target_organ_mass[CONSTS["INDEX"]["Adipose"]])
        adipose_mean = max(adipose_mean, 0.01)

        # Get stochastic variation of the adipose
        temp = lognrnd0(adipose_mean, 0.42)

        # Overwrite the adipose on our individual.
        target_organ_mass[CONSTS["INDEX"]["Adipose"]] = temp

        if not all(target_organ_mass > 0):
            number_of_individuals_discarded += 1
            continue

        target_body_weight = np.sum(target_organ_mass)
        if target_organ_mass[CONSTS["INDEX"]["Adipose"]] / target_body_weight <= CONSTS["MIN_ADIPOSE_FRACTION"]:
            number_of_individuals_discarded += 1
            continue

        target_body_mass_index = calculate_bmi(target_body_weight, target_height)
        if not (bmi_range[0] <= target_body_mass_index <= bmi_range[1]):
            number_of_individuals_discarded += 1
            continue

        # Scaling of individual flows
        target_cardiac_output = scaled_height * mean_cardiac_output
        target_organ_flow = \
            CONSTS["ORGAN"]["Flow"]["Mean"][sex - 1] * target_cardiac_output
        add_stochastic_variation(
            target_organ_flow, 
            CONSTS["ORGAN"]["Flow"]["CoeffOfVar"][sex-1], 
            CONSTS["DISTRIBUTION"]["Flow"][sex-1]
        )
        
        target_cardiac_output = \
            np.sum(target_organ_flow) - target_organ_flow[CONSTS["INDEX"]["Lung"]]
        
        # Ensure Lung Flow and CO are consistent
        target_organ_flow[CONSTS["INDEX"]["Lung"]] = target_cardiac_output

        personal_details[index_of_person, :] = [
            age, 
            sex, 
            ethnicity, 
            target_body_weight, 
            target_height, 
            target_cardiac_output
        ]
        
        tissues[
            index_of_person, 
            :CONSTS["NUMBER_OF_TISSUES"]["Base"], 
            CONSTS["INDEX"]["Mass"]
        ] = target_organ_mass
        
        tissues[
            index_of_person, 
            :CONSTS["NUMBER_OF_TISSUES"]["Base"], 
            CONSTS["INDEX"]["Flow"]
        ] = target_organ_flow
        
        sexes[index_of_person] = sex
        
        ages[index_of_person] = age

        index_of_person += 1
        
        if (index_of_person + 1) % callback_interval == 0:
            if callback(index_of_person + 1, number_of_individuals_discarded):
                break

    if index_of_person < population_size:
        personal_details = np.delete(personal_details, np.s_[index_of_person:], 0)
        tissues = np.delete(tissues, np.s_[index_of_person:], 0)
        sexes = np.delete(sexes, np.s_[index_of_person:], 0)
        ages = np.delete(ages, np.s_[index_of_person:], 0)

    # Extra tissue flows formed from others
    # Combine components that feed the liver
    aggregate_organs(
        tissues,
        CONSTS["INDEX"]["LiverTotal"],
        CONSTS["INDEX"]["LiverFeeder"],
        CONSTS["INDEX"]["Flow"]
    )

    # Likewise, combine slowly/richly perfused tissues
    index_slowly_aggregated = CONSTS["INDEX"]["SlowlyPerfused"][
        np.logical_not(is_slowly_perfused_tissue_discrete)
    ]
    index_richly_aggregated = CONSTS["INDEX"]["RichlyPerfused"][
        np.logical_not(is_richly_perfused_tissue_discrete)
    ]

    aggregate_organs(
        tissues,
        CONSTS["INDEX"]["SlowlyPerfusedAggregate"],
        index_slowly_aggregated,
        CONSTS["INDEX"]["AllProperties"]
    )
    aggregate_organs(
        tissues,
        CONSTS["INDEX"]["RichlyPerfusedAggregate"],
        index_richly_aggregated,
        CONSTS["INDEX"]["AllProperties"]
    )

    # Bronchial Lung flow is based upon pulmonary lung flow
    tissues[
        :,
        CONSTS["INDEX"]["LungBronchial"],
        CONSTS["INDEX"]["Flow"]
    ] = \
        tissues[
        :,
        CONSTS["INDEX"]["Lung"],
        CONSTS["INDEX"]["Flow"]
    ] * \
        CONSTS["BRONCHIAL_FLOW_FRACTION"]

    # Update units for flows
    tissues[:, :, CONSTS["INDEX"]["Flow"]] = convert_flow_units(
        tissues[:, :, CONSTS["INDEX"]["Flow"]],
        flow_units
    )
    personal_details[:, 5] = convert_flow_units(
        personal_details[:, 5],
        flow_units
    )

    # Enzymes abundances and totals
    # Based upon eq'n 11 in Howgate et al 2006. See
    # http://informahealthcare.com/doi/abs/10.1080/00498250600683197

    enzymes = {}
    enzymes["Names"] = enzyme_names
    enzymes["NEnzymes"] = len(enzyme_names)
    enzymes["MPPGLs"] = calculate_mppgl(ages)

    in_vitro_enzyme_rates_in = np.array(in_vitro_enzyme_rates)
    vary_in_vitro_enzyme_rates(
        in_vitro_enzyme_rates_in,
        np.array(in_vitro_enzyme_rate_coeffs_of_var)
    )
    enzymes["InVivoEnzymeRates"] = calculate_in_vivo_enzyme_rate(
        in_vitro_enzyme_rates_in,
        enzymes["MPPGLs"],
        tissues[:, CONSTS["INDEX"]["Liver"], CONSTS["INDEX"]["Mass"]]
    )
    if enzymes["InVivoEnzymeRates"].size > 0:
        enzymes["InVivoEnzymeRates"] = convert_enzyme_rate_units(
            enzymes["InVivoEnzymeRates"],
            enzyme_rate_units, # type: ignore
            molecular_weight
        )

    # Remove organs that have been aggregated.
    # This has to happen _after_ enzyme rates are calculated or liver could be set to zero
    index_to_keep = invert_indices(
        tissues.shape[1],
        np.union1d(index_slowly_aggregated, index_richly_aggregated)
    )
    tissues = tissues[:, index_to_keep, :]
    organ_names = [CONSTS["ORGAN"]["ExtendedNames"][i] for i in index_to_keep]

    # Summary stats for each tissue mass/flow (arithmetic & geometric mean and std dev, some percentiles)
    is_male = sexes == CONSTS["KEY"]["Sex"]["Male"]
    is_female = sexes == CONSTS["KEY"]["Sex"]["Female"]

    summary_stats = {
        "Male": {
            "Mass": generate_stats(tissues[is_male, :, CONSTS["INDEX"]["Mass"]]),
            "Flow": generate_stats(tissues[is_male, :, CONSTS["INDEX"]["Flow"]]),
            "MPPGL": generate_stats(enzymes["MPPGLs"][is_male]),
            "InVivoEnzymeRate": generate_stats(enzymes["InVivoEnzymeRates"][is_male, :])
        },
        "Female": {
            "Mass": generate_stats(tissues[is_female, :, CONSTS["INDEX"]["Mass"]]),
            "Flow": generate_stats(tissues[is_female, :, CONSTS["INDEX"]["Flow"]]),
            "MPPGL": generate_stats(enzymes["MPPGLs"][is_female]),
            "InVivoEnzymeRate": generate_stats(enzymes["InVivoEnzymeRates"][is_female, :])
        }
    }

    # Output structure
    population = {}
    population["Roots"] = {
        "Ethnicities": CONSTS["NAMES"]["Ethnicity"][dataset.name],
        "Sexes": CONSTS["NAMES"]["Sex"],
        "Names": CONSTS["NAMES"]["PersonalDetails"],
        "Values": personal_details
    }

    population["Tissues"] = {
        "Names": organ_names,
        "Properties": CONSTS["NAMES"]["OrganProperties"],
        "Values": tissues
    }

    population["Enzymes"] = enzymes
    population["Summary"] = summary_stats

    population["Inputs"] = collate_inputs(
        population_size,
        dataset,
        population_type,
        seed,
        age_range,
        bmi_range,
        height_range,
        prob_of_male,
        probs_of_ethnicities,
        flow_units,
        enzyme_rate_units
    )

    return population, number_of_individuals_discarded


# if __name__ == '__main__':

#     def callback(gen, dis):
#         print(f"{gen} / {dis}")
#         return gen >= 5

#     population, number_of_individuals_discarded = generate_pop(
#         population_size=10,
#         dataset_name=Dataset.P3M,
#         age_range=(18, 60),
#         bmi_range=(20, 25),
#         height_range=(100, 150),
#         prob_of_male=0.5,
#         probs_of_ethnicities=(0.3, 0.4, 0.3),
#         is_richly_perfused_tissue_discrete=False,
#         is_slowly_perfused_tissue_discrete=False,
#         seed=42,
#         callback=callback
#     )

#     import sys
#     if population is None:
#         sys.exit(1)
#     else:
#         print(f"{number_of_individuals_discarded} discarded")

#     import os
#     from pypopgenbe.impl.poptocsv import pop_to_csv
#     csv = pop_to_csv(population)
#     with open("./dist/test.csv", "w") as f:
#         f.writelines(line + os.linesep for line in csv)
