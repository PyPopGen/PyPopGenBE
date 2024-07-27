import numpy as np
from typing import Any
from pathlib import Path
import json
import pickle
from defineconsts.parsemeanvaluesxml import parse_mean_values_xml
from defineconsts.indicesoforgans import indices_of_organs
from defineconsts.numpyencoder import NumpyEncoder
from defineconsts.readcoeffsfromfile import read_coeffs_from_file

THIS_DIR = Path(__file__).parent


def define_consts_for_popgen() -> dict[str, Any]:

    in_file_path = THIS_DIR / 'data/MeanValues.xml'
    mean_individual = parse_mean_values_xml(in_file_path)

    CONSTS = {
        "NUMBER_OF_TISSUES": {"Base": len(mean_individual["Sex"][0]["tissue"])},
        "ORGAN": {
            "Names": [],
            "Mass": {"Dist": [[], []], "Mean": [[], []], "CoeffOfVar": [[], []]},
            "Flow": {"Dist": [[], []], "Mean": [[], []], "CoeffOfVar": [[], []]}
        },
        "INDEX": {},
        "DISTRIBUTION": {"Mass": {}, "Flow": {}}
    }

    for i in range(CONSTS["NUMBER_OF_TISSUES"]["Base"]):
        for sex in range(2):
            sex_data = mean_individual["Sex"][sex]
            tissue_data = sex_data["tissue"][i]
            tissue_name = tissue_data["type"]

            if sex == 0:
                CONSTS["ORGAN"]["Names"].append(tissue_name)

            param1, param2 = tissue_data["param"]

            CONSTS["ORGAN"]["Mass"]["Dist"][sex].append(param2)
            CONSTS["ORGAN"]["Flow"]["Dist"][sex].append(param1)

            CONSTS["ORGAN"]["Mass"]["Mean"][sex].append(
                param2["distribution"][0]["content"])
            CONSTS["ORGAN"]["Mass"]["CoeffOfVar"][sex].append(
                param2["distribution"][1]["content"])

            CONSTS["ORGAN"]["Flow"]["Mean"][sex].append(
                param1["distribution"][0]["content"])
            CONSTS["ORGAN"]["Flow"]["CoeffOfVar"][sex].append(
                param1["distribution"][1]["content"])

        organ_name = CONSTS["ORGAN"]["Names"][i]
        if organ_name in ["Adipose", "Bone", "Brain", "Skin", "Lung", "Liver", "Muscle"]:
            CONSTS["INDEX"][organ_name] = i

    CONSTS["ORGAN"]["Mass"]["Mean"][0] = np.array(
        CONSTS["ORGAN"]["Mass"]["Mean"][0])
    CONSTS["ORGAN"]["Mass"]["Mean"][1] = np.array(
        CONSTS["ORGAN"]["Mass"]["Mean"][1])
    CONSTS["ORGAN"]["Mass"]["CoeffOfVar"][0] = np.array(
        CONSTS["ORGAN"]["Mass"]["CoeffOfVar"][0])
    CONSTS["ORGAN"]["Mass"]["CoeffOfVar"][1] = np.array(
        CONSTS["ORGAN"]["Mass"]["CoeffOfVar"][1])

    CONSTS["ORGAN"]["Flow"]["Mean"][0] = np.array(
        CONSTS["ORGAN"]["Flow"]["Mean"][0])
    CONSTS["ORGAN"]["Flow"]["Mean"][1] = np.array(
        CONSTS["ORGAN"]["Flow"]["Mean"][1])
    CONSTS["ORGAN"]["Flow"]["CoeffOfVar"][0] = np.array(
        CONSTS["ORGAN"]["Flow"]["CoeffOfVar"][0])
    CONSTS["ORGAN"]["Flow"]["CoeffOfVar"][1] = np.array(
        CONSTS["ORGAN"]["Flow"]["CoeffOfVar"][1])

    CONSTS["INDEX"]["LiverFeeder"] = indices_of_organs(
        ["Liver", "Pancreas", "Spleen", "Stomach",
            "Small intestine", "Large intestine"],
        CONSTS["ORGAN"]["Names"]
    )

    CONSTS["INDEX"]["SlowlyPerfused"] = indices_of_organs(
        ["Adipose", "Bone", "Muscle", "Skin"],
        CONSTS["ORGAN"]["Names"]
    )

    CONSTS["INDEX"]["RichlyPerfused"] = indices_of_organs(
        [
            "Brain", "Gonads", "Heart", "Kidneys", "Large intestine",
            "Liver", "Pancreas", "Small intestine", "Spleen", "Stomach"
        ],
        CONSTS["ORGAN"]["Names"]
    )

    CONSTS["INDEX"]["LiverTotal"] = CONSTS["NUMBER_OF_TISSUES"]["Base"]
    CONSTS["INDEX"]["SlowlyPerfusedAggregate"] = CONSTS["NUMBER_OF_TISSUES"]["Base"] + 1
    CONSTS["INDEX"]["RichlyPerfusedAggregate"] = CONSTS["NUMBER_OF_TISSUES"]["Base"] + 2
    CONSTS["INDEX"]["LungBronchial"] = CONSTS["NUMBER_OF_TISSUES"]["Base"] + 3

    CONSTS["NUMBER_OF_TISSUES"]["SlowlyPerfused"] = len(
        CONSTS["INDEX"]["SlowlyPerfused"])
    CONSTS["NUMBER_OF_TISSUES"]["RichlyPerfused"] = len(
        CONSTS["INDEX"]["RichlyPerfused"])
    CONSTS["NUMBER_OF_TISSUES"]["Extended"] = CONSTS["NUMBER_OF_TISSUES"]["Base"] + 4

    CONSTS["INDEX"]["NewOrganOrder"] = (
        list(range(0, CONSTS["INDEX"]["Lung"])) +
        [CONSTS["INDEX"]["LungBronchial"]] +
        list(range(CONSTS["INDEX"]["Lung"], CONSTS["INDEX"]["LungBronchial"]-1)) +
        list(range(CONSTS["INDEX"]["LungBronchial"],
             CONSTS["NUMBER_OF_TISSUES"]["Extended"]))
    )

    CONSTS["INDEX"]["Mass"] = 0  # 1
    CONSTS["INDEX"]["Flow"] = 1  # 2
    CONSTS["INDEX"]["AllProperties"] = np.array(
        [CONSTS["INDEX"]["Mass"], CONSTS["INDEX"]["Flow"]])

    CONSTS["ORGAN"]["ExtendedNames"] = CONSTS["ORGAN"]["Names"] + [
        "Liver Total", "Slowly Perfused", "Richly Perfused", "Lung Bronchial"
    ]

    # Bronchial lung flow is a fraction of pulmonary lung flow
    CONSTS["BRONCHIAL_FLOW_FRACTION"] = 0.025

    CONSTS["MIN_ADIPOSE_FRACTION"] = 0.1  # Athletes not considered!

    def get_distribution_flags(distributions):
        is_normal = [dist["unit"] == "Normal" for dist in distributions]
        is_lognormal = [dist["unit"] == "Lognormal" for dist in distributions]
        return {"IsNormal": is_normal, "IsLognormal": is_lognormal}

    def calculate_sigma(mu, coeff_var, is_lognormal):
        sigma = mu * coeff_var
        sigma[is_lognormal] = np.sqrt(np.log(coeff_var[is_lognormal] ** 2 + 1))
        return sigma

    def calculate_mu(mu, sigma, is_lognormal):
        mu_log = np.log(mu) - sigma ** 2 / 2
        mu[is_lognormal] = mu_log[is_lognormal]
        return mu

    for sex in range(2):
        mass_distributions = CONSTS["ORGAN"]["Mass"]["Dist"][sex]
        flow_distributions = CONSTS["ORGAN"]["Flow"]["Dist"][sex]

        CONSTS["DISTRIBUTION"]["Mass"][sex] = get_distribution_flags(
            mass_distributions)
        CONSTS["DISTRIBUTION"]["Flow"][sex] = get_distribution_flags(
            flow_distributions)

        isln_mass = CONSTS["DISTRIBUTION"]["Mass"][sex]["IsLognormal"]
        isln_flow = CONSTS["DISTRIBUTION"]["Flow"][sex]["IsLognormal"]

        mass_mean = np.array(CONSTS["ORGAN"]["Mass"]["Mean"][sex])
        mass_coeff_var = np.array(CONSTS["ORGAN"]["Mass"]["CoeffOfVar"][sex])
        flow_mean = np.array(CONSTS["ORGAN"]["Flow"]["Mean"][sex])
        flow_coeff_var = np.array(CONSTS["ORGAN"]["Flow"]["CoeffOfVar"][sex])

        CONSTS["ORGAN"]["Mass"]["Sigma"] = CONSTS["ORGAN"].get(
            "Mass", {}).get("Sigma", {})
        CONSTS["ORGAN"]["Flow"]["Sigma"] = CONSTS["ORGAN"].get(
            "Flow", {}).get("Sigma", {})

        CONSTS["ORGAN"]["Mass"]["Sigma"][sex] = calculate_sigma(
            mass_mean, mass_coeff_var, isln_mass)
        CONSTS["ORGAN"]["Flow"]["Sigma"][sex] = calculate_sigma(
            flow_mean, flow_coeff_var, isln_flow)

        CONSTS["ORGAN"]["Mass"]["Mu"] = CONSTS["ORGAN"].get(
            "Mass", {}).get("Mu", {})
        CONSTS["ORGAN"]["Flow"]["Mu"] = CONSTS["ORGAN"].get(
            "Flow", {}).get("Mu", {})

        CONSTS["ORGAN"]["Mass"]["Mu"][sex] = calculate_mu(
            mass_mean, CONSTS["ORGAN"]["Mass"]["Sigma"][sex], isln_mass)
        CONSTS["ORGAN"]["Flow"]["Mu"][sex] = calculate_mu(
            flow_mean, CONSTS["ORGAN"]["Flow"]["Sigma"][sex], isln_flow)

    CONSTS["NAMES"] = {
        "Sex": ["Male", "Female"],
        "Ethnicity": {
            "ICRP": ["White", "Black", "Non-Black Hispanic", "Other"],
            "P3M": ["White", "Black", "Non-Black Hispanic", "Other"],
            "HSE": ["White", "Black", "Asian", "Other"],
            "NDNS": ["Unknown"]
        },
        "PersonalDetails": ["Age", "Sex", "Ethnicity", "Body Mass", "Height", "Cardiac Output"],
        "OrganProperties": ["Mass", "Flow"],
        # "AllOrgans": list(map(lambda x: x["type"], CONSTS["ORGAN"]["Names"])),
        "Datasets": ["P3M", "ICRP", "HSE", "NDNS"],
        "EnzymeRateParameter": ["Vmax", "CLint"],
        "FlowUnits": ["MilliLitresPerMinute", "LitresPerHour"],
        "EnzymeRateVmaxUnits": [
            "PicoMolsPerMinute", "MicroMolsPerHour", "MilliMolsPerHour",
            "PicoGramsPerMinute", "MicroGramsPerHour", "MilliGramsPerHour"
        ],
        "EnzymeRateCLintUnits": [
            "MicroLitresPerMinute", "MilliLitresPerHour", "LitresPerHour"
        ]
    }

    p3m_ethnicities = ['white', 'black', 'non black hispanic', 'other']
    hse_ethnicities = ['white', 'black', 'asian', 'other']
    ndns_ethnicities = ['unknown']

    CONSTS["Height"] = {"P3M": {}, "HSE": {}, "NDNS": {}}
    CONSTS["BodyWeight"] = {"P3M": {}, "HSE": {}, "NDNS": {}}

    CONSTS["Height"]["P3M"]["Adult"] = read_coeffs_from_file(
        THIS_DIR / 'data' / 'P3M height.cm coeffs, Adult September 2013.csv',
        p3m_ethnicities
    )

    CONSTS["BodyWeight"]["P3M"]["Adult"] = read_coeffs_from_file(
        THIS_DIR / 'data' / 'P3M weight.kg coeffs, Adult September 2013.csv',
        p3m_ethnicities
    )

    CONSTS["Height"]["P3M"]["Child"] = read_coeffs_from_file(
        THIS_DIR / 'data' / 'P3M height.cm coeffs, Child September 2013.csv',
        p3m_ethnicities
    )

    CONSTS["BodyWeight"]["P3M"]["Child"] = read_coeffs_from_file(
        THIS_DIR / 'data' / 'P3M weight.kg coeffs, Child September 2013.csv',
        p3m_ethnicities
    )

    CONSTS["Height"]["HSE"]["Adult"] = read_coeffs_from_file(
        THIS_DIR / 'data' /
        'Health Survey For England combined ethnicity height.cm coeffs, Adult August 2013.csv',
        hse_ethnicities
    )

    CONSTS["BodyWeight"]["HSE"]["Adult"] = read_coeffs_from_file(
        THIS_DIR / 'data' /
        'Health Survey For England combined ethnicity weight.kg coeffs, Adult August 2013.csv',
        hse_ethnicities
    )

    CONSTS["Height"]["HSE"]["Child"] = read_coeffs_from_file(
        THIS_DIR / 'data' /
        'Health Survey For England combined ethnicity height.cm coeffs, Child August 2013.csv',
        hse_ethnicities
    )

    CONSTS["BodyWeight"]["HSE"]["Child"] = read_coeffs_from_file(
        THIS_DIR / 'data' /
        'Health Survey For England combined ethnicity weight.kg coeffs, Child August 2013.csv',
        hse_ethnicities
    )

    CONSTS["Height"]["NDNS"] = read_coeffs_from_file(
        THIS_DIR / 'data' / 'NDNS height.cm coeffs August 2013.csv',
        ndns_ethnicities
    )

    CONSTS["BodyWeight"]["NDNS"] = read_coeffs_from_file(
        THIS_DIR / 'data' / 'NDNS weight.kg coeffs August 2013.csv',
        ndns_ethnicities
    )

    # Reading ICRP height and bodyweight data from CSV files
    icrp_height_data = np.loadtxt(
        THIS_DIR / 'data' / 'figure 9, height by gender, age 0 to 56.csv', delimiter=',', skiprows=1)
    icrp_bodyweight_data = np.loadtxt(
        THIS_DIR / 'data' / 'figure 5, bodyweight by gender, age 0 to 56.csv', delimiter=',', skiprows=1)

    # Assigning height and bodyweight values and ages to CONSTS dictionary
    CONSTS["Height"]["ICRP"] = {
        "Values": icrp_height_data[:, 1:3],
        "Ages": icrp_height_data[:, 0]
    }

    CONSTS["BodyWeight"]["ICRP"] = {
        "Values": icrp_bodyweight_data[:, 1:3],
        "Ages": icrp_bodyweight_data[:, 0]
    }

    # Define the age groups for ICRP
    CONSTS["AgeGroups"] = {"ICRP": [0, 1, 5, 10, 15, 20, 25, 120]}

    # Define the cardiac output for ICRP with the modified constants
    CONSTS["CardiacOutput"] = {
        "ICRP": np.array([
            [600, 600],
            [1200, 1200],
            [3400, 3400],
            [5000, 5000],
            [6100, 6100],
            [6500, 5900],
            [6500, 5900],
            [3413, 3097]
        ])
    }

    # Define the coefficient of variation for height and body weight
    CONSTS["COEFF_OF_VAR"] = {
        "Height": [0.04, 0.04],
        "BodyWeight": [0.14, 0.2]
    }

    # Define the acceptable age ranges for different datasets
    CONSTS["AcceptableAgeRanges"] = {
        "ICRP": [0, 80],
        "P3M": [0, 80],
        "HSE": [0, 70],
        "NDNS": [1.25, 5]
    }

    # Initialize the summary variable with NaN arrays
    z = np.full(CONSTS["NUMBER_OF_TISSUES"]["Extended"], np.nan)
    s = {
        "Mean": z.copy(),
        "StdDev": z.copy(),
        "GeoMean": z.copy(),
        "GeoStdDev": z.copy(),
        "P2pt5": z.copy(),
        "P5": z.copy(),
        "Median": z.copy(),
        "P95": z.copy(),
        "P97pt5": z.copy()
    }

    CONSTS["Summary"] = {
        "Mass": {"Male": s.copy(), "Female": s.copy()},
        "Flow": {"Male": s.copy(), "Female": s.copy()}
    }

    # Define keys for sex and ethnicity
    CONSTS["KEY"] = {
        "Sex": {"Male": 1, "Female": 2},
        "Ethnicity": {
            "P3M": {"White": 1, "Black": 2, "NonBlackHispanic": 3, "Other": 4},
            "ICRP": {"White": 1, "Black": 2, "NonBlackHispanic": 3, "Other": 4},
            "HSE": {"White": 1, "Black": 2, "Asian": 3, "Other": 4},
            "NDNS": {"Unknown": 1}
        }
    }

    with open(THIS_DIR / 'data/popgenconsts.json', 'w', encoding='utf-8') as f:
        json.dump(CONSTS, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    with open(THIS_DIR / '../popgenconsts.pkl', 'wb') as f:
        pickle.dump(CONSTS, f)

    with open(THIS_DIR / '../popgenconsts.pkl', 'rb') as f:
        CONSTS = pickle.load(f)

    return CONSTS


if __name__ == '__main__':
    print(define_consts_for_popgen())
