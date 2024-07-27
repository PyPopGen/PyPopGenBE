from typing import Dict, Any
import unittest
from pathlib import Path
from defineconsts.parsemeanvaluesxml import parse_mean_values_xml

THIS_DIR = Path(__file__).parent


def mean_individual_to_consts(mean_individual: Dict[str, Any]) -> Dict[str, Any]:
    CONSTS = {"NUMBER_OF_TISSUES": {"Base": len(mean_individual["Sex"][0]["tissue"])},
              "ORGAN": {"Names": [], "Mass": {"Dist": [[], []], "Mean": [[], []], "CoeffOfVar": [[], []]},
                        "Flow": {"Dist": [[], []], "Mean": [[], []], "CoeffOfVar": [[], []]}},
              "INDEX": {}}

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

    return CONSTS


class TestParseMeanValuesXml(unittest.TestCase):

    def test_parse_mean_values_xml(self):
        mean_values_path = THIS_DIR.parent / 'defineconsts/data/MeanValues.xml'
        mean_individual = parse_mean_values_xml(mean_values_path)
        consts = mean_individual_to_consts(mean_individual)
        # print(json.dumps(consts, sort_keys=True, indent=4))
        # return
        self.assertEqual(consts['NUMBER_OF_TISSUES']['Base'], 15)
        self.assertEqual(len(consts['ORGAN']['Names']), 15)
        self.assertEqual(consts['ORGAN']['Names'][0], 'Lung')
        self.assertEqual(consts['ORGAN']['Names'][1], 'Brain')
        self.assertAlmostEqual(consts['ORGAN']['Mass']['Mean'][0][0], 0.018)
        self.assertAlmostEqual(consts['ORGAN']['Flow']['Mean'][1][1], 0.13)
        self.assertEqual(consts['INDEX']['Lung'], 0)
        self.assertEqual(consts['INDEX']['Brain'], 1)
