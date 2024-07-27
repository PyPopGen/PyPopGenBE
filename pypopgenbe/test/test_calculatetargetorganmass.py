import numpy as np
import unittest
from pypopgenbe.impl.calculatetargetorganmass import calculate_target_organ_mass


class TestCalculateTargetOrganMass(unittest.TestCase):
    def test_calculate_target_organ_mass(self):
        # Example values for testing
        age = 30
        sex_name = "Male"
        ethnicity_name = "Caucasian"
        mean_body_weight = 70.0
        mean_height = 1.75
        target_body_weight = 75.0
        target_height = 1.80
        organ_masses = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

        index = {
            "Lung": 0,
            "Brain": 1,
            "Liver": 3,
            "Muscle": 9,
            "Adipose": 11,
            "Bone": 12,
            "Skin": 13,
            "LiverFeeder": [3, 4, 5, 6, 7, 8],
            "SlowlyPerfused": [11, 12, 9, 13],
            "RichlyPerfused": [1, 14, 10, 2, 8, 3, 4, 7, 5, 6],
            "LiverTotal": 16,
            "SlowlyPerfusedAggregate": 17,
            "RichlyPerfusedAggregate": 18,
            "LungBronchial": 19,
            "NewOrganOrder": [19, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "Mass": 0,
            "Flow": 1,
            "AllProperties": [0, 1]
        }

        scaled_height, target_organ_mass = calculate_target_organ_mass(
            age, sex_name, ethnicity_name, mean_body_weight,
            mean_height, target_body_weight, target_height,
            organ_masses, index
        )

        self.assertIsInstance(scaled_height, float)
        self.assertIsInstance(target_organ_mass, np.ndarray)
        self.assertEqual(len(target_organ_mass), len(organ_masses))
