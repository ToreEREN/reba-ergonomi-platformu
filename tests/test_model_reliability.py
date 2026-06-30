import unittest

import joblib
import numpy as np

from Utils.FeatureSchema import FEATURE_NAMES, FEATURE_SEMANTICS
from Utils.ModelReliability import (
    paired_model_comparison,
    predict_with_tree_uncertainty,
    uncertainty_summary,
    validate_angle_range,
)


class ModelReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load("ModelExperiments/extra_trees.joblib")

    def test_feature_contract_is_explicit(self):
        self.assertEqual(len(FEATURE_NAMES), 90)
        self.assertIn("NOT physical depth", FEATURE_SEMANTICS["z"])

    def test_tree_uncertainty_shapes(self):
        X = np.zeros((2, 90), dtype=np.float32)
        mean, std = predict_with_tree_uncertainty(self.model, X)
        self.assertEqual(mean.shape, (2, 19))
        self.assertEqual(std.shape, (2, 19))
        self.assertTrue((std >= 0).all())
        summary = uncertainty_summary(mean[:1], std[:1], [str(i) for i in range(19)])
        self.assertIn("mean_tree_std_deg", summary)

    def test_paired_comparison_detects_better_predictions(self):
        rng = np.random.default_rng(42)
        truth = rng.normal(90, 20, (80, 3))
        new = truth + rng.normal(0, 2, truth.shape)
        old = truth + rng.normal(0, 12, truth.shape)
        result = paired_model_comparison(truth, new, old, n_boot=100, seed=42)
        self.assertLess(result["wilcoxon_one_sided_p"], 0.01)
        self.assertGreater(result["r2_gain_ci95"][1], 0)

    def test_angle_range_guard(self):
        self.assertTrue(validate_angle_range([0, 90, 180])["valid"])
        self.assertFalse(validate_angle_range([-1, 90, 181])["valid"])


if __name__ == "__main__":
    unittest.main()
