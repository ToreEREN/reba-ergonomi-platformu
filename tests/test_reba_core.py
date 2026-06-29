import unittest

import pandas as pd

from Utils.Reba import (
    classify_reba_risk,
    compute_reba_scores,
    get_activity_score,
    get_coupling_score,
    get_force_load_score,
    rules,
)


class RebaCoreTests(unittest.TestCase):
    def test_angle_only_predictions_do_not_require_quaternions(self):
        row = {
            "head_vs_neck_yz_deg": 10.0,
            "head_vs_neck_xz_deg": 2.0,
            "trunk_deviation": 10.0,
            "kneeR_deviation": 10.0,
            "kneeL_deviation": 10.0,
            "upperarmL_elevation": 10.0,
            "upperarmR_elevation": 10.0,
            "elbowL_deviation": 30.0,
            "elbowR_deviation": 30.0,
            "wristL_deviation": 5.0,
            "wristR_deviation": 5.0,
        }
        scored = compute_reba_scores(pd.DataFrame([row]), rules)
        self.assertEqual(scored.loc[0, "Score_Step1"], 1)
        self.assertEqual(scored.loc[0, "Score_Step6"], 1)

    def test_manual_modifiers(self):
        self.assertEqual(get_force_load_score(12, True), 3)
        self.assertEqual(get_coupling_score("poor"), 2)
        self.assertEqual(get_activity_score(True, True, True), 3)

    def test_risk_boundaries(self):
        expected = {1: "Negligible risk", 2: "Low risk", 4: "Medium risk", 8: "High risk", 11: "Very high risk"}
        for score, label in expected.items():
            self.assertEqual(classify_reba_risk(score), label)


if __name__ == "__main__":
    unittest.main()
