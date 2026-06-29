import unittest

from Utils.RebaEngine import TABLE_A, TABLE_B, TABLE_C, RebaModifiers, score_reba


class RebaEngineTests(unittest.TestCase):
    def test_reference_table_cells(self):
        self.assertEqual(TABLE_A[0][0][0], 1)
        self.assertEqual(TABLE_A[2][1][1], 5)
        self.assertEqual(TABLE_A[4][2][3], 9)
        self.assertEqual(TABLE_B[0][0][0], 1)
        self.assertEqual(TABLE_B[5][1][2], 9)
        self.assertEqual(TABLE_C[4][4], 6)
        self.assertEqual(TABLE_C[8][7], 11)

    def test_neutral_posture_is_low(self):
        result = score_reba({"neck_deg":10,"trunk_deg":3,"knee_flex_deg":5,
                             "upper_arm_deg":15,"elbow_deg":80,"wrist_deviation_deg":5})
        self.assertLessEqual(result.final, 3)

    def test_high_risk_modifiers_raise_score(self):
        angles = {"neck_deg":35,"trunk_deg":65,"knee_flex_deg":65,
                  "upper_arm_deg":100,"elbow_deg":130,"wrist_deviation_deg":30}
        base = score_reba(angles)
        risky = score_reba(angles, RebaModifiers(
            neck_twist_or_side=True, trunk_twist_or_side=True,
            shoulder_raised=True, arm_abducted=True,
            wrist_twist_or_deviation=True, bilateral_support=False,
            load_score=2, coupling_score=2, activity_score=2,
        ))
        self.assertGreater(risky.final, base.final)
        self.assertGreaterEqual(risky.final, 11)

    def test_supported_arm_reduces_upper_arm_score(self):
        angles = {"neck_deg":10,"trunk_deg":5,"knee_flex_deg":5,
                  "upper_arm_deg":60,"elbow_deg":80,"wrist_deviation_deg":5}
        normal = score_reba(angles)
        supported = score_reba(angles, RebaModifiers(arm_supported=True))
        self.assertEqual(supported.upper_arm, normal.upper_arm - 1)

    def test_missing_angle_is_visible_warning(self):
        result = score_reba({})
        self.assertTrue(result.warnings)
        self.assertLessEqual(result.final, 15)


if __name__ == "__main__":
    unittest.main()
