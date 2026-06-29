import unittest

from Utils.PoseReba2D import compute_observed_reba


def pose(points):
    out = [[0.0, 0.0, 0.0] for _ in range(17)]
    for idx, xy in points.items():
        out[idx] = [float(xy[0]), float(xy[1]), 1.0]
    return out


class PoseReba2DTests(unittest.TestCase):
    def test_visible_posture_changes_scores(self):
        neutral = pose({
            0:(100,50), 5:(80,100), 6:(120,100), 7:(80,170), 8:(120,170),
            9:(80,240), 10:(120,240), 11:(85,210), 12:(115,210),
            13:(85,300), 14:(115,300), 15:(85,390), 16:(115,390),
        })
        risky = pose({
            0:(205,130), 5:(170,170), 6:(210,170), 7:(170,80), 8:(210,80),
            9:(100,80), 10:(280,80), 11:(90,210), 12:(120,210),
            13:(170,260), 14:(200,260), 15:(100,260), 16:(130,260),
        })
        a = compute_observed_reba(neutral)
        b = compute_observed_reba(risky)
        self.assertIsNotNone(a); self.assertIsNotNone(b)
        self.assertGreater(sum(b["steps"]), sum(a["steps"]))
        self.assertGreaterEqual(b["steps"][1], 3)  # trunk
        self.assertGreaterEqual(b["steps"][3], 3)  # upper arm


if __name__ == "__main__":
    unittest.main()
