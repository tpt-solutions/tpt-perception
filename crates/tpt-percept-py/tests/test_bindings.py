"""Tests for the tpt_perception Python bindings.

Run after installing the wheel or `maturin develop`:

    python -m unittest discover -s crates/tpt-percept-py/tests -v
"""

import math
import unittest

import tpt_perception as tp


def box_surface(steps=21):
    vals = [(i - steps // 2) * 0.05 for i in range(steps)]
    pts = []
    for a in vals:
        for b in vals:
            pts += [
                [a, b, 1.0],
                [a, b, -1.0],
                [a, 1.0, b],
                [a, -1.0, b],
                [1.0, a, b],
                [-1.0, a, b],
            ]
    return pts


class IcpTests(unittest.TestCase):
    def test_point_to_point_recovers_translation(self):
        target = box_surface()
        shift = [0.01, -0.008, 0.004]
        source = [[p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]] for p in target]
        result = tp.align_icp(source, target, max_correspondence_distance=0.25)
        self.assertTrue(result["converged"])
        self.assertLess(result["rmse"], 1e-4)
        for got, want in zip(result["translation"], [-s for s in shift]):
            self.assertAlmostEqual(got, want, places=3)

    def test_point_to_plane_runs(self):
        target = box_surface()
        result = tp.align_icp(target, target, max_correspondence_distance=0.25, point_to_plane=True)
        self.assertEqual(len(result["rotation"]), 3)
        self.assertGreater(result["fitness"], 0.9)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            tp.align_icp([], box_surface())


class OdometryTests(unittest.TestCase):
    def test_counts_and_tracks_motion(self):
        base = box_surface()
        odom = tp.LidarOdometry(map_window=3)
        odom.process(base)
        shifted = [[p[0] - 0.05, p[1], p[2]] for p in base]
        rotation, translation, converged = odom.process(shifted)
        self.assertTrue(converged)
        self.assertEqual(odom.scans_processed(), 2)
        self.assertAlmostEqual(translation[0], 0.05, delta=0.03)


class PlaneTests(unittest.TestCase):
    def test_floor_is_found(self):
        floor = [[x * 0.1, y * 0.1, 0.0] for x in range(30) for y in range(30)]
        planes = tp.segment_planes(floor, distance_threshold=0.02, max_planes=1)
        self.assertEqual(len(planes), 1)
        self.assertAlmostEqual(abs(planes[0]["normal"][2]), 1.0, places=6)
        self.assertEqual(len(planes[0]["inliers"]), len(floor))


class DownsampleTests(unittest.TestCase):
    def test_reduces_count(self):
        pts = [[0.01 * i, 0.0, 0.0] for i in range(100)]
        reduced = tp.downsample(pts, 0.1)
        self.assertLess(len(reduced), len(pts))
        self.assertTrue(all(math.isfinite(c) for p in reduced for c in p))


class IoTests(unittest.TestCase):
    def test_missing_file_raises_oserror(self):
        with self.assertRaises(OSError):
            tp.read_pcd_points("/definitely/not/here.pcd")


if __name__ == "__main__":
    unittest.main()
