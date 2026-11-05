"""Recover correspondence across low-FPS cuts and a declared image rotation."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from align_activitynet_repair_sources import choose_alignment


class SourceAlignmentTests(unittest.TestCase):
    def test_bracketing_frame_before_cut_can_match_when_nearest_after_cut_does_not(self):
        black = np.zeros((48, 64, 3), dtype=np.uint8)
        white = np.full((48, 64, 3), 255, dtype=np.uint8)
        probes = {'images': [black, white], 'groups': [[0, 1]]}
        selected = choose_alignment(probes, [black])
        self.assertEqual(selected['probe_indices'], [0])
        self.assertEqual(selected['median_rgb_mae'], 0)
        self.assertEqual(selected['rotate_ccw_degrees'], 0)

    def test_rotation_is_recovered_from_asymmetric_image(self):
        expected = np.zeros((16, 8, 3), dtype=np.uint8)
        expected[:4, :, 0] = 255
        expected[4:, :, 2] = 255
        rotated_source = np.rot90(expected, 1)
        probes = {'images': [rotated_source], 'groups': [[0]]}
        selected = choose_alignment(probes, [expected])
        self.assertEqual(selected['rotate_ccw_degrees'], 270)
        self.assertEqual(selected['median_rgb_mae'], 0)


if __name__ == '__main__':
    unittest.main()
