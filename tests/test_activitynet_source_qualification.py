"""Reject wrong source content/duration without looking at model predictions."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from qualify_activitynet_repair import validate_source


class SourceQualificationTests(unittest.TestCase):
    def setUp(self):
        self.meta = {'source_duration_sec': 70, 'source_total_frames': 1680,
                     'output_width': 64, 'output_height': 48}
        self.images = [np.zeros((48, 64, 3), dtype=np.uint8)]*8
        self.reference = {'output_width': 64, 'output_height': 48}

    def test_consistent_video_source_is_accepted(self):
        result = validate_source(self.meta, 70, self.reference, self.images, self.images)
        self.assertTrue(result['passed'])
        self.assertFalse(result['author_raw_file_byte_identity_verified'])

    def test_short_intro_cannot_replace_full_video(self):
        self.meta.update(source_duration_sec=2.1, source_total_frames=64)
        result = validate_source(self.meta, 70, self.reference, self.images, self.images)
        self.assertIn('duration_mismatch', result['failure_reasons'])

    def test_matching_duration_does_not_accept_wrong_visual_content(self):
        wrong = [np.full((48, 64, 3), 255, dtype=np.uint8)]*8
        result = validate_source(self.meta, 70, self.reference, wrong, self.images)
        self.assertIn('prefix_content_mismatch_or_alignment_uncertain', result['failure_reasons'])

    def test_low_fps_mirror_uses_time_qualification_not_exact_count_equality(self):
        self.meta['source_total_frames'] = 210
        result = validate_source(self.meta, 70, self.reference, self.images, self.images)
        self.assertTrue(result['passed'])


if __name__ == '__main__':
    unittest.main()
