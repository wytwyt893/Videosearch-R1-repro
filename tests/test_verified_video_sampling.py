"""Real encoded fixtures exercise unknown counts, coverage, PTS and version checks."""
from fractions import Fraction
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_activitynet_sampling as audit
import extract_video_frames_verified as verified


def encode(path, frames, pts=None):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('ffv1' if path.suffix == '.mkv' else 'mpeg4', rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        for j in range(frames):
            pixels = np.zeros((48, 64, 3), dtype=np.uint8)
            pixels[:] = [j % 256, (j*3) % 256, (j*7) % 256]
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            if pts is not None:
                frame.pts, frame.time_base = pts[j], Fraction(1, 24)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class VerifiedSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.long = cls.root/'complete_70s.mkv'
        cls.normal = cls.root/'complete_10s.mp4'
        cls.short = cls.root/'complete_half_second.mkv'
        cls.vfr = cls.root/'variable_timestamps.mkv'
        encode(cls.long, 1680)
        encode(cls.normal, 240)
        encode(cls.short, 12)
        encode(cls.vfr, 8, pts=[0, 1, 2, 3, 24, 48, 96, 144])
        cls.long_array, cls.long_meta = verified.extract_verified(cls.long)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete_mkv_with_unknown_header_reaches_the_end(self):
        m = self.long_meta
        self.assertEqual(m['header_source_total_frames'], 0)
        self.assertEqual(m['decoded_source_total_frames'], 1680)
        self.assertEqual(m['output_num_frames'], 64)
        self.assertEqual(m['sampled_indices_src'][-1], 1679)
        self.assertGreater(m['sampled_timestamps_sec'][-1], 69.9)
        self.assertEqual(m['source_frame_span_fraction'], 1)

    def test_same_intact_file_triggers_official_prefix_fallback(self):
        result = verified.compare_official(self.long, self.long_array)
        original = result['official_metadata']
        self.assertEqual(original['sampled_indices_src'], list(range(64)))
        self.assertAlmostEqual(original['sampled_timestamps_sec'][-1], 63/24)
        self.assertEqual(original['source_total_frames'], 64)
        self.assertTrue(result['identical_array_shape'])
        self.assertFalse(result['identical_arrays'])

    def test_known_normal_mp4_is_byte_identical_to_official_extraction(self):
        array, m = verified.extract_verified(self.normal)
        self.assertEqual(m['header_source_total_frames'], 240)
        self.assertEqual(m['decoded_source_total_frames'], 240)
        self.assertEqual(m['output_num_frames'], 10)
        result = verified.compare_official(self.normal, array)
        self.assertTrue(result['identical_arrays'])

    def test_short_video_even_padding_does_not_invent_distinct_frames(self):
        array, m = verified.extract_verified(self.short)
        self.assertEqual(m['sampled_frames_before_even_pad'], 1)
        self.assertEqual(m['output_num_frames'], 2)
        self.assertEqual(m['sampled_indices_src'], [0, 0])
        self.assertTrue(m['even_pad_applied'])
        np.testing.assert_array_equal(array[0], array[1])

    def test_real_pts_are_saved_for_variable_frame_intervals(self):
        array, m = verified.extract_verified(self.vfr)
        self.assertGreater(m['sampled_timestamps_sec'][-1], 5.9)
        self.assertLess(m['sampled_indices_src'][-1]/24, 0.4)
        self.assertGreater(m['positive_pts_gap_seconds']['max'], 1.9)

    def test_reference_probes_use_real_pts_when_frame_rate_changes(self):
        probes = {}
        _, m = verified.extract_verified(self.vfr, probe_timestamps_sec=[0, 2.01, 5.99], probe_sink=probes)
        self.assertEqual(probes['source_indices'], [0, 5, 7])
        self.assertAlmostEqual(probes['timestamps_sec'][-1], 6.0, places=3)
        self.assertEqual(len(probes['images']), 3)

    def test_low_fps_probes_include_both_frames_around_a_cut(self):
        probes = {}
        verified.extract_verified(self.vfr, probe_timestamps_sec=[2.01], probe_sink=probes,
                                  probe_selection='bracket')
        self.assertEqual(probes['source_indices'], [5, 6])
        self.assertEqual(probes['groups'], [[0, 1]])
        self.assertLess(probes['timestamps_sec'][0], 2.01)
        self.assertGreater(probes['timestamps_sec'][1], 2.01)

    def test_original_version_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'version mismatch'):
            verified.extract_verified(self.normal, expected_total_frames=6409)

    def test_undecodable_file_is_rejected(self):
        path = self.root/'not_a_video.mkv'
        path.write_bytes(b'not a video')
        with self.assertRaises(av.error.InvalidDataError):
            verified.extract_verified(path)

    def test_invalid_frame_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            verified.extract_verified(self.normal, max_frames=63)

    def test_audit_does_not_accept_self_consistent_prefix_metadata(self):
        m = dict(verified.compare_official(self.long, self.long_array)['official_metadata'])
        m.update(video_id='fixture', output_npy_path='fixture.npy')
        np.save(self.root/'fixture.npy', self.long_array)
        embedding = {'video_path': m['source_video_path'],
                     'info': {'num_total': 1680, 'fps': 24}}
        with patch.object(audit, 'DATA', self.root):
            result = audit.inspect_record(m, embedding, 3)
        self.assertTrue(result['uniform_against_own_recorded_count'])
        self.assertFalse(result['uniform_against_independent_retrieval_count'])
        self.assertEqual(result['status'], 'SUSPECTED_UNKNOWN_FRAME_FALLBACK')


if __name__ == '__main__':
    unittest.main()
