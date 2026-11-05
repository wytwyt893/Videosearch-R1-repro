"""Source-time clock encoding respects VFR and duplicated even padding."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from activitynet_repair_time import CLOCK_HZ, FIELD, timestamp_payload


class RepairTimeTests(unittest.TestCase):
    def row(self, timestamps):
        return {FIELD: timestamps, 'output_num_frames': len(timestamps),
                'source_duration_sec': max(timestamps)+1}

    def test_actual_vfr_times_survive_integer_transport(self):
        expected = [0.0, .033, 20.4127, 72.5725]
        payload = timestamp_payload(self.row(expected))
        actual = [i/payload['raw_fps'] for i in payload['frames_indices']]
        self.assertEqual(actual, expected)
        self.assertTrue(all(isinstance(i, int) for i in payload['frames_indices']))
        self.assertEqual(payload['raw_fps'], CLOCK_HZ)

    def test_even_padding_keeps_last_frame_time(self):
        payload = timestamp_payload(self.row([0, 2, 4, 4]))
        self.assertEqual(payload['frames_indices'][-1], payload['frames_indices'][-2])

    def test_clock_not_replaced_with_sequential_one_fps_labels(self):
        payload = timestamp_payload(self.row([0, 240, 500, 745.4]))
        self.assertEqual(payload['frames_indices'][-1]/payload['raw_fps'], 745.4)

    def test_out_of_range_or_backwards_times_are_rejected(self):
        for row in ({FIELD: [2, 1], 'output_num_frames': 2, 'source_duration_sec': 3},
                    {FIELD: [0, 5], 'output_num_frames': 2, 'source_duration_sec': 3}):
            with self.assertRaises(ValueError):
                timestamp_payload(row)


if __name__ == '__main__':
    unittest.main()
