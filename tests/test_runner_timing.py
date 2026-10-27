"""Check initialization/warmup accounting rather than multiplying wall time."""
import datetime
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_charades_full import analyze_timing, wait_for_gpu_release


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.start = datetime.datetime(2026, 10, 27, tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
        self.log = ('2026-10-27 00:00:10,000 - evaluator - INFO - Streaming detail jsonl: test\n'
                    '2026-10-27 00:00:30,000 - evaluator - INFO - Saved summary json: test\n')
        self.events = [{'index': i, 'elapsed_seconds': t} for i, t in enumerate([18, 24, 30])]

    def test_startup_and_first_query_warmup_are_counted_once(self):
        r = analyze_timing(self.log, self.events, self.start, 35, target_count=100)
        self.assertEqual(r['startup_seconds'], 10)
        self.assertEqual(r['official_loop_seconds'], 20)
        self.assertEqual(r['steady_mean_seconds_per_query'], 6)
        self.assertEqual(r['full_estimate']['point_seconds'], 677)

    def test_no_projection_without_official_loop_marker(self):
        with self.assertRaises(ValueError):
            analyze_timing('', self.events, self.start, 35)

    def test_nonmonotonic_row_events_are_rejected(self):
        bad = [self.events[1], self.events[0], self.events[2]]
        with self.assertRaises(ValueError):
            analyze_timing(self.log, bad, self.start, 35)

    def test_exit_utilization_sample_requires_two_later_idle_samples(self):
        replies = ['GPU-test, 81153, 0, 69', '', 'GPU-test, 81153, 0, 0', '',
                   'GPU-test, 81153, 0, 0', '']
        with patch('run_charades_full.subprocess.check_output', side_effect=replies), \
                patch('run_charades_full.time.sleep'):
            result = wait_for_gpu_release(['nvidia-smi'], 'GPU-test')
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['samples']), 3)
        self.assertFalse(result['samples'][0]['idle'])


if __name__ == '__main__':
    unittest.main()
