"""Network batch failures resume; integrity failures must remain hard failures."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('didemo_download', SCRIPTS / 'prepare_didemo_assets.py')
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class DownloadRetryTests(unittest.TestCase):
    @mock.patch.object(downloader, 'update')
    @mock.patch.object(downloader.time, 'sleep')
    def test_failed_network_batch_can_resume(self, sleep, update):
        task = mock.Mock(side_effect=[downloader.AssetTransportBatchError('one TLS timeout'), 'complete'])
        self.assertEqual(downloader.run_phase('data', task), 'complete')
        self.assertEqual(task.call_count, 2)
        sleep.assert_called_once_with(10)

    @mock.patch.object(downloader, 'update')
    @mock.patch.object(downloader.time, 'sleep')
    def test_local_integrity_error_is_not_retried(self, sleep, update):
        task = mock.Mock(side_effect=ValueError('SHA256 mismatch'))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            downloader.run_phase('data', task)
        self.assertEqual(task.call_count, 1)
        sleep.assert_not_called()

    @mock.patch.object(downloader, 'update')
    @mock.patch.object(downloader.time, 'sleep')
    def test_repeated_network_failures_have_a_finite_limit(self, sleep, update):
        task = mock.Mock(side_effect=downloader.AssetTransportBatchError('TLS timeout'))
        with self.assertRaises(downloader.AssetTransportBatchError):
            downloader.run_phase('data', task)
        self.assertEqual(task.call_count, 4)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [10, 20, 30])


if __name__ == '__main__':
    unittest.main()
