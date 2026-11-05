"""Independent tar partitions must neither lose nor duplicate source members."""
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import index_activitynet_raw_mirror as indexer


class TarPartitionTests(unittest.TestCase):
    def setUp(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w', format=tarfile.USTAR_FORMAT) as archive:
            for i in range(12):
                body = bytes([i])*1000
                header = tarfile.TarInfo(f'Activity_Videos/v_{i:011d}.mkv')
                header.size = len(body)
                archive.addfile(header, io.BytesIO(body))
        self.raw = buf.getvalue()
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)
        raw = self.raw

        class MemoryArchive:
            def __init__(self, metadata):
                self.boundaries = [0, len(raw)]
                self.downloaded_bytes = self.requests = 0
                self.client = types.SimpleNamespace(close=lambda: None)

            def read(self, offset, size):
                return raw[offset:offset+size]

        self.archive_type = MemoryArchive

    def tearDown(self):
        self.temp.cleanup()

    def test_partition_starts_inside_member_without_losing_headers(self):
        metadata = {'sha': 'fixture'}
        boundaries = [0, 5120, 10240, 15360, len(self.raw)]
        rows = []
        with patch.object(indexer, 'WORK', self.work), patch.object(indexer, 'SplitTar', self.archive_type):
            for i, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
                result = indexer.index_part(i, metadata, set(), start, stop)
                self.assertTrue(result['complete'], result.get('error'))
                rows.extend(result['members'])
        with tarfile.open(fileobj=io.BytesIO(self.raw)) as archive:
            expected = [r.name for r in archive.getmembers()]
        self.assertEqual(sorted(r['name'] for r in rows), sorted(expected))
        self.assertEqual(len({r['offset'] for r in rows}), len(expected))

    def test_resume_reuses_complete_partition(self):
        with patch.object(indexer, 'WORK', self.work), patch.object(indexer, 'SplitTar', self.archive_type):
            first = indexer.index_part(0, {'sha': 'fixture'}, set(), 0, 5120)
            resumed = indexer.index_part(0, {'sha': 'fixture'}, set(), 0, 5120)
        self.assertEqual(first['members'], resumed['members'])
        self.assertEqual(first['next_offset'], resumed['next_offset'])


if __name__ == '__main__':
    unittest.main()
