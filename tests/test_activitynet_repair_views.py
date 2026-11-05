"""Filtered comparisons must keep identical corpora and original query identity."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import build_activitynet_repair_views as builder
from extract_video_frames_verified import sha256
from activitynet_repair_time import FIELD


class RepairViewTests(unittest.TestCase):
    def test_query_and_corpus_filtering_preserves_dense_embedding_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root/'data/activitynet'
            work = root/'cache/activitynet_repair_v1'
            target = root/'data/activitynet_repair_v1'
            for path in ('raw_annotation', 'test/query_embedding', 'test/video_embedding_1fps',
                         'test/video_npy_with_meta'):
                (canonical/path).mkdir(parents=True)
            (work/'qualification').mkdir(parents=True)
            (target/'repaired_video_npy_with_meta').mkdir(parents=True)
            (root/'reports').mkdir()
            builder.save(work/'download_status.json', {'status': 'DOWNLOAD_ATTEMPTS_COMPLETE'})
            builder.save(work/'alternative_source_attempts.json', {
                'status': 'ALTERNATIVE_ATTEMPTS_COMPLETE', 'attempts': {'C': {'status': 'NO_ACCEPTED_SOURCE'}}})
            builder.save(root/'reports/28_activitynet_sampling_coverage.json',
                         {'suspected_fallback_video_ids': ['B', 'C']})
            annotations = [{'video': video, 'desc_id': str(i), 'fig_desc': f'query {i}',
                            'duration': 4, 'time': [0, 2]} for i, video in enumerate(('A', 'B', 'C', 'C'))]
            builder.lines(canonical/'raw_annotation/test.jsonl', annotations)
            mapping = {'C': 0, 'A': 1, 'B': 2}
            builder.save(canonical/'test/video_embedding_1fps/docid2row.json', mapping)
            video_vectors = np.eye(4, dtype=np.float32)[:3]
            np.save(canonical/'test/video_embedding_1fps/segment_embeds.npy', video_vectors)
            builder.lines(canonical/'test/video_embedding_1fps/meta.jsonl',
                          [{'video_id': v} for v in mapping])
            query_order = [2, 0, 3, 1]
            queries = [{'qid': f'{i}_fig', 'query': f'query {i}', 'pos_doc_id': annotations[i]['video']}
                       for i in query_order]
            builder.lines(canonical/'test/query_embedding/query_meta.test.jsonl', queries)
            query_vectors = np.array([[i, i, i, i] for i in query_order], dtype=np.float32)
            np.save(canonical/'test/query_embedding/query_embeddings.test.npy', query_vectors)
            metadata = []
            original_hashes = {}
            for video in mapping:
                path = canonical/f'test/video_npy_with_meta/{video}.npy'
                np.save(path, np.zeros((2, 4, 4, 3), dtype=np.uint8))
                original_hashes[video] = sha256(path)
                metadata.append({'video_id': video, 'output_npy_path': f'test/video_npy_with_meta/{video}.npy',
                                 'output_num_frames': 2, 'output_height': 4, 'output_width': 4,
                                 'raw_fps': 25, 'target_sample_fps': 1, 'source_duration_sec': 4,
                                 'sampled_timestamps_sec': [0, 3.96]})
            builder.lines(canonical/'test/video_npy_with_meta/meta.jsonl', metadata)
            repaired = target/'repaired_video_npy_with_meta/B.npy'
            np.save(repaired, np.full((4, 4, 4, 3), 10, dtype=np.uint8))
            repaired_meta = {**next(r for r in metadata if r['video_id'] == 'B'),
                             'output_num_frames': 4, 'sampled_timestamps_sec': [0, 1.32, 2.64, 3.96]}
            meta_path = repaired.with_suffix('.metadata.json')
            builder.save(meta_path, repaired_meta)
            builder.save(work/'qualification/B.json', {
                'video_id': 'B', 'status': 'QUALIFIED_REPAIRED',
                'npy_path': str(repaired.relative_to(root)), 'npy_sha256': sha256(repaired),
                'metadata_path': str(meta_path.relative_to(root))})
            with patch.object(builder, 'ROOT', root), patch.object(builder, 'CANONICAL', canonical), \
                    patch.object(builder, 'WORK', work), patch.object(builder, 'TARGET', target), \
                    patch.object(sys, 'argv', ['build', '--finalize']):
                builder.main()
            manifest = json.loads((target/'cohort_manifest.json').read_text())
            self.assertEqual(manifest['retained_original_indices'], [0, 1])
            self.assertEqual(manifest['excluded_video_ids'], ['C'])
            self.assertEqual(manifest['excluded_query_count'], 2)
            expected_video_rows = video_vectors[[1, 2]]
            np.testing.assert_array_equal(np.load(target/'common/test/video_embedding_1fps/segment_embeds.npy'),
                                          expected_video_rows)
            np.testing.assert_array_equal(np.load(target/'common/test/query_embedding/query_embeddings.test.npy'),
                                          query_vectors[[1, 3]])
            for name in manifest['variants']:
                view = target/'views'/name/'activitynet'
                self.assertEqual(json.loads((view/'test/video_embedding_1fps/docid2row.json').read_text()), {'A': 0, 'B': 1})
                self.assertFalse((view/'test/video_npy_with_meta/C.npy').exists())
                rows = builder.jsonl(view/'raw_annotation/test.jsonl')
                self.assertEqual([r['original_index'] for r in rows], [0, 1])
            a = target/'views/original_subset/activitynet/test/video_npy_with_meta/B.npy'
            b = target/'views/pixels_repaired_subset/activitynet/test/video_npy_with_meta/B.npy'
            self.assertEqual(np.load(a).shape[0], 2)
            self.assertEqual(np.load(b).shape[0], 4)
            clock_rows = builder.jsonl(target/'views/source_time_repaired_subset/activitynet/test/video_npy_with_meta/meta.jsonl')
            self.assertTrue(all(FIELD in row for row in clock_rows))
            plain_rows = builder.jsonl(target/'views/pixels_repaired_subset/activitynet/test/video_npy_with_meta/meta.jsonl')
            self.assertTrue(all(FIELD not in row for row in plain_rows))
            for video, digest in original_hashes.items():
                self.assertEqual(sha256(canonical/f'test/video_npy_with_meta/{video}.npy'), digest)


if __name__ == '__main__':
    unittest.main()
