#!/usr/bin/env python3
"""Validate full test corpus mappings while leaving video tensors on demand."""
import collections
import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/charades-sta'
sys.path.insert(0, str(ROOT/'upstream/videosearch_r1'))
from utils.video_metadata import load_video_meta_index, resolve_video_meta_for_video_path

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

annotation = rows(DATA/'raw_annotation/test.jsonl')
queries = rows(DATA/'test/query_embedding/query_meta.test.jsonl')
q = np.load(DATA/'test/query_embedding/query_embeddings.test.npy', allow_pickle=False)
v = np.load(DATA/'test/video_embedding_1fps/segment_embeds.npy', allow_pickle=False)
mapping = json.loads((DATA/'test/video_embedding_1fps/docid2row.json').read_text())
indexed = {m['name'] for p in (ROOT/'cache/manifests/tar_index').glob('*.json')
           for m in json.loads(p.read_text())['members']}
assert all(f'test/video_npy_with_meta/{vid}.npy' in indexed or
           f'test/video_npy_with_meta/{vid}.npz' in indexed for vid in mapping)
assert q.ndim == v.ndim == 2 and q.shape[1] == v.shape[1]
assert q.shape[0] == len(queries) == len(annotation)
assert len(mapping) == v.shape[0] and set(mapping.values()) == set(range(len(v)))
assert np.isfinite(q).all() and np.isfinite(v).all()
assert (np.linalg.norm(q, axis=1) > 0).all() and (np.linalg.norm(v, axis=1) > 0).all()
query_map = {r['query']: (i, r) for i, r in enumerate(queries)}
duplicates = {k:n for k,n in collections.Counter(r['query'] for r in queries).items() if n>1}
assert len({r['qid'] for r in queries}) == len(queries)
assert all(a['fig_desc'] in query_map for a in annotation)
assert all(a['video'] in mapping for a in annotation)
for a in annotation[:5]:
    assert a['fig_desc'] not in duplicates
    _, row = query_map[a['fig_desc']]
    assert row['pos_doc_id'] == a['video']
    assert len(a['time']) == 2 and 0 <= a['time'][0] <= a['time'][1] <= a['duration']+0.1
meta = load_video_meta_index(str(DATA/'test/video_npy_with_meta/meta.jsonl'))
missing_meta = [vid for vid in mapping if not resolve_video_meta_for_video_path(
    str(DATA/f'test/video_npy_with_meta/{vid}.npy'), meta)]
assert not missing_meta, missing_meta
index = faiss.read_index(str(DATA/'test/index/index.faiss'))
assert index.ntotal == len(v) and index.d == v.shape[1]
indices = [query_map[a['fig_desc']][0] for a in annotation[:5]]
q5 = q[indices].astype(np.float32)
vn = v.astype(np.float32)
q5 /= np.linalg.norm(q5, axis=1, keepdims=True)
vn /= np.linalg.norm(vn, axis=1, keepdims=True)
reverse = {row:doc for doc,row in mapping.items()}
initial = [reverse[int(i)] for i in (q5 @ vn.T).argmax(axis=1)]
(ROOT/'cache/manifests/smoke_initial_video_paths.json').write_text(json.dumps(
    sorted({f'test/video_npy_with_meta/{vid}.npy' for vid in initial}), indent=2)+'\n')
result = {'timestamp': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'passed': True,
          'annotation_rows': len(annotation), 'query_vectors_shape': list(q.shape),
          'video_vectors_shape': list(v.shape), 'faiss_ntotal': index.ntotal,
          'duplicate_query_strings': len(duplicates), 'video_metadata_complete': True,
          'smoke_indices': list(range(5)), 'smoke_initial_videos': initial,
          'full_test_corpus_preserved': True, 'video_tensors_policy': 'Fetch candidate tensors on demand.'}
(ROOT/'reports/03_data_validation.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
