#!/usr/bin/env python3
"""Freeze paired ActivityNet cohorts after bounded source repair attempts.

Retained queries and retrieval corpus are identical across the three views.
Canonical files remain untouched. This is a derived-data comparison, not an
automatic reproduction of the paper's full test-set metrics.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from activitynet_repair_time import FIELD
from extract_video_frames_verified import sha256

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT/'data/activitynet'
WORK = ROOT/'cache/activitynet_repair_v1'
TARGET = ROOT/'data/activitynet_repair_v1'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))


def jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def symlink(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise ValueError(f'Fresh view path required: {target}')
    target.symlink_to(source.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize', action='store_true', required=True)
    parser.add_argument('--review-ledger', type=Path,
                        help='Optional pre-scoring manual/transform source decisions')
    args = parser.parse_args()
    manifest_path = TARGET/'cohort_manifest.json'
    if manifest_path.exists() or (TARGET/'views').exists():
        raise ValueError('Cohort already frozen; use a new dataset version to change it')
    source_status = json.loads((WORK/'download_status.json').read_text())
    if source_status['status'] != 'DOWNLOAD_ATTEMPTS_COMPLETE':
        raise ValueError('Primary source attempts are not complete')
    alternative_ledger_path = WORK/'alternative_source_attempts.json'
    if not alternative_ledger_path.is_file():
        raise ValueError('Alternative-source attempt ledger required before exclusions')
    alternatives = json.loads(alternative_ledger_path.read_text())
    if alternatives.get('status') != 'ALTERNATIVE_ATTEMPTS_COMPLETE':
        raise ValueError('Alternative source attempts are unfinished')
    annotations = jsonl(CANONICAL/'raw_annotation/test.jsonl')
    video_meta = {r['video_id']: r for r in jsonl(CANONICAL/'test/video_npy_with_meta/meta.jsonl')}
    mapping = json.loads((CANONICAL/'test/video_embedding_1fps/docid2row.json').read_text())
    affected = set(json.loads((ROOT/'reports/28_activitynet_sampling_coverage.json').read_text())['suspected_fallback_video_ids'])
    records = defaultdict(list)
    for path in sorted((WORK/'qualification').glob('*.json')):
        row = json.loads(path.read_text())
        records[row['video_id']].append(row)
    if args.review_ledger:
        review = json.loads(args.review_ledger.read_text())
        for row in review['decisions']:
            records[row['video_id']].append(row)
    accepted = {}
    for video, candidates in records.items():
        valid = [r for r in candidates if r['status'] == 'QUALIFIED_REPAIRED']
        if len(valid) > 1:
            raise ValueError(f'Multiple accepted sources require an explicit selection: {video}')
        if valid:
            row = valid[0]
            if sha256(ROOT/row['npy_path']) != row['npy_sha256']:
                raise ValueError(f'Repaired array failed SHA: {video}')
            accepted[video] = row
    if set(accepted)-affected:
        raise ValueError('Repairs outside the authorized affected-video list')
    unresolved = affected-set(accepted)
    attempts = alternatives['attempts']
    for video in unresolved:
        if video not in attempts:
            raise ValueError(f'No completed alternative source decision for {video}')
    retained = set(mapping)-unresolved
    kept_annotations, excluded_annotations = [], []
    index_map = []
    for i, row in enumerate(annotations):
        if row['video'] in retained:
            index_map.append({'subset_index': len(kept_annotations), 'original_index': i,
                              'qid': str(row['desc_id'])+'_fig', 'video': row['video']})
            kept_annotations.append({**row, 'original_index': i})
        else:
            excluded_annotations.append({'original_index': i, 'video': row['video'],
                                         'qid': str(row['desc_id'])+'_fig',
                                         'reason': 'Ground-truth video has no accepted repaired source.'})
    video_order = [v for v in sorted(mapping, key=mapping.get) if v in retained]
    new_mapping = {v: i for i, v in enumerate(video_order)}
    video_vectors = np.load(CANONICAL/'test/video_embedding_1fps/segment_embeds.npy', allow_pickle=False)
    video_rows = [mapping[v] for v in video_order]
    filtered_video_vectors = np.asarray(video_vectors[video_rows], dtype=np.float32)
    query_metadata = jsonl(CANONICAL/'test/query_embedding/query_meta.test.jsonl')
    query_rows = [i for i, r in enumerate(query_metadata) if r['pos_doc_id'] in retained]
    filtered_queries = [query_metadata[i] for i in query_rows]
    query_vectors = np.load(CANONICAL/'test/query_embedding/query_embeddings.test.npy', allow_pickle=False)
    filtered_query_vectors = np.asarray(query_vectors[query_rows], dtype=np.float32)
    if len(filtered_queries) != len(kept_annotations):
        raise ValueError('Filtered annotation/query cardinalities differ')
    qids = {r['qid']: r for r in filtered_queries}
    for row in kept_annotations:
        qid = str(row['desc_id'])+'_fig'
        if (qids[qid]['pos_doc_id'] != row['video'] or
                qids[qid]['query'].strip() != row['fig_desc'].strip()):
            raise ValueError('Filtered query identity mismatch')
    common = TARGET/'common'
    (common/'test/query_embedding').mkdir(parents=True)
    (common/'test/video_embedding_1fps').mkdir(parents=True)
    (common/'test/index').mkdir(parents=True)
    np.save(common/'test/query_embedding/query_embeddings.test.npy', filtered_query_vectors, allow_pickle=False)
    lines(common/'test/query_embedding/query_meta.test.jsonl', filtered_queries)
    np.save(common/'test/video_embedding_1fps/segment_embeds.npy', filtered_video_vectors, allow_pickle=False)
    save(common/'test/video_embedding_1fps/docid2row.json', new_mapping)
    lines(common/'raw_annotation/test.jsonl', kept_annotations)
    lines(common/'test/test_queries.jsonl', filtered_queries)
    index_source_meta = jsonl(CANONICAL/'test/video_embedding_1fps/meta.jsonl')
    lines(common/'test/video_embedding_1fps/meta.jsonl', [r for r in index_source_meta if r['video_id'] in retained])
    import faiss
    index = faiss.IndexFlatIP(filtered_video_vectors.shape[1])
    index.add(filtered_video_vectors)
    faiss.write_index(index, str(common/'test/index/index.faiss'))
    save(common/'test/index/id_map.json', video_order)
    save(common/'original_index_map.json', index_map)
    common_files = [p for p in common.rglob('*') if p.is_file()]
    common_sha = {str(p.relative_to(TARGET)): sha256(p) for p in common_files}
    variants = {}
    for name, repaired_pixels, source_time in (
            ('original_subset', False, False),
            ('pixels_repaired_subset', True, False),
            ('source_time_repaired_subset', True, True)):
        view = TARGET/'views'/name/'activitynet'
        for source in common_files:
            symlink(source, view/source.relative_to(common))
        pixels = view/'test/video_npy_with_meta'
        pixels.mkdir(parents=True)
        rows = []
        for video in video_order:
            old = video_meta[video]
            if repaired_pixels and video in accepted:
                record = accepted[video]
                physical_path = ROOT/record['npy_path']
                row = json.loads((ROOT/record['metadata_path']).read_text())
            else:
                physical_path = CANONICAL/old['output_npy_path']
                row = dict(old)
            link = pixels/(video+'.npy')
            symlink(physical_path, link)
            row.update(video_id=video, output_npy_path=str(link))
            if source_time:
                row[FIELD] = list(row['sampled_timestamps_sec'])
                row['videosearch_timestamp_transport'] = 'integer_microsecond_clock; do_sample_frames=False'
                row['videosearch_timestamp_source'] = ('decoded_source_pts' if video in accepted
                                                       else 'published_source_sampling_metadata')
            rows.append(row)
        lines(pixels/'meta.jsonl', rows)
        variants[name] = {
            'data_root': str(view), 'queries': len(kept_annotations), 'videos': len(video_order),
            'repaired_pixels': repaired_pixels, 'source_timestamps': source_time,
            'new_pixel_video_count': len(accepted) if repaired_pixels else 0,
            'time_adapter_required': source_time,
            'metadata_sha256': sha256(pixels/'meta.jsonl'),
        }
    exclusion_rows = []
    query_counts = Counter(r['video'] for r in annotations)
    for video in sorted(unresolved):
        exclusion_rows.append({'video_id': video, 'queries_excluded': query_counts[video],
                               'primary_qualification': records.get(video, []),
                               'alternative_attempt': attempts[video],
                               'cause': 'No source passed frozen data qualification; not assumed to be physical video corruption.'})
    manifest = {
        'created_at': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'status': 'COHORT_AND_PAIRED_DATA_VIEWS_FROZEN',
        'model_execution': False, 'gpu_access': False, 'model_scores_used_for_selection': False,
        'original_queries': len(annotations), 'original_videos': len(mapping),
        'suspected_video_ids': len(affected), 'repaired_video_ids': len(accepted),
        'excluded_video_ids': sorted(unresolved), 'retained_queries': len(kept_annotations),
        'retained_videos': len(retained), 'excluded_query_count': len(excluded_annotations),
        'selection_policy': 'Exclude unaccepted affected videos from both the retrieval corpus and their positive queries. Use the exact same retained corpus/questions in all paired variants.',
        'paper_comparison': 'A filtered cohort cannot be reported as the original full Table 2 benchmark. Even an unfiltered cohort uses a declared source/sampling/timestamp repair variant.',
        'label_policy': 'Preserve original query text, qids, GT spans and annotation durations; retain original_index mapping. No answer/result-based filtering.',
        'retriever_policy': 'Reuse original released vectors for retained video/query IDs; do not recompute or change model weights.',
        'inference_protocol': {'T': 2, 'N_soft_tokens': 8, 'VER': 'last_verified; valid answer denominator',
                               'terminal_rerank': 'keep official', 'VR': 'final ranking',
                               'VCMR': 'last actually verified/localized video'},
        'variants': variants, 'common_file_sha256': common_sha,
        'retained_original_indices': [r['original_index'] for r in index_map],
        'exclusions': exclusion_rows, 'excluded_queries': excluded_annotations,
        'accepted_source_records': accepted,
        'canonical_inputs': {str(p.relative_to(ROOT)): sha256(p) for p in (
            CANONICAL/'raw_annotation/test.jsonl', CANONICAL/'test/query_embedding/query_meta.test.jsonl',
            CANONICAL/'test/video_npy_with_meta/meta.jsonl', CANONICAL/'test/video_embedding_1fps/docid2row.json')},
        'cpu_preflight_passed': False, 'gpu_smoke_passed': False, 'full_evaluation_started': False,
    }
    save(manifest_path, manifest)
    save(ROOT/'reports/31_activitynet_repair_cohort.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('status', 'repaired_video_ids', 'excluded_video_ids',
        'retained_queries', 'retained_videos', 'excluded_query_count', 'variants')}, indent=2))


if __name__ == '__main__':
    main()
