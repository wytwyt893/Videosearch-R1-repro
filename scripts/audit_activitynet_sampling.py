#!/usr/bin/env python3
"""Audit released sampling indices against the independent retrieval metadata.

No model execution and no changes to released assets. Metadata agreement is not
proof of actual raw-video content or timestamps; same-version originals are
required for that final verification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/activitynet'


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    return {key: float(value) for key, value in zip(
        ('min', 'median', 'max'), np.quantile(a, (0, .5, 1)))} if len(a) else {}


def inspect_record(meta, embedding, query_count):
    info = embedding['info']
    requested = list(meta['requested_indices_src'])
    sampled = list(meta['sampled_indices_src'])
    count = int(meta['sampled_frames_before_even_pad'])
    declared = int(meta['source_total_frames'])
    reference = int(info['num_total'])
    if count != len(requested) or not requested or reference <= 0:
        raise ValueError(f'Invalid sampling record: {meta["video_id"]}')
    source_uniform = np.linspace(0, reference-1, count).astype(np.int32).tolist()
    own_uniform = np.linspace(0, declared-1, count).astype(np.int32).tolist()
    formula_count = min(declared, int(meta['target_max_frames']), max(
        1, math.floor(float(meta['source_duration_sec'])*float(meta['target_sample_fps']))))
    padded = requested + ([requested[-1]] if meta['even_pad_applied'] else [])
    same_path = meta['source_video_path'] == embedding['video_path']
    fallback_pattern = (declared == count == 64 and requested == list(range(64))
                        and reference > 64 and same_path)
    full_range = requested[0] == 0 and requested[-1] == reference-1
    # A single selected frame cannot span endpoints. This is a separate short-
    # video policy, not evidence that the unknown-frame fallback was used.
    short_single = count == 1 and declared == reference and formula_count == 1
    status = ('SUSPECTED_UNKNOWN_FRAME_FALLBACK' if fallback_pattern else
              'SHORT_VIDEO_ONE_SELECTED_FRAME' if short_single else
              'FULL_RANGE_UNIFORM' if (declared == reference and full_range
                                       and requested == source_uniform) else
              'REVIEW_REQUIRED')
    array_path = DATA / meta['output_npy_path']
    array = np.load(array_path, mmap_mode='r', allow_pickle=False)
    shape = list(array.shape)
    header_matches = (shape == [meta['output_num_frames'], meta['output_height'],
                              meta['output_width'], 3] and str(array.dtype) == 'uint8')
    del array
    return {
        'video': meta['video_id'], 'extension': Path(meta['source_video_path']).suffix,
        'status': status, 'positive_query_count': query_count,
        'npy_recorded_source_frames': declared, 'retrieval_recorded_source_frames': reference,
        'selected_frames_before_pad': count, 'stored_frames_after_pad': len(sampled),
        'first_source_index': requested[0], 'last_source_index': requested[-1],
        'source_frame_span_fraction_vs_retrieval': (requested[-1]-requested[0])/(reference-1)
            if reference > 1 else 1.0,
        'npy_last_source_timestamp_seconds': float(meta['sampled_timestamps_sec'][-1]),
        'npy_recorded_source_duration_seconds': float(meta['source_duration_sec']),
        'retrieval_source_duration_by_count_fps_seconds': reference/float(info['fps']),
        'source_paths_match': same_path,
        'source_frame_counts_match': declared == reference,
        'source_fps_match': math.isclose(float(meta['raw_fps']), float(info['fps']), rel_tol=1e-5),
        'uniform_against_own_recorded_count': requested == own_uniform,
        'uniform_against_independent_retrieval_count': requested == source_uniform,
        'sample_count_matches_official_normal_formula': count == formula_count,
        'actual_indices_match_requests_and_even_pad': sampled == padded,
        'timestamps_match_recorded_source_indices_fps': all(
            math.isclose(t, idx/float(meta['raw_fps']), abs_tol=1e-8)
            for t, idx in zip(meta['sampled_timestamps_sec'], sampled))
            and len(meta['sampled_timestamps_sec']) == len(sampled),
        'npy_header_matches_metadata': header_matches,
    }


def audit(output, report):
    if output.exists() or report.exists():
        raise ValueError('Choose fresh output and report paths')
    inputs = {
        'npy_metadata': DATA/'test/video_npy_with_meta/meta.jsonl',
        'retrieval_metadata': DATA/'test/video_embedding_1fps/meta.jsonl',
        'annotations': DATA/'raw_annotation/test.jsonl',
        'official_extractor': ROOT/'upstream/preprocess/extract_npy/extract_npy.py',
        'official_metadata_loader': ROOT/'upstream/videosearch_r1/utils/video_metadata.py',
    }
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs.values()}
    metadata = read_jsonl(inputs['npy_metadata'])
    embeddings_list = read_jsonl(inputs['retrieval_metadata'])
    embeddings = {r['video_id']: r for r in embeddings_list}
    annotations = read_jsonl(inputs['annotations'])
    by_video = Counter(r['video'] for r in annotations)
    if (len(embeddings) != len(embeddings_list) or
            len({r['video_id'] for r in metadata}) != len(metadata) or
            {r['video_id'] for r in metadata} != set(embeddings) or
            set(by_video) - set(embeddings)):
        raise ValueError('Duplicate IDs or missing corpus mappings')
    rows = [inspect_record(m, embeddings[m['video_id']], by_video[m['video_id']])
            for m in metadata]
    groups = {}
    for name in sorted({r['status'] for r in rows}):
        members = [r for r in rows if r['status'] == name]
        groups[name] = {
            'videos': len(members), 'video_percentage': 100*len(members)/len(rows),
            'positive_queries': sum(r['positive_query_count'] for r in members),
            'positive_query_percentage': 100*sum(r['positive_query_count'] for r in members)/len(annotations),
            'selected_frame_count_distribution': dict(sorted(Counter(
                r['selected_frames_before_pad'] for r in members).items())),
            'source_frame_span_fraction_vs_retrieval': describe([
                r['source_frame_span_fraction_vs_retrieval'] for r in members]),
            'last_recorded_source_timestamp_seconds': describe([
                r['npy_last_source_timestamp_seconds'] for r in members]),
        }
    mp4 = [r for r in rows if r['extension'] == '.mp4']
    single_ids = {r['video'] for r in rows if r['status'] == 'SHORT_VIDEO_ONE_SELECTED_FRAME'}
    checks = {key: sum(bool(r[key]) for r in rows) for key in (
        'source_paths_match', 'source_frame_counts_match', 'source_fps_match',
        'uniform_against_own_recorded_count', 'uniform_against_independent_retrieval_count',
        'sample_count_matches_official_normal_formula',
        'actual_indices_match_requests_and_even_pad',
        'timestamps_match_recorded_source_indices_fps', 'npy_header_matches_metadata')}
    result = {
        'created_at': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'status': 'RELEASED_SAMPLING_METADATA_AUDIT_COMPLETE',
        'gpu_access': False, 'model_execution': False, 'raw_source_videos_decoded': False,
        'scope': 'All released ActivityNet test sampling records and NPY headers; no pixel/full raw-video comparison.',
        'counts': {'videos': len(rows), 'queries': len(annotations),
                   'mp4': len(mp4), 'mkv': sum(r['extension'] == '.mkv' for r in rows)},
        'groups': groups, 'checks_passed_videos': checks,
        'mp4': {'videos': len(mp4), 'selected_64': sum(r['selected_frames_before_pad'] == 64 for r in mp4),
                'selected_fewer_than_64': sum(r['selected_frames_before_pad'] < 64 for r in mp4),
                'all_match_normal_sampling_formula': all(r['sample_count_matches_official_normal_formula'] for r in mp4),
                'all_match_uniform_indices_vs_retrieval': all(r['uniform_against_independent_retrieval_count'] for r in mp4)},
        'suspected_fallback_video_ids': [r['video'] for r in rows if r['status'] == 'SUSPECTED_UNKNOWN_FRAME_FALLBACK'],
        'single_frame_video_ids': sorted(single_ids),
        'single_frame_examples': [{
            'video': m['video_id'], 'source_total_frames': m['source_total_frames'],
            'recorded_source_duration_seconds': m['source_duration_sec'],
            'annotations': [{'index': i, 'annotation_duration': a['duration'], 'time': a['time']}
                            for i, a in enumerate(annotations) if a['video'] == m['video_id']],
            'note': 'Sampling count follows the normal formula, but GT/source duration compatibility still needs verification.',
        } for m in metadata if m['video_id'] in single_ids],
        'review_required_video_ids': [r['video'] for r in rows if r['status'] == 'REVIEW_REQUIRED'],
        'interpretation': {
            'evidence_limit': 'Uniformity refers to stored frame indices. It does not prove decoded pixels, raw-file completeness, PTS uniformity or exact historical extraction code.',
            'unknown_frame_count': 'PyAV stream.frames == 0 means unknown, not necessarily a damaged or empty video. Release records do not retain the original zero value.',
            'query_scope': 'Positive-query percentage counts GT videos; other queries can also retrieve an affected candidate.',
            'header_validation_limit': 'Only NPY headers checked here; whole-file SHA and shapes were checked by the earlier CPU preflight.',
            'time_axis_separate': 'Released NPY loader uses nominal target FPS and sequential indices; full-range pixel selection does not resolve original-source timestamp alignment.',
            'source_availability': 'Current prepared-data download contains extracted arrays, not original MKV/MP4 files; originals are required to regenerate missing visual content.',
        },
        'input_sha256': hashes,
    }
    for path in inputs.values():
        if digest(path) != hashes[str(path.relative_to(ROOT))]:
            raise ValueError(f'Input changed during audit: {path}')
    output.mkdir(parents=True)
    (output/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    with (output/'per_video.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('status', 'counts', 'groups', 'mp4', 'checks_passed_videos')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    audit(args.output, args.report)
