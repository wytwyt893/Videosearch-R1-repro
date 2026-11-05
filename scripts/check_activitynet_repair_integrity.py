#!/usr/bin/env python3
"""Repeat full on-disk verification before repaired ActivityNet GPU smoke."""
from __future__ import annotations

from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np

from activitynet_repair_time import FIELD, timestamp_payload
from run_activitynet_full import ROOT, verify_preflight, write_json
from validate_activitynet_repair_cpu import main as processor_preflight

BASE = ROOT/'data/activitynet_repair_v1'
DEST = ROOT/'reports/34_activitynet_repair_integrity.json'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Source scripts/env.sh; this check must remain CPU-only')
    if DEST.exists():
        raise ValueError('Preserve existing integrity report; use a new report version')
    start = time.monotonic()
    stamp = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    output = ROOT/f'outputs/activitynet_repair_integrity_{stamp}'
    output.mkdir()
    states = {}
    result = {'status': 'checking', 'passed': False, 'model_execution': False,
              'gpu_access': False, 'output_dir': str(output)}

    def record(path, expected=None):
        path = Path(path)
        before = path.stat()
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'File changed during verification: {path}')
        if expected is not None and digest != expected:
            raise ValueError(f'SHA256 mismatch: {path}')
        states[str(path.relative_to(ROOT))] = {
            'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns, 'sha256': digest}
        return digest

    try:
        old = verify_preflight()
        for n, (rel, state) in enumerate(old['file_states'].items(), 1):
            record(ROOT/rel, state['sha256'])
            if n % 300 == 0:
                print(f'[integrity] canonical files {n}/{len(old["file_states"])} fully hashed', flush=True)
        result['canonical_files_sha_verified'] = len(old['file_states'])
        manifest_path = BASE/'cohort_manifest.json'
        manifest_sha = record(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        previous = json.loads((ROOT/'reports/32_activitynet_repair_cpu_preflight.json').read_text())
        if manifest_sha != previous['manifest_sha256'] or not previous['passed']:
            raise ValueError('Frozen cohort differs from its CPU qualification')
        repaired = manifest['accepted_source_records']
        frozen_view = manifest['variants']['source_time_repaired_subset']
        frozen_meta = Path(frozen_view['data_root'])/'test/video_npy_with_meta/meta.jsonl'
        record(frozen_meta, frozen_view['metadata_sha256'])
        frozen_rows = {r['video_id']: r for r in map(json.loads, frozen_meta.read_text().splitlines())}
        for n, (video, entry) in enumerate(repaired.items(), 1):
            record(ROOT/entry['raw_video_path'], entry['source_sha256'])
            record(ROOT/entry['npy_path'], entry['npy_sha256'])
            record(ROOT/entry['metadata_path'])
            meta = json.loads((ROOT/entry['metadata_path']).read_text())
            source = entry.get('acquired_source_metadata')
            if source is None:
                if not entry.get('visual_review_accepted'):
                    raise ValueError(f'No frozen source metadata or visual-review decision: {video}')
                source = frozen_rows[video]
            for key in ('source_sha256', 'source_total_frames', 'sampled_indices_src',
                        'sampled_timestamps_sec', 'output_num_frames'):
                if meta[key] != source[key]:
                    raise ValueError(f'Derived NPY metadata differs from the source decision: {video}/{key}')
            indices = meta['sampled_indices_src']
            unpadded = meta['sampled_frames_before_even_pad']
            expected = np.linspace(0, meta['source_total_frames']-1, unpadded).astype(int).tolist()
            if indices[:unpadded] != expected or indices[-1] != meta['source_total_frames']-1:
                raise ValueError(f'Repaired video does not cover its source frame range: {video}')
            timestamp_payload({**meta, FIELD: meta['sampled_timestamps_sec']})
            if n % 50 == 0:
                print(f'[integrity] repaired source + NPY pairs {n}/{len(repaired)} fully hashed', flush=True)
        result.update(repaired_source_files_sha_verified=len(repaired),
                      repaired_arrays_sha_verified=len(repaired),
                      full_source_frame_coverage_verified=len(repaired))
        for rel, digest in manifest['common_file_sha256'].items():
            record(BASE/rel, digest)
        mapping = json.loads((BASE/'common/test/video_embedding_1fps/docid2row.json').read_text())
        links = 0
        timestamp_sources = Counter()
        view_links = []
        for name, variant in manifest['variants'].items():
            data = Path(variant['data_root'])
            pixels = data/'test/video_npy_with_meta'
            meta_path = pixels/'meta.jsonl'
            record(meta_path, variant['metadata_sha256'])
            for rel in manifest['common_file_sha256']:
                relative = Path(rel).relative_to('common')
                link = data/relative
                if not link.is_symlink() or link.resolve(strict=True) != (BASE/rel).resolve():
                    raise ValueError(f'Common-data link changed: {link}')
                view_links.append({'path': str(link.relative_to(ROOT)), 'target': str(link.resolve())})
            rows = [json.loads(x) for x in meta_path.read_text().splitlines() if x.strip()]
            if len(rows) != len(mapping) or {r['video_id'] for r in rows} != set(mapping):
                raise ValueError(f'Metadata has missing/duplicate videos: {name}')
            if {p.stem for p in pixels.glob('*.npy')} != set(mapping) or list(pixels.glob('*.partial')):
                raise ValueError(f'Extra, missing or partial pixel files: {name}')
            for row in rows:
                video = row['video_id']
                expected_path = (ROOT/repaired[video]['npy_path'] if variant['repaired_pixels'] and video in repaired
                                 else ROOT/f'data/activitynet/test/video_npy_with_meta/{video}.npy')
                link = pixels/(video+'.npy')
                if not link.is_symlink() or link.resolve(strict=True) != expected_path.resolve():
                    raise ValueError(f'Pixel view points to an unexpected file: {link}')
                view_links.append({'path': str(link.relative_to(ROOT)), 'target': str(link.resolve())})
                if variant['source_timestamps']:
                    timestamp_payload(row)
                    if row[FIELD] != row['sampled_timestamps_sec']:
                        raise ValueError(f'Wrong timestamp override: {video}')
                    timestamp_sources[row['videosearch_timestamp_source']] += 1
                links += 1
        import faiss
        vectors = np.load(BASE/'common/test/video_embedding_1fps/segment_embeds.npy', allow_pickle=False)
        index = faiss.read_index(str(BASE/'common/test/index/index.faiss'))
        if not np.array_equal(index.reconstruct_n(0, index.ntotal), vectors):
            raise ValueError('FAISS contents differ from retrieval vectors')
        print('[integrity] actual processor, question mapping and paired-view CPU checks', flush=True)
        processor_report = output/'processor_cpu.json'
        processor_preflight(processor_report)
        record(processor_report)
        for rel in ['reports/32_activitynet_repair_cpu_preflight.json',
                    'data/activitynet_repair_v1/evaluation_plans.json',
                    'scripts/validate_activitynet_repair_cpu.py', 'scripts/check_activitynet_repair_integrity.py',
                    'scripts/activitynet_repair_time.py', 'scripts/activitynet_repaired_eval.py',
                    'scripts/score_activitynet_protocol.py']:
            record(ROOT/rel)
        write_json(output/'file_states.json', states)
        write_json(output/'view_links.json', view_links)
        result.update(status='FULL_REPAIR_INTEGRITY_PASSED', passed=True,
                      created_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                      manifest_sha256=manifest_sha, video_view_links_verified=links,
                      videos=manifest['retained_videos'], queries=manifest['retained_queries'],
                      excluded_videos=manifest['excluded_video_ids'], source_timestamp_counts=dict(timestamp_sources),
                      faiss_vector_contents_verified=True, processor_report=str(processor_report),
                      file_states=str(output/'file_states.json'), view_links=str(output/'view_links.json'),
                      file_states_sha256=record(output/'file_states.json'),
                      view_links_sha256=record(output/'view_links.json'),
                      known_limitations={
                          'original_annotation_end_after_duration': old['data_observations']['annotation_end_after_duration_count'],
                          'short_source_with_longer_annotations': ['v_0dkIbKXXFzI'],
                          'source_byte_identity_to_author_not_proven': True,
                          'low_fps_alternative_sources': 67,
                          'explicit_visual_review_exceptions': 2,
                          'unrepaired_videos_source_time_based_on_published_metadata': 4502,
                          'policy': 'Retain original benchmark labels and all questions; do not infer data correctness from smoke scores.'},
                      elapsed_seconds=round(time.monotonic()-start, 2))
        write_json(DEST, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        result.update(status='FAILED', error=str(error), elapsed_seconds=round(time.monotonic()-start, 2))
        write_json(output/'failure.json', result)
        raise


if __name__ == '__main__':
    main()
