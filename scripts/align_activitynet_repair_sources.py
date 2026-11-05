#!/usr/bin/env python3
"""Qualify unresolved sources after explicit geometry/low-FPS alignment.

Keep the original RGB thresholds and duration checks. A low-FPS source can
straddle a cut: compare each reference image to the two bracketing source
frames. Restore released image geometry before checking aspect ratio. These
source checks are independent of queries' answers or model scores.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np
from PIL import Image

from extract_video_frames_verified import extract_verified, sha256
from qualify_activitynet_repair import ROOT, WORK, DERIVED, POLICY, validate_source, save, now

ALIGNMENT_POLICY = {
    'version': 2, 'base_numeric_qualification_thresholds': POLICY,
    'geometry': 'Select a fixed whole-video rotation from 0/90/180/270 using prefix correspondence, then resize to released tensor H/W.',
    'temporal_correspondence': 'Choose the best appearance match among the two decoded PTS frames bracketing each reference time; no global shift, speed change or GT modification.',
    'actual_sampling': 'Still uniform full decoded frame range, target 1 FPS capped at 64.',
    'model_scores_used': False, 'author_raw_byte_identity_claimed': False,
}


def choose_alignment(probes, expected_images):
    candidates = []
    for turns in range(4):
        chosen, errors = [], []
        for group, expected in zip(probes['groups'], expected_images):
            reference = np.asarray(Image.fromarray(expected).resize((64, 48), Image.Resampling.BICUBIC), dtype=np.float32)
            options = []
            for index in group:
                rotated = np.rot90(probes['images'][index], turns)
                normalized = np.asarray(Image.fromarray(rotated).resize((64, 48), Image.Resampling.BICUBIC), dtype=np.float32)
                options.append((float(np.abs(normalized-reference).mean()), index))
            error, index = min(options)
            chosen.append(index)
            errors.append(error)
        candidates.append({'rotate_ccw_degrees': turns*90, 'probe_indices': chosen,
                           'median_rgb_mae': float(np.median(errors)),
                           'p90_rgb_mae': float(np.quantile(errors, .9)),
                           'per_probe_rgb_mae': errors})
    return min(candidates, key=lambda r: (r['median_rgb_mae'], r['p90_rgb_mae'], r['rotate_ccw_degrees']))


def process(record, metadata, durations, policy_sha):
    video = record['video_id']
    source = record['source']+'__alignment_v2'
    result_path = WORK/'qualification'/f'{video}__{source}.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['policy_sha256'] != policy_sha:
            raise ValueError('Alignment policy changed')
        return result
    reference = metadata[video]
    result = {'video_id': video, 'source': source, 'created_at': now(),
              'policy_sha256': policy_sha, 'raw_video_path': record['raw_video_path'],
              'source_sha256': record['source_sha256'], 'model_execution': False, 'gpu_access': False,
              'canonical_assets_modified': False}
    try:
        raw = ROOT/record['raw_video_path']
        if sha256(raw) != record['source_sha256']:
            raise ValueError('Source SHA mismatch')
        old_validation = record.get('validation', {})
        if 'duration_mismatch' in old_validation.get('failure_reasons', []):
            result.update(status='SOURCE_REJECTED_PENDING_ALTERNATIVE',
                          failure_reasons=['duration_mismatch_not_repairable_by_geometry_or_bracket_matching'],
                          validation=old_validation)
            save(result_path, result)
            return result
        baseline = np.load(ROOT/'data/activitynet'/reference['output_npy_path'], mmap_mode='r', allow_pickle=False)
        positions = np.linspace(0, len(baseline)-1, 8).round().astype(int)
        expected = [np.asarray(baseline[i]) for i in positions]
        times = [reference['sampled_timestamps_sec'][i] for i in positions]
        probes = {}
        array, info = extract_verified(raw, probe_timestamps_sec=times,
                                       probe_sink=probes, probe_selection='bracket')
        alignment = choose_alignment(probes, expected)
        turns = alignment['rotate_ccw_degrees']//90
        actual = [np.rot90(probes['images'][i], turns) for i in alignment['probe_indices']]
        original_aspect = info['output_width']/info['output_height']
        # Geometry of the actual output is restored to the released tensor.
        # Preserve the original source geometry explicitly rather than pretending
        # it matched before the declared spatial normalization.
        final_geometry = {**info, 'output_height': reference['output_height'],
                          'output_width': reference['output_width']}
        validation = validate_source(final_geometry, durations[video], reference, actual, expected)
        validation.update(source_original_aspect_ratio=original_aspect,
                          reference_aspect_ratio=reference['output_width']/reference['output_height'],
                          source_geometry_normalized=True,
                          temporal_probe_mode='best_of_pts_bracketing_frames',
                          rotation_ccw_degrees=alignment['rotate_ccw_degrees'])
        result.update(validation=validation, alignment=alignment, acquired_source_metadata=info,
                      reference_probe_target_times=times,
                      matched_probe_source_indices=[probes['source_indices'][i] for i in alignment['probe_indices']],
                      matched_probe_source_times=[probes['timestamps_sec'][i] for i in alignment['probe_indices']])
        if not validation['passed']:
            result.update(status='SOURCE_REJECTED_PENDING_ALTERNATIVE', failure_reasons=validation['failure_reasons'])
        else:
            array = np.stack([np.asarray(Image.fromarray(np.rot90(frame, turns)).resize(
                (reference['output_width'], reference['output_height']), Image.Resampling.BICUBIC)) for frame in array])
            path = DERIVED/(video+'.npy')
            if path.exists():
                raise ValueError('Accepted repaired tensor already exists')
            tmp = path.with_suffix('.partial')
            with tmp.open('wb') as f:
                np.save(f, array, allow_pickle=False)
            tmp.replace(path)
            repaired_meta = {**info, 'video_id': video, 'output_npy_path': str(path.relative_to(ROOT)),
                             'output_height': reference['output_height'], 'output_width': reference['output_width'],
                             'source_native_extracted_height': info['output_height'],
                             'source_native_extracted_width': info['output_width'],
                             'source_upscaled_to_release_shape': info['output_height']*info['output_width'] < array.shape[1]*array.shape[2],
                             'source_alignment': alignment, 'source_qualification': validation,
                             'qualification_policy_sha256': policy_sha,
                             'reference_source_path': reference['source_video_path']}
            save(path.with_suffix('.metadata.json'), repaired_meta)
            result.update(status='QUALIFIED_REPAIRED', npy_path=str(path.relative_to(ROOT)),
                          npy_sha256=sha256(path), npy_bytes=path.stat().st_size,
                          metadata_path=str(path.with_suffix('.metadata.json').relative_to(ROOT)),
                          output_frames=int(array.shape[0]), output_shape=list(array.shape),
                          source_upscaled_to_release_shape=repaired_meta['source_upscaled_to_release_shape'])
    except Exception as exc:
        result.update(status='SOURCE_REJECTED_PENDING_ALTERNATIVE', failure_reasons=[f'{type(exc).__name__}: {str(exc)[:200]}'])
    save(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    attempts = json.loads((WORK/'alternative_source_attempts.json').read_text())
    if attempts['status'] != 'ALTERNATIVE_ATTEMPTS_COMPLETE':
        raise ValueError('Finish source attempts before alignment review')
    policy_path = WORK/'alignment_policy_v2.json'
    if policy_path.exists() and json.loads(policy_path.read_text()) != ALIGNMENT_POLICY:
        raise ValueError('Alignment policy changed; use a new version')
    save(policy_path, ALIGNMENT_POLICY)
    policy_sha = sha256(policy_path)
    records = [json.loads(p.read_text()) for p in (WORK/'qualification').glob('*.json')]
    accepted = {r['video_id'] for r in records if r['status'] == 'QUALIFIED_REPAIRED'}
    grouped = {}
    for record in records:
        if record['video_id'] not in accepted and '__alignment_v2' not in record['source']:
            grouped.setdefault(record['video_id'], []).append(record)
    metadata = {r['video_id']: r for r in map(json.loads, (ROOT/'data/activitynet/test/video_npy_with_meta/meta.jsonl').open())}
    annotation_durations = {}
    for row in map(json.loads, (ROOT/'data/activitynet/raw_annotation/test.jsonl').open()):
        annotation_durations.setdefault(row['video'], []).append(row['duration'])
    durations = {v: float(np.median(d)) for v, d in annotation_durations.items()}

    def per_video(pair):
        video, candidates = pair
        # Prefer full-rate primary sources, then the lower-FPS archive.
        candidates.sort(key=lambda r: not r['source'].startswith('TornadoLabs'))
        tried = []
        for record in candidates:
            result = process(record, metadata, durations, policy_sha)
            tried.append(result)
            if result['status'] == 'QUALIFIED_REPAIRED':
                return {'video_id': video, 'status': result['status'], 'attempts': tried}
        return {'video_id': video, 'status': 'NO_SOURCE_PASSED_ALIGNMENT', 'attempts': tried}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(per_video, sorted(grouped.items())):
            results.append(result)
            save(WORK/'alignment_results_v2.json', {'status': 'ALIGNMENT_RUNNING', 'results': results})
            print(result['video_id'], result['status'], flush=True)
    save(WORK/'alignment_results_v2.json', {'status': 'ALIGNMENT_ATTEMPTS_COMPLETE', 'results': results,
                                          'model_execution': False, 'gpu_access': False,
                                          'policy_sha256': policy_sha})


if __name__ == '__main__':
    main()
