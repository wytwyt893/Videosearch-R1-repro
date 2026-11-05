#!/usr/bin/env python3
"""Qualify available sources and create isolated full-range ActivityNet tensors.

Source checks are frozen before model scoring. Failed first-choice sources stay
pending for a separate alternative-source pass; they are not silently skipped.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import math
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

from extract_video_frames_verified import extract_verified, sha256

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'cache/activitynet_repair_v1'
DERIVED = ROOT/'data/activitynet_repair_v1/repaired_video_npy_with_meta'


def now():
    return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def save(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


POLICY = {
    'version': 1, 'target_fps': 1.0, 'max_frames': 64,
    'duration_absolute_tolerance_seconds': 1.0, 'duration_relative_tolerance': 0.01,
    'prefix_probes': 8, 'thumbnail_width': 64, 'thumbnail_height': 48,
    'maximum_median_probe_rgb_mae': 20.0, 'maximum_p90_probe_rgb_mae': 40.0,
    'maximum_aspect_ratio_relative_difference': 0.05,
    'minimum_source_average_decoded_fps': 1.0,
    'output_spatial_shape': 'Preserve each released NPY height/width; record mirror upscaling.',
    'sampling_domain': 'Uniform actual decoded source frame indices, like official normal branch.',
    'time_storage': 'Actual decoded presentation times; inference transport is a separate adapter.',
    'scoring_used_for_selection': False,
    'source_identity_limit': 'ID, approximate duration, aspect ratio and prefix appearance checks do not establish author-file byte identity.',
}


def validate_source(meta, expected_duration, reference, actual_probe_images, expected_probe_images):
    duration = meta['source_duration_sec']
    tolerance = max(POLICY['duration_absolute_tolerance_seconds'],
                    POLICY['duration_relative_tolerance']*expected_duration)
    errors = []
    probe_mae = []
    for actual, expected in zip(actual_probe_images, expected_probe_images):
        shape = (POLICY['thumbnail_width'], POLICY['thumbnail_height'])
        a = np.asarray(Image.fromarray(actual).resize(shape, Image.Resampling.BICUBIC), dtype=np.float32)
        b = np.asarray(Image.fromarray(expected).resize(shape, Image.Resampling.BICUBIC), dtype=np.float32)
        probe_mae.append(float(np.abs(a-b).mean()))
    if not probe_mae:
        raise ValueError('No source correspondence probes')
    median = float(np.median(probe_mae))
    p90 = float(np.quantile(probe_mae, .9))
    source_ratio = meta['output_width']/meta['output_height']
    reference_ratio = reference['output_width']/reference['output_height']
    aspect_difference = abs(source_ratio/reference_ratio-1)
    if abs(duration-expected_duration) > tolerance:
        errors.append('duration_mismatch')
    if median > POLICY['maximum_median_probe_rgb_mae'] or p90 > POLICY['maximum_p90_probe_rgb_mae']:
        errors.append('prefix_content_mismatch_or_alignment_uncertain')
    if aspect_difference > POLICY['maximum_aspect_ratio_relative_difference']:
        errors.append('aspect_ratio_mismatch')
    if meta['source_total_frames']/duration < POLICY['minimum_source_average_decoded_fps']:
        errors.append('source_temporal_resolution_below_1fps')
    return {
        'passed': not errors, 'failure_reasons': errors,
        'reference_annotation_duration_seconds': expected_duration,
        'acquired_source_duration_seconds': duration, 'duration_tolerance_seconds': tolerance,
        'median_probe_rgb_mae': median, 'p90_probe_rgb_mae': p90, 'per_probe_rgb_mae': probe_mae,
        'aspect_ratio_relative_difference': aspect_difference,
        'author_raw_file_byte_identity_verified': False,
    }


def process_one(video, receipt_path, metadata, durations, policy_sha):
    receipt = json.loads(receipt_path.read_text())
    raw_path = ROOT/receipt['path']
    source_name = receipt.get('source', receipt.get('publisher', 'unknown')).replace('/', '__')
    record_path = WORK/'qualification'/f'{video}__{source_name}.json'
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        if record.get('policy_sha256') != policy_sha or record.get('source_sha256') != receipt['sha256']:
            raise ValueError(f'Qualification input/policy changed: {video}')
        if record['status'] == 'QUALIFIED_REPAIRED':
            if sha256(ROOT/record['npy_path']) != record['npy_sha256']:
                raise ValueError(f'Derived NPY changed: {video}')
        return record
    reference = metadata[video]
    record = {'video_id': video, 'created_at': now(), 'policy_sha256': policy_sha,
              'source': source_name, 'raw_video_path': str(raw_path.relative_to(ROOT)),
              'source_sha256': receipt['sha256'], 'model_execution': False, 'gpu_access': False,
              'canonical_assets_modified': False}
    try:
        if sha256(raw_path) != receipt['sha256']:
            raise ValueError('Raw source failed download SHA receipt')
        baseline = np.load(ROOT/'data/activitynet'/reference['output_npy_path'], mmap_mode='r', allow_pickle=False)
        probe_positions = np.linspace(0, len(baseline)-1, POLICY['prefix_probes']).round().astype(int)
        probe_times = [reference['sampled_timestamps_sec'][i] for i in probe_positions]
        expected_images = [np.asarray(baseline[i]) for i in probe_positions]
        probes = {}
        array, info = extract_verified(raw_path, fps=POLICY['target_fps'], max_frames=POLICY['max_frames'],
                                       probe_timestamps_sec=probe_times, probe_sink=probes)
        validation = validate_source(info, durations[video], reference, probes['images'], expected_images)
        del baseline
        record.update(validation=validation, acquired_source_metadata=info,
                      reference_probe_source_indices=probes['source_indices'],
                      reference_probe_source_times=probes['timestamps_sec'])
        if not validation['passed']:
            record.update(status='SOURCE_REJECTED_PENDING_ALTERNATIVE',
                          failure_reasons=validation['failure_reasons'])
        else:
            target_shape = [reference['output_height'], reference['output_width'], 3]
            upscaled = info['output_height']*info['output_width'] < target_shape[0]*target_shape[1]
            if list(array.shape[1:]) != target_shape:
                array = np.stack([np.asarray(Image.fromarray(frame).resize(
                    (target_shape[1], target_shape[0]), Image.Resampling.BICUBIC)) for frame in array])
            target = DERIVED/(video+'.npy')
            # A second candidate for a video is only processed if the first
            # candidate was rejected. Never overwrite an accepted source.
            if target.exists():
                raise ValueError('A repaired target already exists without this qualification receipt')
            temporary = target.with_suffix('.partial')
            with temporary.open('wb') as handle:
                np.save(handle, array, allow_pickle=False)
            temporary.replace(target)
            output_metadata = {**info, 'video_id': video, 'output_npy_path': str(target.relative_to(ROOT)),
                               'output_height': target_shape[0], 'output_width': target_shape[1],
                               'source_native_extracted_height': info['output_height'],
                               'source_native_extracted_width': info['output_width'],
                               'source_upscaled_to_release_shape': upscaled,
                               'qualification_policy_sha256': policy_sha,
                               'source_qualification': validation,
                               'reference_raw_fps': reference['raw_fps'],
                               'reference_source_path': reference['source_video_path']}
            save(target.with_suffix('.metadata.json'), output_metadata)
            record.update(status='QUALIFIED_REPAIRED', npy_path=str(target.relative_to(ROOT)),
                          npy_sha256=sha256(target), npy_bytes=target.stat().st_size,
                          metadata_path=str(target.with_suffix('.metadata.json').relative_to(ROOT)),
                          output_frames=int(array.shape[0]), output_shape=list(array.shape),
                          source_upscaled_to_release_shape=upscaled)
    except Exception as exc:
        record.update(status='SOURCE_REJECTED_PENDING_ALTERNATIVE',
                      failure_reasons=[f'{type(exc).__name__}: {str(exc)[:200]}'])
    save(record_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--video-id')
    args = parser.parse_args()
    WORK.mkdir(exist_ok=True)
    (WORK/'qualification').mkdir(exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    policy_path = WORK/'qualification_policy.json'
    if policy_path.exists() and json.loads(policy_path.read_text()) != POLICY:
        raise ValueError('Qualification policy changed; create a new repair version')
    save(policy_path, POLICY)
    policy_sha = sha256(policy_path)
    metadata = {r['video_id']: r for r in map(json.loads, (ROOT/'data/activitynet/test/video_npy_with_meta/meta.jsonl').open())}
    durations_by_video = {}
    for row in map(json.loads, (ROOT/'data/activitynet/raw_annotation/test.jsonl').open()):
        durations_by_video.setdefault(row['video'], []).append(row['duration'])
    durations = {vid: float(np.median(values)) for vid, values in durations_by_video.items()}
    results, scheduled, running = {}, set(), {}
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            receipts = sorted((WORK/'raw_videos').glob('*.receipt.json'))
            # Alternative-source downloader puts receipts in this directory.
            receipts += sorted((WORK/'alternative_raw_videos').glob('*.receipt.json'))
            for receipt in receipts:
                row = json.loads(receipt.read_text())
                video = row['video_id']
                key = str(receipt)
                if key in scheduled or (args.video_id and video != args.video_id):
                    continue
                # An accepted primary source always wins over a lower-quality
                # archive source. Alternative receipts are added after primary
                # qualification completes, so there is no simultaneous writer.
                if results.get(video, {}).get('status') == 'QUALIFIED_REPAIRED':
                    scheduled.add(key)
                    continue
                if any(v == video for v in running.values()):
                    continue
                scheduled.add(key)
                running[pool.submit(process_one, video, receipt, metadata, durations, policy_sha)] = video
            if running:
                done, _ = concurrent.futures.wait(running, timeout=2,
                                                   return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    video = running.pop(future)
                    result = future.result()
                    results[video] = result
                    print(f'[{len(results)}] {video} {result["status"]} {result.get("failure_reasons", [])}', flush=True)
            count_ok = sum(x['status'] == 'QUALIFIED_REPAIRED' for x in results.values())
            status = {'updated_at': now(), 'status': 'QUALIFYING', 'processed_videos': len(results),
                      'qualified_videos': count_ok, 'pending_alternative_videos': len(results)-count_ok,
                      'running_videos': len(running), 'elapsed_seconds': time.monotonic()-started,
                      'model_execution': False, 'gpu_access': False, 'policy_sha256': policy_sha}
            save(WORK/'qualification_status.json', status)
            save(WORK/'qualification_results.json', results)
            if not running:
                download_status_path = WORK/'download_status.json'
                download_done = download_status_path.exists() and json.loads(download_status_path.read_text())['status'] == 'DOWNLOAD_ATTEMPTS_COMPLETE'
                if not args.watch or args.video_id or download_done:
                    status['status'] = 'AVAILABLE_SOURCE_QUALIFICATION_COMPLETE'
                    save(WORK/'qualification_status.json', status)
                    return
                time.sleep(3)


if __name__ == '__main__':
    main()
