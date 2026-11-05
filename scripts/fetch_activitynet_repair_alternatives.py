#!/usr/bin/env python3
"""Retrieve and qualify bounded alternative sources for unresolved affected IDs."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
from pathlib import Path
import threading
import time

import numpy as np

from extract_video_frames_verified import sha256
from fetch_activitynet_raw_sample import SplitTar, PROBE
from qualify_activitynet_repair import ROOT, WORK, process_one, now, save

LOCAL = threading.local()


def fetch_member(video, member, archive_metadata):
    directory = WORK/'alternative_raw_videos'
    directory.mkdir(exist_ok=True)
    target = directory/(video+Path(member['name']).suffix)
    receipt_path = target.with_suffix('.receipt.json')
    if target.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if target.stat().st_size != member['size'] or sha256(target) != receipt['sha256']:
            raise ValueError('Alternative source failed its receipt')
        return receipt_path
    if member['size'] > 256*1024*1024:
        raise ValueError('Alternative member exceeds bounded 256 MiB per-video size')
    if not hasattr(LOCAL, 'remote'):
        LOCAL.remote = SplitTar(archive_metadata)
    remote = LOCAL.remote
    temporary = target.with_suffix('.partial')
    with temporary.open('wb') as handle:
        for position in range(0, member['size'], 4*1024*1024):
            size = min(4*1024*1024, member['size']-position)
            content = remote.read(member['offset']+position, size)
            if len(content) != size:
                raise ValueError('Alternative source size mismatch')
            handle.write(content)
    if temporary.stat().st_size != member['size']:
        raise ValueError('Alternative member incomplete')
    temporary.replace(target)
    receipt = {'video_id': video, 'created_at': now(),
               'status': 'DOWNLOADED_PENDING_QUALIFICATION',
               'source': 'friedrichor/ActivityNet_Captions',
               'revision': archive_metadata['sha'], 'archive_member': member['name'],
               'archive_body_offset': member['offset'], 'bytes': member['size'],
               'path': str(target.relative_to(ROOT)), 'sha256': sha256(target),
               'author_raw_file_identity_verified': False}
    save(receipt_path, receipt)
    return receipt_path


def attempt(video, members, metadata, durations, policy_sha, archive_metadata):
    # Same-ID duplicate archive members need manual resolution rather than a
    # hidden arbitrary choice. The published archive normally has one file/ID.
    distinct = {(m['name'], m['size'], m['offset']) for m in members}
    if len(distinct) != 1:
        return {'video_id': video, 'status': 'AMBIGUOUS_ARCHIVE_MEMBERS', 'candidates': members}
    try:
        receipt = fetch_member(video, members[0], archive_metadata)
        result = process_one(video, receipt, metadata, durations, policy_sha)
        return {'video_id': video, 'status': result['status'], 'qualification': result,
                'receipt': str(receipt.relative_to(ROOT))}
    except Exception as exc:
        return {'video_id': video, 'status': 'ALTERNATIVE_FETCH_FAILED',
                'reason': f'{type(exc).__name__}: {str(exc)[:240]}'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--watch-index', action='store_true')
    args = parser.parse_args()
    primary_status = json.loads((WORK/'qualification_status.json').read_text())
    if primary_status['status'] != 'AVAILABLE_SOURCE_QUALIFICATION_COMPLETE':
        raise ValueError('Primary source qualification must finish first')
    affected = set(json.loads((ROOT/'reports/28_activitynet_sampling_coverage.json').read_text())['suspected_fallback_video_ids'])
    primary = json.loads((WORK/'qualification_results.json').read_text())
    accepted = {v for v, r in primary.items() if r['status'] == 'QUALIFIED_REPAIRED'}
    targets = affected-accepted
    archive_metadata = json.loads((PROBE/'friedrichor__ActivityNet_Captions.json').read_text())
    metadata = {r['video_id']: r for r in map(json.loads, (ROOT/'data/activitynet/test/video_npy_with_meta/meta.jsonl').open())}
    grouped = {}
    for row in map(json.loads, (ROOT/'data/activitynet/raw_annotation/test.jsonl').open()):
        grouped.setdefault(row['video'], []).append(row['duration'])
    durations = {v: float(np.median(d)) for v, d in grouped.items()}
    policy_sha = sha256(WORK/'qualification_policy.json')
    index_root = WORK/'archive_index_parallel'
    results, scheduled, running = {}, set(), {}

    def checkpoint(status):
        save(WORK/'alternative_source_attempts.json', {
            'updated_at': now(), 'status': status, 'target_videos': len(targets),
            'attempted_videos': len(results),
            'qualified_alternative_videos': sum(r['status'] == 'QUALIFIED_REPAIRED' for r in results.values()),
            'attempts': results, 'model_execution': False, 'gpu_access': False,
            'source_revision': archive_metadata['sha'],
        })

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            candidate_path = index_root/'candidates.json'
            candidates = json.loads(candidate_path.read_text()) if candidate_path.exists() else {}
            for video in sorted(targets & set(candidates)):
                if video not in scheduled:
                    scheduled.add(video)
                    future = pool.submit(attempt, video, candidates[video], metadata, durations,
                                         policy_sha, archive_metadata)
                    running[future] = video
            if running:
                done, _ = concurrent.futures.wait(running, timeout=2,
                                                   return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    video = running.pop(future)
                    result = future.result()
                    results[video] = result
                    print(f'[{len(results)}/{len(targets)}] {video} {result["status"]}', flush=True)
            checkpoint('ALTERNATIVE_ATTEMPTS_RUNNING')
            status_path = index_root/'status.json'
            index_status = json.loads(status_path.read_text()) if status_path.exists() else {}
            index_done = index_status.get('status') == 'INDEX_ATTEMPTS_COMPLETE'
            if not running and (index_done or not args.watch_index):
                for video in targets-set(results):
                    results[video] = {'video_id': video, 'status': 'NO_ARCHIVE_MEMBER_FOUND',
                                      'index_status': index_status,
                                      'note': 'No source found within this pinned mirror search; not evidence that the original video is damaged.'}
                checkpoint('ALTERNATIVE_ATTEMPTS_COMPLETE')
                return
            if not running:
                time.sleep(3)


if __name__ == '__main__':
    main()
