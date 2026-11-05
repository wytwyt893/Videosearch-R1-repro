#!/usr/bin/env python3
"""Download the affected ActivityNet sources from the pinned public index.

No model/GPU execution; receipts describe a public mirror, not author-file
identity. Subsequent decoding/content/timing qualification is mandatory.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from zoneinfo import ZoneInfo

import httpx

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'cache/activitynet_repair_v1'
INDEX = ROOT/'cache/activitynet_raw_probe/matching_raw_video_candidates.json'
REVISION = 'bc9c567ea67be43daf443cddb832ed983e5ccae5'
LOCAL = threading.local()


def now():
    return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def download_one(row):
    video = 'v_'+row['video_id']
    target = WORK/'raw_videos'/(video+'.mp4')
    receipt = target.with_suffix('.receipt.json')
    if receipt.is_file() and target.is_file():
        old = json.loads(receipt.read_text())
        if target.stat().st_size != row['size_bytes'] or sha(target) != old['sha256']:
            raise ValueError(f'Completed download failed its receipt: {video}')
        return old
    old_path = ROOT/'cache/activitynet_raw_probe/TornadoLabs__activitynet'/target.name
    if old_path.is_file() and old_path.with_suffix('.receipt.json').is_file() and not target.exists():
        previous = json.loads(old_path.with_suffix('.receipt.json').read_text())
        if old_path.stat().st_size == row['size_bytes'] and sha(old_path) == previous['sha256']:
            os.link(old_path, target)
    partial = target.with_suffix('.partial')
    failures = []
    if not hasattr(LOCAL, 'client'):
        LOCAL.client = httpx.Client(proxy=os.environ.get('HTTPS_PROXY'), follow_redirects=True,
                                   timeout=httpx.Timeout(40, connect=12))
    for attempt in range(4):
        if target.is_file() and target.stat().st_size == row['size_bytes']:
            result = {'video_id': video, 'status': 'DOWNLOADED_PENDING_QUALIFICATION',
                      'created_at': now(), 'path': str(target.relative_to(ROOT)),
                      'source_url': row['video_url'], 'metadata_revision': REVISION,
                      'bytes': target.stat().st_size, 'sha256': sha(target),
                      'source': 'TornadoLabs/activitynet', 'attempt_failures': failures,
                      'author_raw_file_identity_verified': False}
            save(receipt, result)
            return result
        try:
            start = partial.stat().st_size if partial.exists() else 0
            if start > row['size_bytes']:
                raise ValueError('Partial file exceeds published size')
            if start == row['size_bytes']:
                partial.replace(target)
                continue
            url = row['video_url']
            if attempt % 2:
                url = url.replace('https://huggingface.co/', 'https://hf.co/', 1)
            end = row['size_bytes']-1
            with LOCAL.client.stream('GET', url, headers={
                'Accept-Encoding': 'identity', 'Range': f'bytes={start}-{end}',
            }) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{row["size_bytes"]}':
                    raise ValueError('Server did not honor exact Range')
                with partial.open('ab') as handle:
                    for block in response.iter_bytes(256*1024):
                        if handle.tell()+len(block) > row['size_bytes']:
                            raise ValueError('Media response exceeds published size')
                        handle.write(block)
            if partial.stat().st_size != row['size_bytes']:
                raise ValueError('Media response is incomplete')
            partial.replace(target)
            result = {'video_id': video, 'status': 'DOWNLOADED_PENDING_QUALIFICATION',
                      'created_at': now(), 'path': str(target.relative_to(ROOT)),
                      'source_url': row['video_url'], 'metadata_revision': REVISION,
                      'bytes': target.stat().st_size, 'sha256': sha(target),
                      'source': 'TornadoLabs/activitynet', 'attempt_failures': failures,
                      'author_raw_file_identity_verified': False}
            save(receipt, result)
            return result
        except (httpx.HTTPError, ValueError, OSError) as exc:
            failures.append({'attempt': attempt+1, 'error': type(exc).__name__,
                             'message': str(exc).split('For more information')[0][:180]})
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (401, 403, 404, 410):
                break
            if attempt < 3:
                time.sleep(attempt+1)
    return {'video_id': video, 'status': 'DOWNLOAD_FAILED', 'created_at': now(),
            'source_url': row['video_url'], 'attempt_failures': failures,
            'needs_alternative_source': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    WORK.mkdir(exist_ok=True)
    (WORK/'raw_videos').mkdir(exist_ok=True)
    rows = sorted(json.loads(INDEX.read_text()), key=lambda x: x['size_bytes'])
    plan = {'created_at': now(), 'source_revision': REVISION, 'index_sha256': sha(INDEX),
            'target_videos': len(rows), 'declared_total_bytes': sum(x['size_bytes'] for x in rows),
            'workers': args.workers, 'model_execution': False, 'gpu_access': False,
            'qualification_required': True, 'targets': rows}
    plan_path = WORK/'download_plan.json'
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        if previous['index_sha256'] != plan['index_sha256'] or previous['targets'] != plan['targets']:
            raise ValueError('Source inventory changed; create a new repair version')
    else:
        save(plan_path, plan)
    results = {}
    started = time.monotonic()

    def checkpoint(status):
        values = list(results.values())
        save(WORK/'download_status.json', {
            'updated_at': now(), 'status': status, 'target_videos': len(rows),
            'completed_videos': len(values),
            'downloaded_videos': sum(x['status'] == 'DOWNLOADED_PENDING_QUALIFICATION' for x in values),
            'failed_videos': sum(x['status'] == 'DOWNLOAD_FAILED' for x in values),
            'completed_bytes': sum(x.get('bytes', 0) for x in values),
            'elapsed_seconds': time.monotonic()-started,
            'model_execution': False, 'gpu_access': False,
        })
        save(WORK/'download_results.json', results)

    checkpoint('DOWNLOADING')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, row): row for row in rows}
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {'video_id': 'v_'+row['video_id'], 'status': 'DOWNLOAD_FAILED',
                          'error': f'{type(exc).__name__}: {exc}', 'needs_alternative_source': True}
            results[result['video_id']] = result
            checkpoint('DOWNLOADING')
            print(f'[{len(results)}/{len(rows)}] {result["video_id"]} {result["status"]}', flush=True)
    checkpoint('DOWNLOAD_ATTEMPTS_COMPLETE')
    print(json.dumps(json.loads((WORK/'download_status.json').read_text()), indent=2), flush=True)


if __name__ == '__main__':
    main()
