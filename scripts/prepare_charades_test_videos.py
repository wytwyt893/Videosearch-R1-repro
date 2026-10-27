#!/usr/bin/env python3
"""Fetch and validate all released Charades test pixel arrays, using CPUs only.

Reuses the existing manifest-aware tar reader. Historical smoke reports, model
weights, annotations, embeddings, official source, and GPU processes are untouched.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from zoneinfo import ZoneInfo

import numpy as np

from remote_tar import RemoteTar

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/charades-sta'
STATUS = ROOT / 'reports/07_charades_test_videos.json'
RECEIPTS = ROOT / 'reports/07_charades_test_video_receipts.jsonl'


class BufferedRemoteTar(RemoteTar):
    """Amortize HTTP overhead across adjacent members without changing bytes."""
    def __init__(self, shard, read_ahead_bytes):
        super().__init__(shard)
        self.read_ahead_bytes = read_ahead_bytes
        self.read_limit = shard['size']

    def read(self, offset, size):
        if self.cache_start <= offset and offset + size <= self.cache_start + len(self.cache):
            return super().read(offset, size)
        count = min(max(size, self.read_ahead_bytes), self.read_limit - offset)
        return super().read(offset, count)[:size]


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save_status(value):
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(STATUS)


def inventory():
    manifest_path = ROOT / 'cache/manifests/manifest.json'
    bucket_path = ROOT / 'cache/manifests/bucket_shards.json'
    manifest = json.loads(manifest_path.read_text())
    bucket = {r['path']: r for r in json.loads(bucket_path.read_text())}
    canonical = {}
    for entry in manifest['shards']:
        row = dict(bucket['datasets/charades-sta/shards/' + entry['name']])
        if entry.get('parts'):
            row['parts_meta'] = [bucket['datasets/charades-sta/' + p] for p in entry['parts']]
            row['size'] = sum(p['size'] for p in row['parts_meta'])
            row['xetHash'] = None
        canonical[row['path']] = row
    mapping = json.loads((DATA / 'test/video_embedding_1fps/docid2row.json').read_text())
    wanted = {f'test/video_npy_with_meta/{v}.npy' for v in mapping}
    if len(wanted) != 1334 or any(Path(v).name != v for v in mapping):
        raise ValueError('Unexpected test corpus ids')
    groups, found = [], set()
    for path in sorted((ROOT / 'cache/manifests/tar_index').glob('*.json')):
        index = json.loads(path.read_text())
        selected = [m for m in index['members'] if m['name'] in wanted]
        if not selected:
            continue
        row = index['shard']
        if row != canonical[row['path']]:
            raise ValueError(f'Index disagrees with manifest-selected assets: {path}')
        for member in selected:
            if member['name'] in found or member['offset'] + member['size'] > row['size']:
                raise ValueError('Duplicate or out-of-bounds tar member')
            found.add(member['name'])
        groups.append({'shard': row, 'members': selected,
                       'index_file': str(path.relative_to(ROOT)), 'index_sha256': digest(path)})
    if found != wanted:
        raise ValueError(f'Test videos missing from indices: {len(wanted - found)}')
    meta_rows = [json.loads(line) for line in
                 (DATA / 'test/video_npy_with_meta/meta.jsonl').read_text().splitlines() if line.strip()]
    meta = {r['video_id']: r for r in meta_rows}
    if len(meta_rows) != len(meta) or set(meta) != set(mapping):
        raise ValueError('Test video metadata does not cover the exact retrieval corpus')
    return groups, meta, {'manifest_sha256': digest(manifest_path),
                          'bucket_metadata_sha256': digest(bucket_path)}


def prior_receipts():
    receipts = {r['path']: r for r in
                json.loads((ROOT / 'reports/03_data_receipts.json').read_text())}
    if RECEIPTS.exists():
        for line in RECEIPTS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                receipts[r['path']] = r
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--read-ahead-mib', type=int, choices=(8, 16, 32, 64), default=32)
    parser.add_argument('--schedule', choices=('shards', 'files'), default='shards')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or (args.plan_only and args.validate_only):
        parser.error('Use 1-8 workers and only one inspection mode')
    groups, metadata, sources = inventory()
    all_members = [m for group in groups for m in group['members']]
    existing = [m for m in all_members if (DATA / m['name']).is_file()
                and (DATA / m['name']).stat().st_size == m['size']]
    total_bytes = sum(m['size'] for m in all_members)
    initial_bytes = sum(m['size'] for m in existing)
    summary = {'expected_videos': len(all_members), 'existing_complete_size': len(existing),
               'missing_videos': len(all_members) - len(existing), 'total_bytes': total_bytes,
               'missing_bytes': total_bytes - initial_bytes, 'logical_shards_needed': len(groups)}
    if args.plan_only:
        print(json.dumps(summary, indent=2))
        return
    lock_path = ROOT / 'cache/charades_test_video_download.lock'
    with lock_path.open('w') as process_lock:
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process_lock.write(str(os.getpid()) + '\n')
        process_lock.flush()
        state = {
            'started': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'pid': os.getpid(), 'status': 'validating' if args.validate_only else 'downloading',
            'gpu_used': False, 'model_execution': False,
            'read_ahead_mib_per_worker': args.read_ahead_mib,
            'download_schedule': args.schedule,
            'source': 'hf://buckets/VideoSearchR1/data/datasets/charades-sta',
            'data_scope': 'Complete test corpus only; no training videos or other datasets.',
            'initial_inventory': summary, 'source_metadata_hashes': sources,
            'sources': [{'shard': g['shard'], 'index_file': g['index_file'],
                         'index_sha256': g['index_sha256']} for g in groups],
            'receipt_file': str(RECEIPTS.relative_to(ROOT)),
            'integrity_scope': 'Exact manifest-selected HTTP ranges, file sizes, local SHA256, and NPY shape/dtype checked. Per-file official digests are not published.',
            'processed_videos': 0, 'processed_bytes': 0, 'failures': [],
        }
        save_status(state)
        print('[plan] ' + json.dumps(summary), flush=True)
        start = time.monotonic()
        mutex = threading.Lock()
        if not args.validate_only:
            jobs = groups
            if args.schedule == 'files':
                known = prior_receipts()
                cached_names = {m['name'] for m in existing if m['name'] in known}
                jobs = [{**group, 'members': [member]}
                        for group in groups for member in group['members']
                        if member['name'] not in cached_names]
                state['processed_videos'] = len(cached_names)
                state['processed_bytes'] = sum(m['size'] for m in all_members if m['name'] in cached_names)
            with RECEIPTS.open('a') as receipts:
                def fetch_group(group):
                    remote = BufferedRemoteTar(group['shard'], args.read_ahead_mib * 1024**2)
                    try:
                        for member in group['members']:
                            try:
                                if args.schedule == 'files':
                                    remote.read_limit = member['offset'] + member['size']
                                receipt = remote.extract(member)
                                receipt.update(shard=group['shard']['path'], offset=member['offset'],
                                               index_sha256=group['index_sha256'])
                                with mutex:
                                    receipts.write(json.dumps(receipt) + '\n')
                                    receipts.flush()
                                    state['processed_videos'] += 1
                                    state['processed_bytes'] += member['size']
                                    state['elapsed_seconds'] = round(time.monotonic() - start, 1)
                                    if state['processed_videos'] % 10 == 0:
                                        save_status(state)
                                        print(f'[download] {state["processed_videos"]}/1334 files; '
                                              f'{state["processed_bytes"] / 2**30:.2f} GiB checked/downloaded; '
                                              f'{state["elapsed_seconds"]:.0f}s', flush=True)
                            except Exception as error:
                                with mutex:
                                    state['failures'].append({'path': member['name'],
                                                              'error': f'{type(error).__name__}: {str(error)[:300]}'})
                                    save_status(state)
                                    print(f'[failed] {member["name"]}: {type(error).__name__}', flush=True)
                    finally:
                        remote.client.close()
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                    for _ in pool.map(fetch_group, jobs):
                        pass
            state['status'] = 'validating'
            save_status(state)
        receipts = prior_receipts()
        checked, checked_bytes, validated_hashes = 0, 0, {}
        for member in all_members:
            path = DATA / member['name']
            if not path.is_file() or path.stat().st_size != member['size']:
                raise ValueError(f'Test video incomplete: {member["name"]}; rerun to resume complete files')
            array = np.load(path, allow_pickle=False, mmap_mode='r')
            meta = metadata[path.stem]
            expected = (meta['output_num_frames'], meta['output_height'], meta['output_width'], 3)
            if (array.shape != expected or array.dtype != np.uint8 or array.flags.f_contiguous
                    or meta['target_sample_fps'] != 1.0 or meta['target_max_frames'] != 64
                    or not 0 < array.shape[0] <= 64 or array.shape[0] % 2):
                raise ValueError(f'Video pixels disagree with official metadata: {path.name}')
            array._mmap.close()
            sha = digest(path)
            receipt = receipts.get(member['name'])
            if receipt is None or sha != receipt['sha256'] or receipt['bytes'] != member['size']:
                raise ValueError(f'Local SHA256 receipt mismatch: {path.name}')
            validated_hashes[member['name']] = sha
            checked += 1
            checked_bytes += member['size']
            if checked % 100 == 0:
                print(f'[validate] {checked}/1334 pixel arrays, metadata and SHA256 OK', flush=True)
        state.update(status='complete', passed=True, validated_videos=checked,
                     validated_bytes=checked_bytes, file_sha256=validated_hashes,
                     elapsed_seconds=round(time.monotonic() - start, 2),
                     completed_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        save_status(state)
        print('[complete] ' + json.dumps({'videos': checked, 'bytes': checked_bytes,
                                         'elapsed_seconds': state['elapsed_seconds'],
                                         'gpu_used': False, 'report': str(STATUS)}), flush=True)


if __name__ == '__main__':
    main()
