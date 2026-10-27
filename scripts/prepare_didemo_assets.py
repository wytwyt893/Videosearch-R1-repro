#!/usr/bin/env python3
"""Download pinned DiDeMo Stage2 and complete test assets, without GPU use.

Requires previously fetched official metadata in cache/manifests/didemo.
Charades files and frozen evaluation settings are never modified.
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
from zoneinfo import ZoneInfo

import numpy as np

from prepare_models import prepare
from remote_tar import RemoteTar

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'cache/manifests/didemo'
DATA = ROOT / 'data/didemo'
WORK = ROOT / 'cache/didemo_download'
REVISION = 'a576c16040c35a5ec71468c66ec7628e7db71997'
MUTEX = threading.RLock()
STATE = {}


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def update(phase, **values):
    with MUTEX:
        STATE.setdefault(phase, {}).update(values)
        STATE['updated_at'] = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        save(WORK / 'status.json', STATE)


class DidemoTar(RemoteTar):
    def extract(self, member):
        target = DATA / member['name']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size == member['size']:
            return {'path': member['name'], 'bytes': member['size'], 'sha256': digest(target)}
        temporary = target.with_name(target.name + '.partial')
        offset, remaining = member['offset'], member['size']
        sha = hashlib.sha256()
        with temporary.open('wb') as handle:
            while remaining:
                count = min(8 * 1024**2, remaining)
                block = self.read(offset, count)
                if len(block) != count:
                    raise ValueError('Incomplete official tar range')
                handle.write(block)
                sha.update(block)
                offset += count
                remaining -= count
        if temporary.stat().st_size != member['size']:
            raise ValueError('Unexpected downloaded size')
        temporary.replace(target)
        return {'path': member['name'], 'bytes': member['size'], 'sha256': sha.hexdigest()}


def canonical_shards():
    manifest = json.loads((META / 'manifest.json').read_text())
    bucket = {r['path']: r for r in json.loads((META / 'bucket_shards.json').read_text())}
    shards = []
    for entry in manifest['shards']:
        path = 'datasets/didemo/shards/' + entry['name']
        if not entry.get('parts') and path not in bucket:
            raise ValueError(f'Manifest-selected whole shard is missing: {path}')
        # Some logical tars only exist as parts; no legacy whole object exists.
        row = dict(bucket.get(path, {'type': 'file', 'path': path,
                                    'size': entry['size'], 'xetHash': None}))
        if entry.get('parts'):
            row['parts_meta'] = [bucket['datasets/didemo/' + p] for p in entry['parts']]
            row['size'] = sum(p['size'] for p in row['parts_meta'])
            row['xetHash'] = None
        shards.append(row)
    return sorted(shards, key=lambda row: row['path'])


def index_one(shard):
    path = META / 'tar_index' / (Path(shard['path']).name + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        result = json.loads(path.read_text())
        if result['shard'] != shard:
            raise ValueError(f'Index changed: {path.name}')
        return result
    remote = DidemoTar(shard)
    try:
        result = remote.index()
    finally:
        remote.client.close()
    save(path, result)
    print(f'[index] {path.name}: {len(result["members"])} members', flush=True)
    return result


def prepare_model():
    meta = json.loads((META / 'model.json').read_text())
    config = json.loads((META / 'stage2_config.json').read_text())
    if (meta['sha'] != REVISION or config['text_config']['hidden_size'] != 2048
            or config['text_config']['num_hidden_layers'] != 28):
        raise ValueError('Unexpected Stage2 revision / 2B architecture')
    update('models', status='downloading', revision=REVISION,
           expected_runtime_bytes=sum(r['size'] for r in meta['siblings'] if not r['rfilename'].startswith('eval/')))
    result = prepare('didemo/model')
    if len(result['refine_keys']) != 16:
        raise ValueError('Unexpected Stage2 refine tensor count')
    save(WORK / 'models.json', result)
    update('models', status='complete', revision=REVISION, refine_tensors=16,
           verified_files=len(result['files']), receipt='cache/didemo_download/models.json')
    print('[models complete] fixed DiDeMo Stage2, official sizes/hashes verified', flush=True)


def prepare_data(workers):
    update('data', status='indexing_core', source='hf://buckets/VideoSearchR1/data/datasets/didemo',
           scope='Complete test assets only; no training videos.')
    shards = canonical_shards()
    indices = [index_one(shards[0])]
    core = [(indices[0], member) for member in indices[0]['members'] if
            member['name'] == 'raw_annotation/test.jsonl' or
            (member['name'].startswith('test/') and not
             (member['name'].startswith('test/video_npy_with_meta/') and
              Path(member['name']).suffix in {'.npy', '.npz'}))]
    receipt_path = WORK / 'data_receipts.jsonl'
    receipts = {}
    if receipt_path.exists():
        receipts = {r['path']: r for r in
                    (json.loads(line) for line in receipt_path.read_text().splitlines() if line.strip())}

    def extract(index, member):
        remote = DidemoTar(index['shard'])
        try:
            result = remote.extract(member)
        finally:
            remote.client.close()
        old = receipts.get(member['name'])
        if old and result['sha256'] != old['sha256']:
            raise ValueError(f'Existing receipt hash mismatch: {member["name"]}')
        result.update(shard=index['shard']['path'], offset=member['offset'],
                      shard_xet_hash=index['shard']['xetHash'], parts_meta=index['shard'].get('parts_meta'))
        with MUTEX:
            with receipt_path.open('a') as handle:
                handle.write(json.dumps(result) + '\n')
            receipts[member['name']] = result
        return result

    for index, member in core:
        extract(index, member)
        print(f'[core] {member["name"]}: {member["size"]} bytes', flush=True)
    expected_core = ['raw_annotation/test.jsonl', 'test/query_embedding/query_embeddings.test.npy',
                     'test/query_embedding/query_meta.test.jsonl',
                     'test/video_embedding_1fps/segment_embeds.npy',
                     'test/video_embedding_1fps/docid2row.json', 'test/video_npy_with_meta/meta.jsonl']
    if any(not (DATA / p).is_file() for p in expected_core):
        raise ValueError('Expected test core files were not all in the first logical shard')
    mapping = json.loads((DATA / expected_core[4]).read_text())
    if any(Path(doc).name != doc for doc in mapping) or set(mapping.values()) != set(range(len(mapping))):
        raise ValueError('Invalid test corpus mapping')
    wanted = {f'test/video_npy_with_meta/{doc}.npy' for doc in mapping}
    found = {m['name'] for i in indices for m in i['members'] if m['name'] in wanted}
    for start in range(1, len(shards), workers):
        if found == wanted:
            break
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            batch = list(pool.map(index_one, shards[start:start+workers]))
        indices.extend(batch)
        found.update(m['name'] for i in batch for m in i['members'] if m['name'] in wanted)
        update('data', status='indexing_test', expected_videos=len(wanted), indexed_test_videos=len(found),
               logical_shards_indexed=len(indices))
        print(f'[test index] {len(found)}/{len(wanted)} videos located', flush=True)
    if found != wanted:
        raise ValueError('Incomplete test-video tar index')
    jobs = [(index, member) for index in indices for member in index['members'] if member['name'] in wanted]
    if len(jobs) != len(wanted):
        raise ValueError('Duplicate test video tar members')
    update('data', status='downloading_test', expected_videos=len(wanted),
           expected_video_bytes=sum(m['size'] for _, m in jobs), completed_videos=0, completed_video_bytes=0)

    def fetch(job):
        index, member = job
        result = extract(index, member)
        with MUTEX:
            state = STATE['data']
            update('data', completed_videos=state['completed_videos']+1,
                   completed_video_bytes=state['completed_video_bytes']+result['bytes'])
            if state['completed_videos'] % 10 == 0:
                print(f'[pixels] {state["completed_videos"]}/{len(wanted)}; '
                      f'{state["completed_video_bytes"]/2**30:.2f} GiB', flush=True)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, job): job[1]['name'] for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append({'path': futures[future], 'error': f'{type(error).__name__}: {str(error)[:200]}'})
                update('data', failures=failures)
    if failures:
        raise ValueError(f'{len(failures)} test video downloads failed; rerun to resume complete files')
    update('data', status='validating')
    metadata = {r['video_id']: r for r in (json.loads(line) for line in
                (DATA / expected_core[5]).read_text().splitlines() if line.strip())}
    if set(metadata) != set(mapping):
        raise ValueError('Video metadata does not cover exact test corpus')
    for index, member in jobs:
        path = DATA / member['name']
        meta = metadata[path.stem]
        arr = np.load(path, allow_pickle=False, mmap_mode='r')
        expected = (meta['output_num_frames'], meta['output_height'], meta['output_width'], 3)
        valid = arr.shape == expected and arr.dtype == np.uint8 and arr.flags.c_contiguous
        arr._mmap.close()
        if not valid or path.stat().st_size != member['size'] or digest(path) != receipts[member['name']]['sha256']:
            raise ValueError(f'Pixel array or SHA256 validation failed: {path.name}')
    annotations = [json.loads(line) for line in (DATA / expected_core[0]).read_text().splitlines() if line.strip()]
    queries = np.load(DATA / expected_core[1], allow_pickle=False)
    videos = np.load(DATA / expected_core[3], allow_pickle=False)
    if (queries.shape != (len(annotations), 2048) or videos.shape != (len(mapping), 2048)
            or not np.isfinite(queries).all() or not np.isfinite(videos).all()):
        raise ValueError('Invalid complete test vectors')
    update('data', status='complete', expected_queries=len(annotations),
           validated_videos=len(wanted), query_vectors=list(queries.shape), video_vectors=list(videos.shape),
           integrity_scope='Official manifest-selected exact ranges and sizes; local SHA256 receipts; NPY shape/dtype/metadata. No official per-video digest is published.')
    print(f'[data complete] {len(annotations)} test queries, {len(mapping)} video pixels/vectors', flush=True)


def main():
    global STATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('models', 'data', 'all'), default='all')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('Use 1-8 workers')
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / 'download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        STATE = json.loads((WORK/'status.json').read_text()) if (WORK/'status.json').exists() else {}
        STATE.update(status='running', pid=os.getpid(), gpu_used=False, model_execution=False,
                     phase=args.phase, started_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        update('run', status='running')
        tasks = {'models': prepare_model, 'data': lambda: prepare_data(args.workers)}
        if args.phase != 'all':
            tasks = {args.phase: tasks[args.phase]}
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(task): phase for phase, task in tasks.items()}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    errors.append(f'{type(error).__name__}: {str(error)[:300]}')
                    update(futures[future], status='failed', error=errors[-1])
                    print(f'[failed] {errors[-1]}', flush=True)
        STATE.update(status='failed' if errors else 'complete', errors=errors)
        update('run', status=STATE['status'])
        if errors:
            raise RuntimeError('; '.join(errors))
        print('[complete] DiDeMo asset preparation finished, no model execution', flush=True)


if __name__ == '__main__':
    main()
