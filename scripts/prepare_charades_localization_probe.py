#!/usr/bin/env python3
"""Prepare a train-only, correct-initial-retrieval localization probe on CPU."""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime
import json
import os
from pathlib import Path
import random
from zoneinfo import ZoneInfo

import numpy as np

from audit_charades_annotations import tags
from charades_timestamp_eval import source_payload
from inspect_charades_localization_cases import ROOT, jsonl, save, sha
from remote_tar import RemoteTar, index_one

DATA = ROOT / 'data/charades-sta'
SEED = 20261107
REPORT = ROOT / 'reports/47_charades_localization_probe_prepare.json'
BASE_FILES = ['raw_annotation/train.jsonl', 'train/video_npy_with_meta/meta.jsonl',
              'train/query_embedding/query_embeddings.train.npy', 'train/query_embedding/query_meta.train.jsonl',
              'train/video_embedding_1fps/segment_embeds.npy', 'train/video_embedding_1fps/docid2row.json']


def canonical():
    bucket = {r['path']: r for r in json.loads((ROOT / 'cache/manifests/bucket_shards.json').read_text())}
    manifest = json.loads((ROOT / 'cache/manifests/manifest.json').read_text())
    result = {}
    for entry in manifest['shards']:
        row = dict(bucket['datasets/charades-sta/shards/' + entry['name']])
        if entry.get('parts'):
            row['parts_meta'] = [bucket['datasets/charades-sta/' + p] for p in entry['parts']]
            row['size'] = sum(p['size'] for p in row['parts_meta'])
            row['xetHash'] = None
        result[entry['name']] = row
    return result


def inventory():
    allowed = {r['path']: r for r in canonical().values()}
    result = {}
    for p in sorted((ROOT / 'cache/manifests/tar_index').glob('*.json')):
        index = json.loads(p.read_text())
        if index['shard'] != allowed[index['shard']['path']]:
            raise ValueError(f'Manifest mismatch: {p}')
        for member in index['members']:
            if member['name'] in result:
                raise ValueError('Duplicate indexed member')
            result[member['name']] = dict(member=member, shard=index['shard'], index_sha256=sha(p))
    return result


def extract(entry):
    remote = RemoteTar(entry['shard'])
    try:
        receipt = remote.extract(entry['member'])
    finally:
        remote.client.close()
    receipt.update(shard=entry['shard'], offset=entry['member']['offset'], index_sha256=entry['index_sha256'])
    print(f'[asset] {receipt["path"]}: {receipt["bytes"]} bytes', flush=True)
    return receipt


def fetch_base():
    found = inventory()
    if BASE_FILES[1] not in found:
        result = index_one(canonical()['charades-sta-00107.tar'])
        if 'failed' in result:
            raise RuntimeError('Train metadata shard index failed')
        found = inventory()
    missing = set(BASE_FILES) - found.keys()
    if missing:
        raise ValueError(f'Missing base asset indices: {missing}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(extract, [found[p] for p in BASE_FILES]))
    destination = ROOT / 'cache/localization_probe'
    destination.mkdir(exist_ok=True)
    save(destination / 'base_receipts.json', receipts)


def select():
    if REPORT.exists():
        report = json.loads(REPORT.read_text())
        print(json.dumps({'status': report['status'], 'output': report['output'], 'reuse_frozen_selection': True}))
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Keep CUDA hidden for CPU preparation')
    annotations = jsonl(DATA / 'raw_annotation/train.jsonl')
    metadata = {r['video_id']: r for r in jsonl(DATA / 'train/video_npy_with_meta/meta.jsonl')}
    test_vids = {r['video'] for r in jsonl(DATA / 'raw_annotation/test.jsonl')}
    found = inventory()
    qmeta = jsonl(DATA / 'train/query_embedding/query_meta.train.jsonl')
    query_by_qid = {r['qid']: (i, r) for i, r in enumerate(qmeta)}
    query_counts = {}
    for r in annotations:
        query_counts[r['fig_desc']] = query_counts.get(r['fig_desc'], 0) + 1
    video_rows = json.loads((DATA / 'train/video_embedding_1fps/docid2row.json').read_text())
    videos = np.asarray(np.load(DATA / 'train/video_embedding_1fps/segment_embeds.npy'), dtype=np.float32)
    queries = np.asarray(np.load(DATA / 'train/query_embedding/query_embeddings.train.npy'), dtype=np.float32)
    videos /= np.maximum(np.linalg.norm(videos, axis=1, keepdims=True), 1e-12)
    queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    eligible = []
    for i, r in enumerate(annotations):
        video, start, end = r['video'], *r['time']
        before, after = tags(r['cog_desc']), tags(r['fig_desc'])
        path = f'train/video_npy_with_meta/{video}.npy'
        if (video in test_vids or video not in metadata or path not in found or video not in video_rows
                or query_counts[r['fig_desc']] != 1 or not before or not before <= after
                or len(before) > 2 or len(after - before) > 1 or not 3 <= end - start <= 20
                or not 0 <= start < end <= min(r['duration'], metadata[video]['source_duration_sec'])
                or end - start > .8 * metadata[video]['source_duration_sec']):
            continue
        qid = str(r['desc_id']) + '_fig'
        if qid not in query_by_qid or query_by_qid[qid][1]['query'] != r['fig_desc']:
            continue
        eligible.append(i)
    random.Random(SEED).shuffle(eligible)
    selected, seen = [], set()
    evaluated = 0
    for i in eligible:
        r = annotations[i]
        if r['video'] in seen:
            continue
        q_row, _ = query_by_qid[str(r['desc_id']) + '_fig']
        scores = videos @ queries[q_row]
        evaluated += 1
        row = int(video_rows[r['video']])
        if int(np.argmax(scores)) != row:
            continue
        sorted_scores = np.sort(scores)
        selected.append(dict(index=i, qid=str(r['desc_id']) + '_fig', video_id=r['video'],
                             query=r['fig_desc'], coarse_description=r['cog_desc'], gt=r['time'],
                             annotation_duration=r['duration'], metadata=metadata[r['video']],
                             initial_rank=1, initial_top1_video=r['video'],
                             initial_score=float(scores[row]), top1_margin=float(sorted_scores[-1] - sorted_scores[-2]),
                             group='dev' if len(selected) < 12 else 'holdout',
                             pixel_path=str(DATA / f'train/video_npy_with_meta/{r["video"]}.npy')))
        seen.add(r['video'])
        if len(selected) == 36:
            break
    if len(selected) != 36:
        raise ValueError(f'Only {len(selected)} eligible correct-top1 cases')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    output = ROOT / f'outputs/charades_localization_probe_{stamp}'
    output.mkdir()
    report = dict(status='FROZEN_SELECTION_PENDING_PIXELS', output=str(output), seed=SEED,
                  selection=selected, development_count=12, holdout_count=24,
                  train_rows=len(annotations), train_videos=len(metadata), train_retrieval_corpus=len(videos),
                  eligible_before_initial_rank_filter=len(eligible), initial_retrievals_evaluated=evaluated,
                  selection_before_new_inference=True, no_official_test_video_overlap=True,
                  sampling_scope='One query per video; train-only within manifest-indexed pixel assets; explicit text/span quality filters and initial full-train-corpus top1=GT.',
                  evaluation_scope='Conditional correct-initial-video localization probe, not full T=2/N=8 VCMR. Original verification is frozen after initial generation; no SQR is invoked.',
                  holdout_limit='Video-disjoint split within training data; the released checkpoint may have trained on these videos. Not an unseen-video benchmark.',
                  source_bindings={p: sha(DATA / p) for p in BASE_FILES},
                  manifest_sha256=sha(ROOT / 'cache/manifests/manifest.json'))
    save(output / 'selection.json', report)
    save(REPORT, report)
    print(json.dumps({'status': report['status'], 'output': str(output), 'selected': len(selected)}, ensure_ascii=False))


def fetch_pixels():
    report = json.loads(REPORT.read_text())
    found = inventory()
    paths = [f'train/video_npy_with_meta/{r["video_id"]}.npy' for r in report['selection']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(extract, [found[p] for p in paths]))
    for row in report['selection']:
        path, meta = Path(row['pixel_path']), row['metadata']
        arr = np.load(path, mmap_mode='r', allow_pickle=False)
        assert list(arr.shape) == [meta['output_num_frames'], meta['output_height'], meta['output_width'], 3]
        assert arr.dtype == np.uint8
        source_payload(meta)
        row['pixel_sha256'] = sha(path)
    report['status'] = 'PIXELS_READY_PENDING_CPU_PROCESSOR_CHECK'
    report['pixel_receipts'] = receipts
    save(Path(report['output']) / 'assets.json', report)
    save(REPORT, report)
    print(json.dumps({'status': report['status'], 'videos': len(receipts), 'total_pixel_bytes': sum(r['bytes'] for r in receipts)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['fetch-base', 'select', 'fetch-pixels'])
    args = parser.parse_args()
    {'fetch-base': fetch_base, 'select': select, 'fetch-pixels': fetch_pixels}[args.action]()
