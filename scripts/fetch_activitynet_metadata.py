#!/usr/bin/env python3
"""Fetch public ActivityNet metadata; freeze the Stage2 revision before assets."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import quote, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import httpx

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'cache/manifests/activitynet'
REPO = 'VideoSearchR1/activitynet-stage2'


def get_json(url):
    """Only official HTTPS endpoints; retain TLS verification, bound retries."""
    parts = urlsplit(url)
    if parts.scheme != 'https' or parts.hostname not in {'hf.co', 'huggingface.co'}:
        raise ValueError('Unexpected metadata endpoint')
    for attempt in range(8):
        host = 'hf.co' if attempt % 2 == 0 else 'huggingface.co'
        actual = urlunsplit(parts._replace(netloc=host))
        try:
            with httpx.Client(proxy=os.environ.get('HTTPS_PROXY'), follow_redirects=True,
                              timeout=httpx.Timeout(45, connect=12)) as client:
                response = client.get(actual)
                response.raise_for_status()
                return response.json(), response.links.get('next', {}).get('url')
        except httpx.HTTPStatusError as error:
            if error.response.status_code not in {408, 429, 500, 502, 503, 504}:
                raise
        except httpx.TransportError:
            pass
        print(f'[metadata transport retry] {parts.path}: {attempt + 1}/8', flush=True)
        if attempt == 7:
            raise RuntimeError(f'Metadata transfer exhausted: {parts.path}')
        time.sleep(2)


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    META.mkdir(parents=True, exist_ok=True)
    model_path = META / 'model.json'
    if model_path.exists():
        model = json.loads(model_path.read_text())
    else:
        discovery, _ = get_json(f'https://hf.co/api/models/{REPO}?blobs=true')
        revision = discovery['sha']
        model, _ = get_json(f'https://hf.co/api/models/{REPO}/revision/{revision}?blobs=true')
        if model['sha'] != revision:
            raise ValueError('Pinned model metadata changed revision')
        save(model_path, model)
    if model['id'] != REPO or len(model['sha']) != 40 or model.get('private') or model.get('gated'):
        raise ValueError('Unexpected or gated ActivityNet checkpoint')
    revision = model['sha']
    urls = {
        'model.json': f'https://hf.co/api/models/{REPO}/revision/{revision}?blobs=true',
        'stage2_config.json': f'https://hf.co/{REPO}/resolve/{revision}/config.json',
        'manifest.json': 'https://hf.co/buckets/VideoSearchR1/data/resolve/' + quote('datasets/activitynet/manifest.json', safe=''),
    }
    for name in ('stage2_config.json', 'manifest.json'):
        path = META / name
        if not path.exists():
            value, _ = get_json(urls[name])
            save(path, value)
        print('[metadata cached]', name, flush=True)
    config = json.loads((META/'stage2_config.json').read_text())
    manifest = json.loads((META/'manifest.json').read_text())
    if manifest['dataset'] != 'activitynet' or config['text_config']['hidden_size'] != 2048:
        raise ValueError('Unexpected dataset or Stage2 architecture')
    bucket_path = META / 'bucket_shards.json'
    if not bucket_path.exists():
        values = []
        url = 'https://hf.co/api/buckets/VideoSearchR1/data/tree/' + quote('datasets/activitynet/shards', safe='') + '?recursive=true'
        while url:
            page, url = get_json(url)
            if not isinstance(page, list):
                raise ValueError('Unexpected bucket tree response')
            values.extend(page)
            print(f'[bucket listing] {len(values)} entries', flush=True)
        save(bucket_path, values)
    bucket = json.loads(bucket_path.read_text())
    urls['bucket_shards.json'] = 'https://hf.co/api/buckets/VideoSearchR1/data/tree/datasets%2Factivitynet%2Fshards?recursive=true'
    sources = {
        'recorded_at': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'urls': urls, 'bucket_prefix': 'hf://buckets/VideoSearchR1/data/datasets/activitynet/shards',
        'revision': revision,
        'bucket_is_revisionless': True,
        'local_metadata_sha256': {name: hashlib.sha256((META/name).read_bytes()).hexdigest() for name in urls},
    }
    save(META/'sources.json', sources)
    print(json.dumps({'revision': revision, 'model_runtime_bytes': sum(r['size'] for r in model['siblings']
                      if not r['rfilename'].startswith('eval/')), 'manifest_bytes': manifest['total_size'],
                      'manifest_files': manifest['total_files'], 'logical_shards': len(manifest['shards']),
                      'physical_bucket_entries': len(bucket)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
