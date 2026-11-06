#!/usr/bin/env python3
"""Fetch pinned official runtime snapshots with the separate HF tools environment."""
from __future__ import annotations
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import quote
from zoneinfo import ZoneInfo
from http_assets import download

ROOT = Path(__file__).resolve().parents[1]


def prepare(name: str):
    meta = json.loads((ROOT / f'cache/manifests/{name}.json').read_text())
    repo, revision = meta['id'], meta['sha']
    target = ROOT / 'checkpoints' / repo.replace('/', '__')
    print(f'[model] {repo}@{revision} -> {target}', flush=True)
    # The proxy omits metadata headers expected by snapshot_download. Use
    # revision-pinned HTTPS files and verify size plus official blob/LFS hashes.
    for row in sorted(meta['siblings'], key=lambda r: r['size']):
        rel = row['rfilename']
        if rel.startswith('eval/'):
            continue
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size == row['size']:
            continue
        temporary = path.with_name(path.name + '.partial')
        endpoint = os.environ.get('HF_ENDPOINT', 'https://hf.co').rstrip('/')
        url = f'{endpoint}/{repo}/resolve/{revision}/{quote(rel, safe="/")}'
        if row.get('lfs') and row['size'] > 64*1024*1024:
            download(url, path, row['size'], row['lfs']['sha256'])
            continue
        subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--retry-all-errors',
                        '--connect-timeout', '20', '--speed-limit', '10240', '--speed-time', '60',
                        '--continue-at', '-', '--silent', '--show-error', '--output', str(temporary), url], check=True)
        assert temporary.stat().st_size == row['size'], temporary
        temporary.replace(path)
        print(f'[downloaded] {repo}/{rel}: {row["size"]} bytes', flush=True)
    files = []
    for row in meta['siblings']:
        rel = row['rfilename']
        if rel.startswith('eval/'):
            continue
        path = target / rel
        assert path.is_file(), path
        assert path.stat().st_size == row['size'], (path, row['size'])
        with path.open('rb') as handle:
            sha = hashlib.file_digest(handle, 'sha256').hexdigest()
        if row.get('lfs'):
            assert sha == row['lfs']['sha256'], path
        elif row.get('blobId'):
            h = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    h.update(block)
            assert h.hexdigest() == row['blobId'], path
        files.append({'path': rel, 'bytes': path.stat().st_size, 'sha256': sha})
    with (target / 'model.safetensors').open('rb') as handle:
        size = int.from_bytes(handle.read(8), 'little')
        header = json.loads(handle.read(size))
    config = json.loads((target / 'config.json').read_text())
    return {'repo': repo, 'revision': revision, 'path': str(target), 'files': files,
            'config': config, 'tensor_count': len(header) - ('__metadata__' in header),
            'refine_keys': [k for k in header if any(x in k for x in ('refine_', 'query_embedder'))],
            'has_query_embedder_dir': (target / 'query_embedder').is_dir()}


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        models = list(pool.map(prepare, ['model', 'embedder']))
    output = {'timestamp': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'models': models, 'verified': True,
              'note': 'Published eval predictions excluded; runtime files verified against official metadata.'}
    (ROOT / 'reports/03_models.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({m['repo']: {'revision': m['revision'], 'files': len(m['files']),
                                  'refine_keys': m['refine_keys']} for m in models}, indent=2), flush=True)


if __name__ == '__main__':
    main()
