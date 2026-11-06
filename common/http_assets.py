"""Bounded, resumable HTTP Range downloads with final official SHA256 verification."""
from __future__ import annotations
import concurrent.futures
import hashlib
import os
from pathlib import Path
import threading
import time

import httpx


def download(url, target: Path, size: int, sha256: str, workers=4):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        with target.open('rb') as handle:
            if target.stat().st_size == size and hashlib.file_digest(handle, 'sha256').hexdigest() == sha256:
                return
        raise ValueError(f'Existing completed asset failed verification: {target}')
    chunk_size = 16*1024*1024
    parts = target.with_name(target.name+'.chunks')
    parts.mkdir(exist_ok=True)
    count = (size+chunk_size-1)//chunk_size
    previous = target.with_name(target.name+'.partial')
    # Reuse complete prefix chunks left by the previous official curl download.
    if previous.exists():
        with previous.open('rb') as handle:
            for index in range(min(count, previous.stat().st_size//chunk_size)):
                path = parts/f'{index:05d}'
                data = handle.read(chunk_size)
                if not path.exists():
                    path.write_bytes(data)
    local = threading.local()
    clients = []
    lock = threading.Lock()
    with httpx.Client(proxy=os.environ.get('HTTPS_PROXY'), follow_redirects=True, timeout=60) as client:
        r = client.head(url)
        r.raise_for_status()
        resolved = str(r.url)

    def one(index):
        start = index*chunk_size
        end = min(size-1, start+chunk_size-1)
        path = parts/f'{index:05d}'
        if path.is_file() and path.stat().st_size == end-start+1:
            return
        if not hasattr(local, 'client'):
            local.client = httpx.Client(proxy=os.environ.get('HTTPS_PROXY'), follow_redirects=True,
                                       timeout=httpx.Timeout(90, connect=20))
            with lock:
                clients.append(local.client)
        for attempt in range(4):
            temporary = path.with_suffix('.tmp')
            try:
                with local.client.stream('GET', resolved if attempt == 0 else url,
                                         headers={'Range': f'bytes={start}-{end}'}) as response:
                    response.raise_for_status()
                    assert response.status_code == 206
                    assert response.headers.get('content-range') == f'bytes {start}-{end}/{size}'
                    with temporary.open('wb') as handle:
                        for block in response.iter_bytes(1024*1024):
                            handle.write(block)
                assert temporary.stat().st_size == end-start+1
                temporary.replace(path)
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt+1)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for done, _ in enumerate(pool.map(one, range(count)), 1):
                if done % 16 == 0 or done == count:
                    print(f'[chunks] {target.name}: {done}/{count}', flush=True)
    finally:
        for client in clients:
            client.close()
    temporary = target.with_name(target.name+'.assembling')
    digest = hashlib.sha256()
    with temporary.open('wb') as output:
        for index in range(count):
            with (parts/f'{index:05d}').open('rb') as source:
                for block in iter(lambda: source.read(1024*1024), b''):
                    output.write(block)
                    digest.update(block)
    assert temporary.stat().st_size == size
    assert digest.hexdigest() == sha256, f'Official SHA256 mismatch: {target}'
    temporary.replace(target)
    print(f'[verified] {target.name}: {size} bytes SHA256 OK', flush=True)
