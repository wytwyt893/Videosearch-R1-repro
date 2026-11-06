#!/usr/bin/env python3
"""Index official uncompressed tar shards using HTTP Range; extract chosen files."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import time
from urllib.parse import quote

import httpx

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / 'cache/manifests/tar_index'
DATA = ROOT / 'data/charades-sta'
BASE = os.environ.get('HF_ENDPOINT', 'https://hf.co').rstrip('/') + '/buckets/VideoSearchR1/data/resolve/'


class RemoteTar:
    def __init__(self, shard):
        self.row = shard
        self.client = httpx.Client(proxy=os.environ.get('HTTPS_PROXY'), follow_redirects=True,
                                   timeout=httpx.Timeout(60, connect=20))
        self.url = BASE + quote(shard['path'], safe='')
        self.parts = shard.get('parts_meta', [shard])
        self.resolved = {}
        self.cache_start = -1
        self.cache = b''

    def read(self, offset, size):
        if self.cache_start <= offset and offset + size <= self.cache_start + len(self.cache):
            return self.cache[offset-self.cache_start:offset-self.cache_start+size]
        stop = min(self.row['size']-1, offset+max(size, 16384)-1)
        pieces, part_start = [], 0
        for index, part in enumerate(self.parts):
            part_end = part_start + part['size']
            begin, end = max(offset, part_start), min(stop+1, part_end)
            if begin < end:
                pieces.append(self.read_part(index, begin-part_start, end-part_start-1))
            part_start = part_end
            if part_start > stop:
                break
        self.cache_start, self.cache = offset, b''.join(pieces)
        assert len(self.cache) == stop-offset+1
        return self.cache[:size]

    def read_part(self, index, start, stop):
        part = self.parts[index]
        url = BASE + quote(part['path'], safe='')
        for attempt in range(4):
            try:
                if index not in self.resolved:
                    head = self.client.head(url)
                    head.raise_for_status()
                    self.resolved[index] = str(head.url)
                with self.client.stream('GET', self.resolved[index], headers={'Range': f'bytes={start}-{stop}'}) as r:
                    if r.status_code in (401, 403):
                        self.resolved.pop(index, None)
                    r.raise_for_status()
                    if r.status_code != 206 or r.headers.get('content-range') != f'bytes {start}-{stop}/{part["size"]}':
                        raise ValueError('Server did not honor exact Range')
                    content = r.read()
                assert len(content) == stop-start+1
                return content
            except Exception:
                if attempt == 3:
                    raise
                self.client.close()
                self.client = httpx.Client(proxy=os.environ.get('HTTPS_PROXY'), follow_redirects=True,
                                           timeout=httpx.Timeout(60, connect=20))
                time.sleep(1+attempt)

    def index(self):
        offset, output, pax = 0, [], {}
        while offset+512 <= self.row['size']:
            block = self.read(offset, 512)
            if block == bytes(512):
                break
            info = tarfile.TarInfo.frombuf(block, 'utf-8', 'strict')
            if info.type in (b'x', b'g'):
                raw = self.read(offset+512, info.size)
                position = 0
                while position < len(raw):
                    space = raw.index(b' ', position)
                    length = int(raw[position:space])
                    key, value = raw[space+1:position+length-1].decode().split('=', 1)
                    pax[key] = value
                    position += length
            elif info.isfile():
                name = pax.get('path', info.name)
                size = int(pax.get('size', info.size))
                assert size == info.size, 'PAX size extension requires explicit handling'
                assert not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
                output.append({'name': name, 'size': size, 'offset': offset+512})
                pax = {}
            else:
                pax = {}
            offset += 512 + ((info.size+511)//512)*512
        return {'shard': self.row, 'members': output}

    def extract(self, member):
        target = DATA / member['name']
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name+'.partial')
        h = hashlib.sha256()
        if target.exists() and target.stat().st_size == member['size']:
            with target.open('rb') as handle:
                sha = hashlib.file_digest(handle, 'sha256').hexdigest()
            return {'path': member['name'], 'bytes': member['size'], 'sha256': sha}
        with temporary.open('wb') as handle:
            remaining, offset = member['size'], member['offset']
            while remaining:
                count = min(8*1024*1024, remaining)
                data = self.read(offset, count)
                assert len(data) == count
                handle.write(data)
                h.update(data)
                remaining -= count
                offset += count
        temporary.replace(target)
        return {'path': member['name'], 'bytes': member['size'], 'sha256': h.hexdigest()}


def index_one(shard):
    path = INDEX / (Path(shard['path']).name+'.json')
    if path.exists():
        result = json.loads(path.read_text())
        assert result['shard'] == shard
        return result
    remote = RemoteTar(shard)
    try:
        result = remote.index()
    except Exception as error:
        print(f'[failed] {Path(shard["path"]).name}: {type(error).__name__}: {error}', flush=True)
        return {'failed': shard['path']}
    finally:
        remote.client.close()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(path)
    print(f'[indexed] {Path(shard["path"]).name}: {len(result["members"])} files', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['index', 'extract'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--shard-limit', type=int, default=0)
    parser.add_argument('--paths-json', type=Path)
    args = parser.parse_args()
    INDEX.mkdir(parents=True, exist_ok=True)
    files = json.loads((ROOT/'cache/manifests/bucket_shards.json').read_text())
    by_path = {r['path']: r for r in files}
    manifest = json.loads((ROOT/'cache/manifests/manifest.json').read_text())
    shards = []
    for entry in manifest['shards']:
        prefix = 'datasets/charades-sta/'
        shard = dict(by_path[prefix+'shards/'+entry['name']])
        if entry.get('parts'):
            shard['parts_meta'] = [by_path[prefix+p] for p in entry['parts']]
            shard['size'] = sum(p['size'] for p in shard['parts_meta'])
            shard['xetHash'] = None  # Manifest selects parts; the legacy whole tar is a different asset.
        shards.append(shard)
    shards.sort(key=lambda r: r['path'])
    if args.shard_limit:
        shards = shards[:args.shard_limit]
    if args.action == 'index':
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(index_one, shards))
        failures = [r['failed'] for r in results if 'failed' in r]
        if failures:
            raise RuntimeError(f'Incomplete tar index; rerun to resume failed shards: {failures}')
        print(f'[index complete] {len(results)} shards, {sum(len(r["members"]) for r in results)} files', flush=True)
    else:
        requested = set(json.loads(args.paths_json.read_text()))
        found, receipts = set(), []
        for path in sorted(INDEX.glob('*.json')):
            index = json.loads(path.read_text())
            selected = [m for m in index['members'] if m['name'] in requested]
            if not selected:
                continue
            remote = RemoteTar(index['shard'])
            try:
                for member in selected:
                    receipt = remote.extract(member)
                    receipts.append({**receipt, 'shard': index['shard']['path'],
                                     'shard_xet_hash': index['shard']['xetHash'],
                                     'parts_meta': index['shard'].get('parts_meta'), 'offset': member['offset']})
                    found.add(member['name'])
                    print(f'[extracted] {member["name"]}: {member["size"]} bytes', flush=True)
            finally:
                remote.client.close()
        assert found == requested, f'Missing files: {requested-found}'
        output = ROOT/'reports/03_data_receipts.json'
        old = json.loads(output.read_text()) if output.exists() else []
        by_name = {r['path']: r for r in old+receipts}
        output.write_text(json.dumps(list(by_name.values()), indent=2)+'\n')


if __name__ == '__main__':
    main()
