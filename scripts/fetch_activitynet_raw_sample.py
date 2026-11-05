#!/usr/bin/env python3
"""Find one affected ActivityNet MKV in a public, pinned split-tar mirror.

Read tar headers and the selected member via exact HTTP Range; do not download
the whole archive. Outputs only under cache/, never replaces released NPYs.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import time
from zoneinfo import ZoneInfo

import httpx

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT/'cache/activitynet_raw_probe'
REPO = 'friedrichor/ActivityNet_Captions'


class SplitTar:
    def __init__(self, metadata):
        self.revision = metadata['sha']
        self.parts = sorted((x for x in metadata['siblings']
                             if x['rfilename'].startswith('ActivityNet_Videos.tar.part-')),
                            key=lambda x: x['rfilename'])
        self.boundaries = [0]
        for part in self.parts:
            self.boundaries.append(self.boundaries[-1]+part['size'])
        self.client = httpx.Client(proxy=os.environ.get('HTTPS_PROXY'), follow_redirects=True,
                                   timeout=httpx.Timeout(25, connect=12))
        self.resolved = {}
        self.requests = 0
        self.downloaded_bytes = 0
        self.cache_offset, self.cache = -1, b''

    def read_part(self, index, first, last):
        part = self.parts[index]
        error = None
        for attempt in range(3):
            try:
                domain = 'hf.co' if attempt != 1 else 'huggingface.co'
                url = f'https://{domain}/datasets/{REPO}/resolve/{self.revision}/{part["rfilename"]}?download=true&range_attempt={attempt}'
                if index not in self.resolved:
                    head = self.client.head(url)
                    self.requests += 1
                    head.raise_for_status()
                    self.resolved[index] = str(head.url)
                with self.client.stream('GET', self.resolved[index], headers={
                    'Range': f'bytes={first}-{last}', 'Accept-Encoding': 'identity',
                }) as response:
                    self.requests += 1
                    response.raise_for_status()
                    if (response.status_code != 206 or response.headers.get('Content-Range') !=
                            f'bytes {first}-{last}/{part["size"]}'):
                        raise ValueError('Server did not honor the exact requested byte range')
                    data = response.read()
                if len(data) != last-first+1:
                    raise ValueError('Range body size mismatch')
                self.downloaded_bytes += len(data)
                return data
            except (httpx.HTTPError, ValueError) as exc:
                error = exc
                self.resolved.pop(index, None)
                # A new client avoids repeatedly reusing a failed TLS tunnel.
                self.client.close()
                self.client = httpx.Client(proxy=os.environ.get('HTTPS_PROXY'), follow_redirects=True,
                                           timeout=httpx.Timeout(25, connect=12))
                print(f'[range retry] part={index} attempt={attempt+1} {type(exc).__name__}', flush=True)
                if attempt < 2:
                    time.sleep(attempt+1)
        raise RuntimeError(f'Failed part {index} byte range {first}-{last}: {type(error).__name__}') from error

    def read(self, offset, size):
        if self.cache_offset <= offset and offset+size <= self.cache_offset+len(self.cache):
            return self.cache[offset-self.cache_offset:offset-self.cache_offset+size]
        end = min(self.boundaries[-1], offset+max(size, 16384))
        position, chunks = offset, []
        while position < end:
            index = bisect_right(self.boundaries, position)-1
            stop = min(end, self.boundaries[index+1])
            chunks.append(self.read_part(index, position-self.boundaries[index],
                                         stop-self.boundaries[index]-1))
            position = stop
        self.cache_offset, self.cache = offset, b''.join(chunks)
        return self.cache[:size]


def pax_fields(raw):
    fields, position = {}, 0
    while position < len(raw):
        space = raw.index(b' ', position)
        length = int(raw[position:space])
        key, value = raw[space+1:position+length-1].decode().split('=', 1)
        fields[key] = value
        position += length
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-members', type=int, default=400)
    parser.add_argument('--max-video-bytes', type=int, default=32*1024*1024)
    parser.add_argument('--video-id', help='Optional exact affected video ID, e.g. v_BrgYIg6UXhU')
    args = parser.parse_args()
    metadata = json.loads((PROBE/'friedrichor__ActivityNet_Captions.json').read_text())
    affected = set(json.loads((ROOT/'reports/28_activitynet_sampling_coverage.json').read_text())[
        'suspected_fallback_video_ids'])
    if args.video_id:
        if args.video_id not in affected:
            raise ValueError('Requested video must be an affected test video')
        affected = {args.video_id}
    remote = SplitTar(metadata)
    offset, local_pax, global_pax, long_name = 0, {}, {}, None
    folder = PROBE/'friedrichor__ActivityNet_Captions'
    folder.mkdir(exist_ok=True)
    scanned = []
    started = time.monotonic()
    try:
        while offset+512 <= remote.boundaries[-1] and len(scanned) < args.max_members:
            block = remote.read(offset, 512)
            if block == bytes(512):
                break
            info = tarfile.TarInfo.frombuf(block, 'utf-8', 'strict')
            body_offset = offset+512
            size = info.size
            if info.type in (b'x', b'g'):
                fields = pax_fields(remote.read(body_offset, size))
                (global_pax if info.type == b'g' else local_pax).update(fields)
            elif info.type == b'L':
                long_name = remote.read(body_offset, size).rstrip(b'\0').decode()
            elif info.isfile():
                fields = {**global_pax, **local_pax}
                name = fields.get('path', long_name or info.name)
                size = int(fields.get('size', size))
                member_path = PurePosixPath(name)
                if member_path.is_absolute() or '..' in member_path.parts:
                    raise ValueError('Unsafe archive path')
                scanned.append({'name': name, 'size': size, 'offset': body_offset})
                video_id = member_path.stem
                if len(scanned) % 20 == 0:
                    print(f'[scan] members={len(scanned)} network_mib={remote.downloaded_bytes/2**20:.2f} elapsed={time.monotonic()-started:.1f}s last={name}', flush=True)
                if video_id in affected and member_path.suffix == '.mkv' and size <= args.max_video_bytes:
                    target = folder/member_path.name
                    if target.exists():
                        raise ValueError(f'Fresh target required: {target}')
                    temporary = target.with_suffix('.partial')
                    digest = hashlib.sha256()
                    with temporary.open('wb') as handle:
                        for position in range(0, size, 4*1024*1024):
                            count = min(4*1024*1024, size-position)
                            content = remote.read(body_offset+position, count)
                            if len(content) != count:
                                raise ValueError('Extracted member length mismatch')
                            handle.write(content)
                            digest.update(content)
                    temporary.replace(target)
                    receipt = {
                        'created_at': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                        'publisher': REPO, 'revision': remote.revision,
                        'video_id': video_id, 'archive_member': name,
                        'archive_absolute_body_offset': body_offset, 'bytes': size,
                        'sha256': digest.hexdigest(), 'path': str(target.relative_to(ROOT)),
                        'headers_scanned': len(scanned), 'http_requests': remote.requests,
                        'network_bytes': remote.downloaded_bytes,
                        'parts': remote.parts, 'model_execution': False, 'gpu_access': False,
                        'same_as_author_original_verified': False,
                    }
                    target.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
                    (folder/'scanned_headers.json').write_text(json.dumps(scanned, indent=2)+'\n')
                    print('[downloaded] '+json.dumps({k: receipt[k] for k in (
                        'video_id', 'path', 'bytes', 'sha256', 'headers_scanned', 'network_bytes')}, ensure_ascii=False), flush=True)
                    return
                local_pax, long_name = {}, None
            else:
                local_pax, long_name = {}, None
            offset += 512+((size+511)//512)*512
        (folder/'scanned_headers.json').write_text(json.dumps(scanned, indent=2)+'\n')
        raise RuntimeError(f'No matching MKV found in first {len(scanned)} members')
    finally:
        remote.client.close()


if __name__ == '__main__':
    main()
