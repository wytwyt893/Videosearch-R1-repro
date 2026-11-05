#!/usr/bin/env python3
"""Index the pinned raw-video split tar with parallel, bounded Range reads.

Independent partitions begin at checksum-validated tar headers on 512-byte
boundaries. Member payloads are skipped; only small part prefixes and headers
are read. Progress is resumable and failures remain explicit.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from zoneinfo import ZoneInfo

from fetch_activitynet_raw_sample import SplitTar, PROBE, ROOT, pax_fields

WORK = ROOT/'cache/activitynet_repair_v1/archive_index'


def save(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def first_header(remote, part_index, start=None, stop=None):
    start = remote.boundaries[part_index] if start is None else start
    stop = remote.boundaries[part_index+1] if stop is None else stop
    if start == 0:
        return 0
    for window in (4*2**20, 8*2**20, 16*2**20, 32*2**20, 64*2**20):
        raw = remote.read(start, min(window, stop-start))
        for pos in range(0, len(raw)-511, 512):
            block = raw[pos:pos+512]
            if not block[257:263].startswith(b'ustar'):
                continue
            try:
                info = tarfile.TarInfo.frombuf(block, 'utf-8', 'strict')
            except (tarfile.TarError, ValueError, UnicodeError):
                continue
            if info.isfile() and info.name.startswith('Activity_Videos/') and info.size > 0:
                return start+pos
    raise RuntimeError('No validated tar header within bounded 64 MiB partition prefix')


def index_part(index, metadata, affected, partition_start=None, partition_stop=None, seed_path=None):
    path = WORK/f'part_{index:03d}.json'
    remote = SplitTar(metadata)
    rows, local_pax, global_pax, long_name = [], {}, {}, None
    try:
        beginning = remote.boundaries[index] if partition_start is None else partition_start
        ending = remote.boundaries[index+1] if partition_stop is None else partition_stop
        if path.exists():
            state = json.loads(path.read_text())
            if state['revision'] != metadata['sha']:
                raise ValueError('Archive revision changed')
            if state.get('complete'):
                return state
            rows = state['members']
            offset = state['next_offset']
            local_pax, global_pax, long_name = state.get('local_pax', {}), state.get('global_pax', {}), state.get('long_name')
        elif seed_path is not None and seed_path.exists():
            seed = json.loads(seed_path.read_text())
            if seed['revision'] != metadata['sha']:
                raise ValueError('Seed revision mismatch')
            if seed['next_offset'] > beginning:
                rows = [r for r in seed['members'] if beginning <= r['offset']-512 < ending]
                offset = seed['next_offset']
                local_pax, global_pax, long_name = seed.get('local_pax', {}), seed.get('global_pax', {}), seed.get('long_name')
            else:
                offset = first_header(remote, index, beginning, ending)
        else:
            offset = first_header(remote, index, beginning, ending)
        start_offset = offset

        def state(complete=False, error=None):
            return {'revision': metadata['sha'], 'part_index': index,
                    'partition_start_header': start_offset, 'partition_start': beginning,
                    'partition_stop': ending, 'next_offset': offset,
                    'complete': complete, 'error': error, 'members': rows,
                    'local_pax': local_pax, 'global_pax': global_pax, 'long_name': long_name,
                    'affected_members': [r for r in rows if r['video_id'] in affected],
                    'network_bytes_this_attempt': remote.downloaded_bytes,
                    'http_requests_this_attempt': remote.requests}

        while offset < ending and offset+512 <= remote.boundaries[-1]:
            block = remote.read(offset, 512)
            if block == bytes(512):
                break
            info = tarfile.TarInfo.frombuf(block, 'utf-8', 'strict')
            body, size = offset+512, info.size
            if info.type in (b'x', b'g'):
                fields = pax_fields(remote.read(body, size))
                if info.type == b'g' and ('path' in fields or 'size' in fields):
                    raise ValueError('Global PAX path/size requires sequential global-state verification')
                (global_pax if info.type == b'g' else local_pax).update(fields)
            elif info.type == b'L':
                long_name = remote.read(body, size).rstrip(b'\0').decode()
            elif info.isfile():
                fields = {**global_pax, **local_pax}
                name = fields.get('path', long_name or info.name)
                size = int(fields.get('size', size))
                member = PurePosixPath(name)
                if member.is_absolute() or '..' in member.parts:
                    raise ValueError('Unsafe member path')
                rows.append({'name': name, 'video_id': member.stem, 'size': size, 'offset': body})
                local_pax, long_name = {}, None
            else:
                local_pax, long_name = {}, None
            offset += 512+((size+511)//512)*512
            if rows and len(rows) % 100 == 0:
                save(path, state())
                print(f'[part {index}] indexed={len(rows)} affected={sum(r["video_id"] in affected for r in rows)} network_mib={remote.downloaded_bytes/2**20:.2f}', flush=True)
        result = state(complete=True)
        save(path, result)
        return result
    except Exception as exc:
        result = {'revision': metadata['sha'], 'part_index': index, 'complete': False,
                  'error': f'{type(exc).__name__}: {str(exc)[:200]}',
                  'members': rows, 'affected_members': [r for r in rows if r['video_id'] in affected]}
        # Preserve a resumable checkpoint if there was one, and keep the final
        # failure separate rather than discarding the last verified position.
        save(WORK/f'part_{index:03d}.failure.json', result)
        return result
    finally:
        remote.client.close()


def main():
    global WORK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--segments-per-part', type=int, default=1)
    args = parser.parse_args()
    original_work = WORK
    if args.segments_per_part > 1:
        WORK = ROOT/'cache/activitynet_repair_v1/archive_index_parallel'
    WORK.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((PROBE/'friedrichor__ActivityNet_Captions.json').read_text())
    affected = set(json.loads((ROOT/'reports/28_activitynet_sampling_coverage.json').read_text())['suspected_fallback_video_ids'])
    parts = sorted((r for r in metadata['siblings'] if r['rfilename'].startswith('ActivityNet_Videos.tar.part-')),
                   key=lambda r: r['rfilename'])
    partitions, start = [], 0
    for physical_index, part in enumerate(parts):
        blocks = part['size']//512
        for segment in range(args.segments_per_part):
            first = start+(blocks*segment//args.segments_per_part)*512
            last = start+(blocks*(segment+1)//args.segments_per_part)*512
            partitions.append((first, last, original_work/f'part_{physical_index:03d}.json'))
        start += part['size']
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(index_part, i, metadata, affected, first, last,
                            seed if args.segments_per_part > 1 else None): i
                for i, (first, last, seed) in enumerate(partitions)}
        for future in concurrent.futures.as_completed(jobs):
            result = future.result()
            results[result['part_index']] = result
            candidates = {}
            for partition in results.values():
                for row in partition['affected_members']:
                    candidates.setdefault(row['video_id'], []).append(row)
            save(WORK/'candidates.json', candidates)
            status = {
                'updated_at': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                'status': 'INDEXING' if len(results) < len(partitions) else 'INDEX_ATTEMPTS_COMPLETE',
                'source_revision': metadata['sha'], 'partitions_finished': len(results),
                'partitions_total': len(partitions),
                'segments_per_physical_part': args.segments_per_part,
                'partitions_complete': sum(bool(r['complete']) for r in results.values()),
                'errors': {str(i): r['error'] for i, r in results.items() if not r['complete']},
                'affected_video_ids_found': len(candidates), 'model_execution': False, 'gpu_access': False,
            }
            save(WORK/'status.json', status)
            print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
