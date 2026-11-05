#!/usr/bin/env python3
"""Bounded second pass for transient source downloads, preserving first-pass evidence."""
import concurrent.futures
import json
from pathlib import Path
import time

from download_activitynet_repair_sources import WORK, INDEX, LOCAL, download_one, now, save


def retry(row):
    result = None
    for outer in range(2):
        if hasattr(LOCAL, 'client'):
            LOCAL.client.close()
            del LOCAL.client
        source = dict(row)
        source['video_url'] += f'?repair_retry={outer+1}'
        result = download_one(source)
        if result['status'] == 'DOWNLOADED_PENDING_QUALIFICATION':
            return result
    return result


def main():
    path = WORK/'download_results.json'
    previous = json.loads(path.read_text())
    status = json.loads((WORK/'download_status.json').read_text())
    if status['status'] != 'DOWNLOAD_ATTEMPTS_COMPLETE':
        raise ValueError('First pass must finish before the bounded retry pass')
    archive = WORK/'download_first_pass_results.json'
    if not archive.exists():
        save(archive, previous)
        save(WORK/'download_first_pass_status.json', status)
    rows = [r for r in json.loads(INDEX.read_text())
            if previous.get('v_'+r['video_id'], {}).get('status') != 'DOWNLOADED_PENDING_QUALIFICATION']
    status.update(status='DOWNLOADING_RETRY', updated_at=now(), retry_targets=len(rows))
    save(WORK/'download_status.json', status)
    retry_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(retry, rows):
            previous[result['video_id']] = result
            retry_results[result['video_id']] = result
            save(path, previous)
            save(WORK/'download_retry_results.json', retry_results)
            print(result['video_id'], result['status'], flush=True)
    status.update(status='DOWNLOAD_ATTEMPTS_COMPLETE', updated_at=now(),
                  downloaded_videos=sum(r['status'] == 'DOWNLOADED_PENDING_QUALIFICATION' for r in previous.values()),
                  failed_videos=sum(r['status'] == 'DOWNLOAD_FAILED' for r in previous.values()),
                  completed_bytes=sum(r.get('bytes', 0) for r in previous.values()),
                  retry_attempts_complete=True)
    save(WORK/'download_status.json', status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
