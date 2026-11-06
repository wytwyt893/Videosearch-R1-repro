#!/usr/bin/env python3
"""Read-only CPU validation of the numbered experiment layout and migration."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'experiments/00_setup/reports/51_workspace_reorganization.json'


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    manifest = json.loads(MANIFEST.read_text())
    cleanup = {c['path']: c for c in manifest['cleanup']}
    for moved in manifest['moves']:
        old, new = ROOT / moved['old_path'], ROOT / moved['new_path']
        assert old.is_file() and new.is_file(), moved
        assert old.resolve() == new.resolve(), moved
        expected = cleanup.get(moved['new_path'], {}).get('after', {}).get('sha256', moved['sha256'])
        assert sha(new) == expected, moved['new_path']
        if moved['new_path'] not in cleanup:
            assert new.stat().st_mtime_ns == moved['mtime_ns'], moved['new_path']
    for link in manifest['compatibility_links']:
        old, new = ROOT / link['old_path'], ROOT / link['new_path']
        assert old.is_symlink() and old.resolve() == new.resolve(), link
    for link in manifest.get('additional_compatibility_links', []):
        path, target = ROOT / link['path'], ROOT / link['target']
        assert path.is_symlink() and path.resolve() == target.resolve(), link
    for row in manifest['frozen_files']:
        path = ROOT / row['path']
        assert sha(path) == row['sha256'], row['path']
        assert path.stat().st_mtime_ns == row['mtime_ns'], row['path']
    script_entries = 0
    for group in manifest['groups']:
        folder = ROOT / 'experiments' / group['id']
        assert (folder / 'README.md').is_file() and (folder / 'RESULTS.md').is_file()
        for entry in group['scripts']:
            path, implementation = ROOT / entry['entry'], ROOT / entry['implementation']
            assert path.is_symlink() and path.resolve() == implementation.resolve(), entry
            assert sha(path) == entry['sha256'], entry
            script_entries += 1
    for c in manifest['cleanup']:
        if c['operation'] == 'exact_duplicate_replaced_by_symlink':
            path, target = ROOT / c['path'], ROOT / c['target']
            assert path.is_symlink() and path.resolve() == target.resolve(), c
            assert sha(path) == c['before']['sha256'], c
        elif c['operation'] == 'lossless_npz_compression':
            import numpy as np
            with np.load(ROOT / c['path'], allow_pickle=False) as arrays:
                assert set(arrays.files) == set(c['arrays'])
                for key, expected in c['arrays'].items():
                    array = arrays[key]
                    assert list(array.shape) == expected['shape']
                    assert str(array.dtype) == expected['dtype']
                    assert hashlib.sha256(array.tobytes()).hexdigest() == expected['sha256']
    commit = subprocess.check_output(['git', '-C', str(ROOT / 'upstream'), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == manifest['official']['commit']
    assert not subprocess.check_output(['git', '-C', str(ROOT / 'upstream'), 'status', '--porcelain'], text=True).strip()
    # Existing qualification code and reports are reused byte for byte.
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'source scripts/env.sh first'
    sys.path.insert(0, str(ROOT / 'scripts'))
    from run_charades_timestamp_full import verify
    from charades_localization_probe_worker import verify_plan
    full = verify(ROOT / 'reports/43_charades_timestamp_full_cpu.json')
    probe = verify_plan()
    import torch
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(
        status='LAYOUT_CPU_VALIDATION_PASSED', experiment_groups=len(manifest['groups']),
        script_entries=script_entries, moved_files=len(manifest['moves']),
        frozen_files=len(manifest['frozen_files']),
        full_timestamp_qualification=full['status'], localization_probe_qualification=probe['status'],
        cuda_initialized=False, model_execution=False,
        released_payload_MiB=round(manifest['cleanup_released_payload_bytes'] / 1024**2, 2)
    ), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
