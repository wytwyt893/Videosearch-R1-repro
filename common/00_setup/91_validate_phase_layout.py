#!/usr/bin/env python3
"""Read-only CPU validation of stage/benchmark organization and navigation."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import runpy
import subprocess

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'common/env.sh').is_file() and (p / 'experiments').is_dir())


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    ledger = json.loads((ROOT / 'common/00_setup/reports/53_workspace_phase_reorganization.json').read_text())
    for operation in ledger['operations']:
        old, new = ROOT / operation['old_path'], ROOT / operation['new_path']
        assert old.is_symlink() and old.resolve() == new.resolve(), operation
        if operation['kind'] == 'shared_implementation':
            assert sha(new) == operation['sha256'], operation
            assert new.stat().st_mtime_ns == operation['mtime_ns'], operation
    for row in ledger['frozen_files']:
        path = ROOT / row['path']
        assert sha(path) == row['sha256'], row['path']
        assert path.stat().st_mtime_ns == row['mtime_ns'], row['path']
    index = json.loads((ROOT / 'experiments/index.json').read_text())
    assert index['schema_version'] == 2
    for phase in index['phases']:
        assert (ROOT / phase['directory'] / 'README.md').is_file()
    for group in index['experiments']:
        folder = ROOT / group['directory']
        assert folder.is_dir() and not folder.is_symlink()
        assert (folder / 'README.md').is_file() and (folder / 'RESULTS.md').is_file()
        for entry in group['scripts']:
            path = ROOT / entry['entry']
            implementation = ROOT / entry.get('implementation', entry['entry'])
            assert path.is_file() and path.resolve() == implementation.resolve()
            if entry.get('sha256'):
                assert sha(path) == entry['sha256']
    grouped_tests = [ROOT / p for group in index['test_groups'] for p in group['files']]
    assert len(grouped_tests) == len({p.resolve() for p in grouped_tests})
    assert all(p.is_file() for p in grouped_tests)
    # Check current navigation, not verbatim historical Markdown snapshots.
    markdown = {ROOT / 'README.md', ROOT / 'AGENTS.md', ROOT / 'tests/README.md', ROOT / 'scripts/README.md'}
    markdown.update((ROOT / 'docs').glob('*.md'))
    markdown.update((ROOT / 'common').glob('*.md'))
    markdown.update((ROOT / 'common/00_setup').glob('*.md'))
    markdown.update((ROOT / 'experiments').glob('*.md'))
    for phase in index['phases']:
        markdown.update(p for p in (ROOT / phase['directory']).rglob('*.md')
                        if not {'reports', 'outputs', 'logs'} & set(p.relative_to(ROOT).parts))
    local_links = 0
    for path in sorted(markdown):
        for a, b in re.findall(r'(?<!!)\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))\)', path.read_text()):
            href = (a or b).split('#')[0]
            if not href or re.match(r'\w+://', href):
                continue
            assert (path.parent / href).exists(), (str(path.relative_to(ROOT)), href)
            local_links += 1
    raw_files = []
    for group in index['experiments']:
        for directory in ['outputs', 'logs']:
            raw_files.extend(str(p.relative_to(ROOT)) for p in (ROOT / group['directory'] / directory).rglob('*')
                             if p.is_file())
    raw_files = sorted(set(raw_files))
    payload = '\0'.join(raw_files) + '\0'
    ignored = subprocess.check_output(['git', 'check-ignore', '--stdin', '-z'], cwd=ROOT,
                                      input=payload.encode()).decode().strip('\0').split('\0')
    assert set(raw_files) == set(ignored), 'Some canonical raw outputs/logs are not ignored'
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        runpy.run_path(str(ROOT / 'common/00_setup/90_validate_layout.py'))['main']()
    historical = json.loads(captured.getvalue())
    result = dict(status='PHASE_LAYOUT_CPU_VALIDATION_PASSED', phases=len(index['phases']),
                  registered_subexperiments=len(index['experiments']), shared_implementations=len(index['common_implementations']),
                  classified_test_modules=len(grouped_tests), historical_validation=historical,
                  current_markdown_files=len(markdown), current_local_links=local_links,
                  canonical_raw_files_ignored=len(raw_files), broken_navigation_links=0,
                  model_execution=False, gpu_access=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
