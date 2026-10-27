#!/usr/bin/env python3
"""Use official evaluation unchanged; fetch missing video tensors on file lookup."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def resolve_with_fetch(video_root, video_id):
    resolved = original(video_root, video_id)
    if resolved is not None:
        return resolved
    assert Path(video_root).resolve() == ROOT/'data/charades-sta/test/video_npy_with_meta'
    selected = [f'test/video_npy_with_meta/{video_id}{ext}' for ext in ('.npy', '.npz')
                if f'test/video_npy_with_meta/{video_id}{ext}' in available]
    assert len(selected) == 1, f'No unique official tensor for {video_id}'
    request = ROOT/f'cache/tmp/smoke_tensor_{os.getpid()}.json'
    request.write_text(json.dumps(selected)+'\n')
    print(f'[smoke asset] fetching official tensor for {video_id}', flush=True)
    subprocess.run([str(ROOT/'envs/hf_tools/bin/python'), '-B', str(ROOT/'scripts/remote_tar.py'),
                    'extract', '--paths-json', str(request)], check=True)
    resolved = original(video_root, video_id)
    assert resolved is not None
    return resolved


def main():
    global original, available
    source = Path(sys.argv.pop(1)).resolve()
    assert source == ROOT/'upstream/scripts/internal/stage1/eval_verified_test_rerank_temporal_grounding.py'
    spec = importlib.util.spec_from_file_location('official_smoke_eval', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original = module._resolve_video_path
    available = {m['name'] for p in (ROOT/'cache/manifests/tar_index').glob('*.json')
                 for m in json.loads(p.read_text())['members']}
    module._resolve_video_path = resolve_with_fetch
    module.main()


if __name__ == '__main__':
    main()
