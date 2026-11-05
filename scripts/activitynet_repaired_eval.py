#!/usr/bin/env python3
"""Load the pinned official evaluator with an explicit derived-data time adapter.

This wrapper performs no inference unless a caller executes the official main.
It is separate from all released-data evaluation entry points.
"""
import importlib.util
import functools
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'upstream/scripts/internal/stage1/eval_verified_test_rerank_temporal_grounding.py'


def load_official():
    spec = importlib.util.spec_from_file_location('activitynet_repaired_official_eval', SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from activitynet_repair_time import install
    install()
    audit_path = os.environ.get('VIDEOSEARCH_REPAIR_INPUT_AUDIT')
    if audit_path:
        install_input_audit(module, Path(audit_path))
    return module


def install_input_audit(module, destination):
    """Observe the official generation payload without changing its inputs."""
    original = module._build_generation_inputs
    signature = inspect.signature(original)

    @functools.wraps(original)
    def observed(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        result = original(*args, **kwargs)
        meta = result['video_metadatas'][0]
        times = bound['processor']._calculate_timestamps(meta['frames_indices'], meta['fps'], 2)
        path = Path(bound['video_path'])
        row = {'video_id': path.stem, 'video_path': str(path), 'resolved_path': str(path.resolve()),
               'input_metadata': meta, 'video_kwargs': result['video_kwargs'],
               'model_timestamp_labels_seconds': times,
               'do_sample_frames': result['video_kwargs'].get('do_sample_frames')}
        with destination.open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
        return result

    module._build_generation_inputs = observed


if __name__ == '__main__':
    source = Path(sys.argv.pop(1)).resolve()
    if source != SOURCE:
        raise ValueError('Only the fixed official evaluation module is supported')
    load_official().main()
