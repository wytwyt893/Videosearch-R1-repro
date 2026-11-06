#!/usr/bin/env python3
"""Isolated Charades clock ablation; no changes to released files or weights.

Source time means the released sampled_timestamps_sec (source index/raw FPS),
not newly decoded container PTS. Original frame indices and raw FPS are passed
through directly; the already sampled arrays are never resampled.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'upstream/scripts/internal/stage1/eval_verified_test_rerank_temporal_grounding.py'


def source_payload(row):
    times = row.get('sampled_timestamps_sec')
    indices = row.get('sampled_indices_src')
    fps = float(row.get('raw_fps', 0))
    if not isinstance(times, list) or not isinstance(indices, list) or len(times) != len(indices):
        raise ValueError('Source timestamp/index correspondence is required')
    if not math.isfinite(fps) or fps <= 0 or any(
            not math.isclose(float(t), float(i)/fps, abs_tol=1e-7) for t, i in zip(times, indices)):
        raise ValueError('Released source times disagree with source indices/raw FPS')
    n = row['output_num_frames']
    duration = float(row['source_duration_sec'])
    total = int(row['source_total_frames'])
    if (not times or len(times) != n or any(not math.isfinite(float(t)) or t < 0 for t in times)
            or any(b < a for a,b in zip(times,times[1:]))
            or not math.isfinite(duration) or duration <= 0 or times[-1] > duration+1e-3
            or total <= 0 or any(type(i) is not int or not 0 <= i < total for i in indices)):
        raise ValueError('Invalid source time/frame sequence')
    return {'fps':1.0, 'raw_fps':fps, 'frames_indices':list(indices), 'total_num_frames':total}


def install_clock(clock):
    if clock not in {'nominal', 'source'}:
        raise ValueError('Unknown clock')
    import utils.video_metadata as metadata
    current = metadata._extract_meta_payload
    original = getattr(current, '_charades_original', current)
    if clock == 'nominal':
        metadata._extract_meta_payload = original
        return

    def extract(row):
        # Fail closed: every Charades video in this experiment must have times.
        return source_payload(row)

    extract._charades_original = original
    metadata._extract_meta_payload = extract


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def install_input_audit(module, destination, clock):
    original = module._build_generation_inputs
    signature = inspect.signature(original)

    @functools.wraps(original)
    def observed(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        result = original(*args, **kwargs)
        if result['video_kwargs'].get('do_sample_frames') is not False:
            raise ValueError('Clock ablation requires no further frame sampling')
        metadata = result['video_metadatas'][0]
        times = bound['processor']._calculate_timestamps(
            list(metadata['frames_indices']), metadata['fps'], 2)
        row = {'clock': clock, 'video_id': Path(bound['video_path']).stem,
               'query': bound['query_text'], 'input_metadata': metadata,
               'video_kwargs': result['video_kwargs'],
               'time_labels': [f'{t:.1f}' for t in times],
               'video_shapes': [list(v.shape) for v in result['video_inputs']],
               'video_tensor_sha256': [tensor_sha(v) for v in result['video_inputs']],
               'unexpanded_prompt_sha256': hashlib.sha256(result['text'].encode()).hexdigest()}
        with destination.open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
        return result

    module._build_generation_inputs = observed


def load_official(clock, audit=None):
    spec = importlib.util.spec_from_file_location('charades_timestamp_official_eval', SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    install_clock(clock)
    if audit is not None:
        install_input_audit(module, Path(audit), clock)
    return module


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clock', choices=['nominal', 'source'], required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [str(SOURCE)] + remaining
    load_official(args.clock, args.audit).main()
