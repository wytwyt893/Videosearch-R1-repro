#!/usr/bin/env python3
"""CPU-only, two-pass extraction into a fresh directory for raw-video checks.

Count *decoded* frames before sampling; never turn an unknown header frame count
into a 64-frame source. Keep the official frame-index sampling and resize rules,
but also record actual presentation timestamps. This tool is not connected to
the inference runner and does not replace any released tensors or timestamps.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import av
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT/'upstream/preprocess/extract_npy/extract_npy.py'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def scan_video(path):
    """Collect timestamps/counts, not all decoded image arrays, with strict errors."""
    timestamps, frame_durations = [], []
    with av.open(str(path), 'r', options={'err_detect': 'explode'}) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        declared_count = int(stream.frames)
        fps = float(stream.average_rate) if stream.average_rate else None
        stream_duration = (float(stream.duration*stream.time_base)
                           if stream.duration is not None and stream.time_base is not None else None)
        container_duration = container.duration/av.time_base if container.duration is not None else None
        for frame in container.decode(stream):
            if frame.is_corrupt:
                raise ValueError('Corrupt decoded frame')
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Missing presentation timestamp; refusing to guess frame times')
            ts = float(frame.pts*frame.time_base)
            if not math.isfinite(ts) or (timestamps and ts < timestamps[-1]):
                raise ValueError('Nonfinite or backwards presentation timestamps')
            timestamps.append(ts)
            duration = getattr(frame, 'duration', None)
            frame_durations.append(float(duration*frame.time_base) if duration else None)
    if not timestamps:
        raise ValueError('No video frames decoded')
    positive_gaps = [b-a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    last_duration = frame_durations[-1]
    if last_duration is None or last_duration <= 0:
        if positive_gaps:
            last_duration = float(np.median(positive_gaps))
        elif fps is not None and math.isfinite(fps) and fps > 0:
            last_duration = 1/fps
        else:
            raise ValueError('Cannot determine the last frame duration')
    pts_extent = timestamps[-1]-timestamps[0]+last_duration
    # Preserve the normal official duration when it is available and plausible.
    # Unknown Matroska stream durations are recovered from video PTS, not a
    # container's potentially longer audio track or count*guessed FPS.
    valid_stream_duration = (stream_duration is not None and math.isfinite(stream_duration)
                             and stream_duration > 0
                             and stream_duration+max(last_duration, 1e-3) >= pts_extent)
    duration = stream_duration if valid_stream_duration else pts_extent
    return {
        'header_source_total_frames': declared_count,
        'decoded_source_total_frames': len(timestamps),
        'header_count_unknown': declared_count <= 0,
        'positive_header_count_matches_decode': declared_count == len(timestamps) if declared_count > 0 else None,
        'header_average_fps': fps,
        'header_stream_duration_seconds': stream_duration,
        'header_container_duration_seconds': container_duration,
        'source_duration_sec': duration,
        'source_duration_method': 'stream_duration' if valid_stream_duration else 'decoded_pts_extent_with_last_frame_duration',
        'first_pts_seconds': timestamps[0], 'last_pts_seconds': timestamps[-1],
        'last_frame_duration_seconds': last_duration,
        'positive_pts_gap_seconds': {'min': min(positive_gaps), 'max': max(positive_gaps)} if positive_gaps else {},
        '_timestamps': timestamps,
    }


def resize_like_official(arr, max_pixels=200704, min_pixels=256):
    height, width = arr.shape[:2]
    new_width, new_height = width, height
    if max_pixels is not None and height*width > max_pixels:
        factor = math.sqrt(max_pixels/(height*width))
        new_width, new_height = max(1, int(width*factor)), max(1, int(height*factor))
        width, height = new_width, new_height
    if min_pixels is not None and height*width < min_pixels:
        factor = math.sqrt(min_pixels/(height*width))
        new_width, new_height = max(1, int(width*factor)), max(1, int(height*factor))
    if (new_height, new_width) == arr.shape[:2]:
        return arr
    return np.asarray(Image.fromarray(arr).resize((new_width, new_height), Image.BICUBIC), dtype=np.uint8)


def extract_verified(path, fps=1.0, max_frames=64, expected_total_frames=None,
                     probe_timestamps_sec=None, probe_sink=None, probe_selection='nearest'):
    if not math.isfinite(fps) or fps <= 0 or max_frames < 2 or max_frames % 2:
        raise ValueError('FPS must be positive and max_frames must be an even integer >= 2')
    path = Path(path)
    source_sha = sha256(path)
    scan = scan_video(path)
    count = scan['decoded_source_total_frames']
    if expected_total_frames is not None and count != expected_total_frames:
        raise ValueError(f'Decoded {count} frames, expected {expected_total_frames}; '
                         'possible version mismatch or inaccurate reference frame-count estimate')
    selected_count = min(count, max_frames, max(1, math.floor(scan['source_duration_sec']*fps)))
    indices = np.linspace(0, count-1, selected_count).astype(np.int32).tolist()
    timestamps = scan.pop('_timestamps')
    relative_all_pts = np.asarray(timestamps, dtype=np.float64)-timestamps[0]
    probe_indices = []
    probe_groups = []
    if probe_timestamps_sec is not None:
        if probe_sink is None:
            raise ValueError('A probe_sink dictionary is required with probe timestamps')
        for timestamp in probe_timestamps_sec:
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError('Invalid reference probe timestamp')
            position = int(np.searchsorted(relative_all_pts, timestamp))
            choices = sorted({max(0, min(count-1, position)), max(0, min(count-1, position-1))})
            if probe_selection == 'nearest':
                chosen = [min(choices, key=lambda j: (abs(relative_all_pts[j]-timestamp), j))]
            elif probe_selection == 'bracket':
                exact = [j for j in choices if abs(relative_all_pts[j]-timestamp) <= 1e-6]
                chosen = exact[:1] or choices
            else:
                raise ValueError('Unknown reference probe selection')
            probe_groups.append(list(range(len(probe_indices), len(probe_indices)+len(chosen))))
            probe_indices.extend(chosen)
    wanted = set(indices) | set(probe_indices)
    selected = {}
    decoded_count = 0
    with av.open(str(path), 'r', options={'err_detect': 'explode'}) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        for i, frame in enumerate(container.decode(stream)):
            if frame.is_corrupt or i >= len(timestamps) or frame.pts is None or frame.time_base is None:
                raise ValueError('Decode changed between the counting and extraction passes')
            if not math.isclose(float(frame.pts*frame.time_base), timestamps[i], abs_tol=1e-9):
                raise ValueError('Timestamps changed between extraction passes')
            if i in wanted:
                selected[i] = resize_like_official(frame.to_ndarray(format='rgb24'))
            decoded_count += 1
    if decoded_count != count or set(selected) != wanted:
        raise ValueError('Frame count or selected frames changed between extraction passes')
    frames = [selected[i] for i in indices]
    sampled_indices = list(indices)
    even_pad = bool(len(frames) % 2)
    if even_pad:
        frames.append(frames[-1].copy())
        sampled_indices.append(sampled_indices[-1])
    array = np.stack(frames).astype(np.uint8)
    if sha256(path) != source_sha:
        raise ValueError('Raw video changed during extraction')
    if probe_sink is not None:
        probe_sink.update(images=[selected[i] for i in probe_indices], source_indices=probe_indices,
                          timestamps_sec=[float(relative_all_pts[i]) for i in probe_indices],
                          groups=probe_groups)
    relative_pts = [timestamps[i]-timestamps[0] for i in sampled_indices]
    meta = {
        'created_at': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'tool': 'local_verified_two_pass_extractor', 'gpu_access': False, 'model_execution': False,
        'source_video_path': str(path.resolve()), 'source_sha256': source_sha,
        **scan, 'source_total_frames': count, 'raw_fps': scan['header_average_fps'],
        'target_sample_fps': fps, 'target_max_frames': max_frames,
        'sampling_domain': 'source_frame_index_uniform_like_official_normal_branch',
        'requested_indices_src': indices, 'sampled_indices_src': sampled_indices,
        'sampled_timestamps_sec': relative_pts,
        'sampled_absolute_pts_seconds': [timestamps[i] for i in sampled_indices],
        'sampled_frames_before_even_pad': selected_count, 'even_pad_applied': even_pad,
        'output_num_frames': len(frames), 'output_height': int(array.shape[1]),
        'output_width': int(array.shape[2]), 'output_dtype': str(array.dtype),
        'source_frame_span_fraction': (indices[-1]-indices[0])/(count-1) if count > 1 else 1.0,
        'timestamp_policy': 'Actual decoded presentation timestamps, relative to first video PTS. Not wired into inference.',
        'limitation': 'Decoding to EOF verifies this file, not its completeness against an unavailable original or annotation. Uniform source frame indices need not mean uniform PTS for VFR.',
        'time_axis_integration_done': False,
    }
    return array, meta


def compare_official(path, verified, fps=1.0, max_frames=64):
    """Evaluate only the three official pure extraction functions, CPU-only."""
    tree = ast.parse(OFFICIAL.read_text())
    names = {'_get_sample_indices', '_resize_frame', '_extract_frames'}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'np': np, 'av': av, 'Image': Image, 'math': math,
                 'os': __import__('os'), 'datetime': datetime.datetime, 'timezone': datetime.timezone,
                 'video_fps': fps, 'video_maxlen': max_frames,
                 'video_max_pixels': 200704, 'video_min_pixels': 256}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(OFFICIAL), 'exec'), namespace)
    comparison = {'official_source_sha256': sha256(OFFICIAL), 'original_functions_only': True,
                  'raw_source_identity_independently_verified': False}
    try:
        original_array, original_meta = namespace['_extract_frames'](str(path))
    except Exception as exc:
        comparison.update(official_extraction_error=f'{type(exc).__name__}: {exc}')
        return comparison
    comparison.update(
        official_metadata=original_meta,
        official_npy_shape=list(original_array.shape),
        identical_array_shape=original_array.shape == verified.shape,
        identical_arrays=np.array_equal(original_array, verified),
    )
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='A new, separate output directory')
    parser.add_argument('--fps', type=float, default=1.0)
    parser.add_argument('--max-frames', type=int, default=64)
    parser.add_argument('--expected-total-frames', type=int)
    parser.add_argument('--compare-official', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output directory exists; refusing to overwrite assets')
    output = args.output.resolve()
    canonical = (ROOT/'data').resolve()
    if output == canonical or canonical in output.parents:
        raise ValueError('Write derived checks outside the canonical data directory')
    array, metadata = extract_verified(args.video, args.fps, args.max_frames, args.expected_total_frames)
    if args.compare_official:
        metadata['official_comparison'] = compare_official(args.video, array, args.fps, args.max_frames)
    output.mkdir(parents=True)
    np.save(output/'frames.npy', array, allow_pickle=False)
    metadata['output_npy_path'] = str(output/'frames.npy')
    metadata['output_npy_sha256'] = sha256(output/'frames.npy')
    (output/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: metadata[k] for k in (
        'header_source_total_frames', 'decoded_source_total_frames', 'source_duration_sec',
        'sampled_frames_before_even_pad', 'output_num_frames', 'source_frame_span_fraction',
        'sampled_timestamps_sec')}, indent=2))


if __name__ == '__main__':
    main()
