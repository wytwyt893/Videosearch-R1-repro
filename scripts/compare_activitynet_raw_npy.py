#!/usr/bin/env python3
"""Compare one released ActivityNet NPY with a retrieved source video on CPU.

Keep byte/pixel identity separate from content similarity when a public mirror
uses another encode. All derived arrays are written to a fresh output directory.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def find_record(path, video_id):
    with path.open() as handle:
        return next(r for r in map(json.loads, handle) if r.get('video_id') == video_id)


def read_source_frames(path, source_indices):
    import av
    from extract_video_frames_verified import resize_like_official
    wanted = set(source_indices)
    images, pts = {}, {}
    thumbnails, all_pts = [], []
    with av.open(str(path), 'r', options={'err_detect': 'explode'}) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if frame.is_corrupt or frame.pts is None or frame.time_base is None:
                raise ValueError('Invalid source frame')
            timestamp = float(frame.pts*frame.time_base)
            rgb = frame.to_ndarray(format='rgb24')
            all_pts.append(timestamp)
            thumbnails.append(np.asarray(Image.fromarray(rgb).resize((32, 24), Image.Resampling.BICUBIC)))
            if i in wanted:
                images[i] = resize_like_official(rgb)
                pts[i] = timestamp
    if set(images) != wanted:
        raise ValueError('Raw source does not contain all referenced frame indices')
    return (np.stack([images[i] for i in source_indices]),
            [pts[i]-all_pts[0] for i in source_indices],
            np.stack(thumbnails), [p-all_pts[0] for p in all_pts])


def align_spatial(array, target_shape):
    if list(array.shape[1:]) == list(target_shape):
        return array
    height, width, _ = target_shape
    return np.stack([np.asarray(Image.fromarray(a).resize((width, height), Image.Resampling.BICUBIC))
                     for a in array])


def compare(array, target):
    aligned = align_spatial(array, target.shape[1:])
    delta = aligned.astype(np.float32)-target.astype(np.float32)
    per_frame = np.abs(delta).mean(axis=(1, 2, 3))
    return {
        'mean_absolute_rgb_error_0_to_255': float(per_frame.mean()),
        'max_per_frame_mean_absolute_rgb_error': float(per_frame.max()),
        'rmse_rgb_0_to_255': float(np.sqrt(np.square(delta).mean())),
        'native_shapes_match': array.shape == target.shape,
        'exact_native_pixels': bool(np.array_equal(array, target)),
        'per_frame_mae': per_frame.tolist(),
        'spatial_resize_for_comparison': array.shape != target.shape,
    }


def analyze(video_path, video_id, output):
    from extract_video_frames_verified import extract_verified, compare_official, sha256
    if output.exists():
        raise ValueError('Choose a fresh output directory')
    if (ROOT/'data').resolve() in output.resolve().parents:
        raise ValueError('Derived output must remain outside canonical data')
    npy_meta_path = ROOT/'data/activitynet/test/video_npy_with_meta/meta.jsonl'
    embed_meta_path = ROOT/'data/activitynet/test/video_embedding_1fps/meta.jsonl'
    meta = find_record(npy_meta_path, video_id)
    embedding = find_record(embed_meta_path, video_id)
    reference = embedding['info']
    official_path = ROOT/'data/activitynet'/meta['output_npy_path']
    official_hash = sha256(official_path)
    official = np.load(official_path, allow_pickle=False)
    derived, derived_meta = extract_verified(video_path)
    prefix, prefix_pts, thumbnails, source_pts = read_source_frames(video_path, meta['sampled_indices_src'])
    prefix_comparison = compare(prefix, official)
    # This is a negative comparison at the same number of ordinal positions,
    # not a ground-truth VCMR metric or a test of model accuracy.
    positions = np.linspace(0, len(derived)-1, len(official)).round().astype(int)
    full_range_comparison = compare(derived[positions], official)
    source_float = thumbnails.astype(np.float32)
    searches = []
    for i in np.linspace(0, len(official)-1, 8).round().astype(int):
        target = np.asarray(Image.fromarray(official[i]).resize((32, 24), Image.Resampling.BICUBIC), dtype=np.float32)
        distances = np.abs(source_float-target).mean(axis=(1, 2, 3))
        best = int(distances.argmin())
        searches.append({'released_npy_index': int(i),
                         'recorded_source_timestamp_seconds': meta['sampled_timestamps_sec'][i],
                         'best_source_frame_index': best,
                         'best_source_timestamp_seconds': source_pts[best],
                         'best_thumbnail_rgb_mae': float(distances[best]),
                         'note': 'Appearance match at 32x24; repeated/static scenes can have nonunique matches.'})
    receipt_path = video_path.with_suffix('.receipt.json')
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if receipt and receipt['sha256'] != derived_meta['source_sha256']:
        raise ValueError('Downloaded file does not match its recorded receipt')
    exact_prefix = prefix_comparison['exact_native_pixels']
    report = {
        'created_at': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'status': 'EXACT_RAW_PREFIX_PIXEL_MATCH' if exact_prefix else 'RAW_CONTENT_COMPARISON_REQUIRES_VERSION_REVIEW',
        'video_id': video_id, 'model_execution': False, 'gpu_access': False,
        'raw_video': str(video_path.resolve()), 'download_receipt': receipt,
        'official_npy': str(official_path), 'official_npy_sha256': official_hash,
        'official_shape': list(official.shape), 'official_source_metadata': meta,
        'retrieval_source_metadata': embedding,
        'verified_extraction_metadata': derived_meta,
        'source_frame_count_matches_retrieval_record': derived_meta['source_total_frames'] == reference['num_total'],
        'source_fps_matches_retrieval_record': (derived_meta['raw_fps'] is not None and
            math.isclose(derived_meta['raw_fps'], reference['fps'], rel_tol=1e-5)),
        'exact_author_raw_file_identity_verified': False,
        'raw_referenced_frame_indices': meta['sampled_indices_src'], 'raw_referenced_frame_pts_seconds': prefix_pts,
        'released_npy_vs_raw_referenced_indices': prefix_comparison,
        'released_npy_vs_full_range_ordinal_positions': full_range_comparison,
        'whole_video_appearance_search': searches,
        'official_extractor_on_this_raw_file': compare_official(video_path, derived),
        'interpretation_limits': [
            'Similarity is evidence of visual correspondence, not byte identity with an unavailable author source.',
            'A mismatch against OpenCV num_total can come from frame-count estimation, stream duration or a different encode; the mismatch alone does not identify the cause.',
            'Full-range extraction operates on this acquired source only; no claim that all other sources are available or valid.',
            'Actual timestamps are recorded but not connected to official inference. No model score or paper reproduction is established here.',
        ],
        'canonical_assets_modified': False, 'time_axis_integration_done': False,
    }
    if sha256(official_path) != official_hash:
        raise ValueError('Released NPY changed during analysis')
    output.mkdir(parents=True)
    np.save(output/'raw_referenced_frames.npy', prefix, allow_pickle=False)
    np.save(output/'verified_full_range.npy', derived, allow_pickle=False)
    report['output_files'] = {name: {'path': str(output/name), 'sha256': sha256(output/name)}
                              for name in ('raw_referenced_frames.npy', 'verified_full_range.npy')}
    (output/'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    with (output/'per_frame.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['npy_index', 'recorded_source_index', 'recorded_source_time_seconds',
                         'raw_same_index_time_seconds', 'raw_prefix_rgb_mae'])
        writer.writerows((i, meta['sampled_indices_src'][i], meta['sampled_timestamps_sec'][i],
                          prefix_pts[i], prefix_comparison['per_frame_mae'][i]) for i in range(len(official)))
    print(json.dumps({k: report[k] for k in ('status', 'video_id', 'source_frame_count_matches_retrieval_record',
                                           'source_fps_matches_retrieval_record')}, indent=2))
    print(json.dumps({'raw_frames': derived_meta['source_total_frames'],
                      'source_last_pts_seconds': derived_meta['last_pts_seconds'],
                      'released_npy_prefix_mae': prefix_comparison['mean_absolute_rgb_error_0_to_255'],
                      'released_npy_full_range_mae': full_range_comparison['mean_absolute_rgb_error_0_to_255'],
                      'appearance_matches': searches}, indent=2))


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads((output/'analysis.json').read_text())
    baseline = np.load(report['official_npy'], allow_pickle=False)
    prefix = np.load(output/'raw_referenced_frames.npy', allow_pickle=False)
    derived = np.load(output/'verified_full_range.npy', allow_pickle=False)
    arrays = (baseline, prefix, derived)
    times = (report['official_source_metadata']['sampled_timestamps_sec'],
             report['raw_referenced_frame_pts_seconds'],
             report['verified_extraction_metadata']['sampled_timestamps_sec'])
    titles = ('Released NPY: recorded source time', 'Acquired video: same frame indices',
              'Acquired video: verified full-range sampling')
    fig, axes = plt.subplots(3, 8, figsize=(16, 7))
    fig.subplots_adjust(left=.02, right=.99, top=.87, bottom=.04, hspace=.60, wspace=.08)
    for row, (array, pts, title) in enumerate(zip(arrays, times, titles)):
        for ax, index in zip(axes[row], np.linspace(0, len(array)-1, 8).round().astype(int)):
            ax.imshow(array[index])
            ax.set_title(f'i={index}, t={pts[index]:.2f}s', fontsize=9)
            ax.axis('off')
        fig.text(.02, (.92, .62, .32)[row], title, fontsize=11)
    fig.suptitle(f'{report["video_id"]}: pixel sampling audit (no model execution)', fontsize=14)
    fig.savefig(output/'frame_comparison.png', dpi=160)
    fig.savefig(output/'frame_comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--video-id')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if args.plot_only:
        plot(args.output)
    else:
        if args.raw is None:
            parser.error('--raw is required for analysis')
        analyze(args.raw, args.video_id or args.raw.stem, args.output)
