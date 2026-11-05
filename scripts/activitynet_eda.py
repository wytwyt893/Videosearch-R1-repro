#!/usr/bin/env python3
"""CPU exploration of the released ActivityNet test core, separate from preflight."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/activitynet'


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {'N': 0}
    return {'N': len(a), 'min': float(a.min()), 'p10': float(np.quantile(a, .1)),
            'median': float(np.median(a)), 'mean': float(a.mean()),
            'p90': float(np.quantile(a, .9)), 'max': float(a.max())}


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def plot(series_path, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    series = json.loads(series_path.read_text())
    plots = [
        ('video_duration_seconds', 'Source video duration', 'seconds', '#397ca5'),
        ('moment_duration_seconds', 'Ground-truth moment duration', 'seconds', '#469b78'),
        ('queries_per_video', 'Queries per ground-truth video', 'query count', '#cc8d41'),
        ('query_word_counts', 'Fine-grained query length', 'whitespace-separated words', '#9466a5'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title, xlabel, color) in zip(axes.flat, plots):
        a = np.asarray(series[key])
        ax.hist(a, bins=min(40, max(5, len(np.unique(a)))), color=color, edgecolor='white')
        ax.axvline(np.median(a), color='#333333', linestyle='--', linewidth=1,
                   label=f'median {np.median(a):.2f}')
        ax.set(title=title, xlabel=xlabel, ylabel='videos' if key == 'video_duration_seconds' else 'examples')
        if key == 'queries_per_video':
            ax.set_ylabel('videos')
        ax.legend(frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('ActivityNet-FIG: released evaluation split', fontsize=14)
    fig.tight_layout()
    fig.savefig(target, dpi=170)
    fig.savefig(target.with_suffix('.pdf'))
    plt.close(fig)


def analyze(output, report):
    if output.exists() or report.exists():
        raise ValueError('Choose fresh output and report paths')
    paths = {key: DATA / relative for key, relative in {
        'annotations': 'raw_annotation/test.jsonl',
        'query_metadata': 'test/query_embedding/query_meta.test.jsonl',
        'query_vectors': 'test/query_embedding/query_embeddings.test.npy',
        'video_vectors': 'test/video_embedding_1fps/segment_embeds.npy',
        'mapping': 'test/video_embedding_1fps/docid2row.json',
        'video_metadata': 'test/video_npy_with_meta/meta.jsonl',
    }.items()}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise ValueError(f'Complete test core is required before EDA: {missing}')
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()}
    annotations = jsonl(paths['annotations'])
    qmeta = jsonl(paths['query_metadata'])
    metadata = {r['video_id']: r for r in jsonl(paths['video_metadata'])}
    mapping = json.loads(paths['mapping'].read_text())
    qv = np.load(paths['query_vectors'], allow_pickle=False)
    vv = np.load(paths['video_vectors'], allow_pickle=False)
    if (qv.shape != (len(qmeta), 2048) or vv.shape != (len(mapping), 2048)
            or not np.isfinite(qv).all() or not np.isfinite(vv).all()
            or set(mapping) != set(metadata) or set(mapping.values()) != set(range(len(mapping)))):
        raise ValueError('Invalid full-corpus metadata / vectors')
    by_video = Counter(a['video'] for a in annotations)
    if set(by_video) - set(mapping):
        raise ValueError('Annotated positive videos are missing from the full corpus')
    text_groups = defaultdict(list)
    for j, a in enumerate(annotations):
        text_groups[a['fig_desc'].strip()].append(j)
    duplicates = {k: v for k, v in text_groups.items() if len(v) > 1}
    by_qid = {r['qid']: r for r in qmeta}
    mapping_mismatches = []
    for j, a in enumerate(annotations):
        qid = str(a['desc_id']) + '_fig'
        q = by_qid.get(qid)
        if q is None or q['pos_doc_id'] != a['video'] or q['query'].strip() != a['fig_desc'].strip():
            mapping_mismatches.append(j)
    durations = [v['source_duration_sec'] for v in metadata.values()]
    annotation_durations = defaultdict(set)
    for a in annotations:
        annotation_durations[a['video']].add(a['duration'])
    annotation_duration_by_video = {vid: float(np.median(list(values)))
                                    for vid, values in annotation_durations.items()}
    duration_gaps = [metadata[vid]['source_duration_sec']-duration
                     for vid, duration in annotation_duration_by_video.items()]
    moments = [a['time'][1]-a['time'][0] for a in annotations]
    if any(not np.isfinite(a['time']).all() or a['time'][0] < 0 or a['time'][1] <= a['time'][0]
           for a in annotations):
        raise ValueError('Nonfinite, negative, or empty GT interval')
    frame_counts = [v['output_num_frames'] for v in metadata.values()]
    frame_targets = Counter((v['target_sample_fps'], v['target_max_frames']) for v in metadata.values())
    payload_bytes = sum(v['output_num_frames']*v['output_height']*v['output_width']*3 for v in metadata.values())
    time_ratios = [v['sampled_timestamps_sec'][-1]/(v['output_num_frames']-1)
                   for v in metadata.values() if v['output_num_frames'] > 1]
    source_long = {vid for vid, v in metadata.items() if v['source_duration_sec'] > 64}
    expected_paths = {DATA/'test/video_npy_with_meta'/f'{vid}.npy' for vid in mapping}
    complete_pixels = [p for p in expected_paths if p.is_file()]
    indexed_members = {}
    for path in (ROOT/'cache/manifests/activitynet/tar_index').glob('*.json'):
        for member in json.loads(path.read_text())['members']:
            if DATA/member['name'] in expected_paths:
                indexed_members[member['name']] = member['size']
    state_path = ROOT/'cache/activitynet_download/status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    sources = json.loads((ROOT/'cache/manifests/activitynet/sources.json').read_text())
    result = {
        'created_at': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'status': 'CORE_EDA_COMPLETE', 'model_execution': False, 'gpu_access': False,
        'scope': 'Descriptive exploration of complete released test core; not pixel integrity or inference preflight.',
        'official_test_alias': 'activitynet_fig_val_2 -> raw_annotation/test.jsonl',
        'counts': {'queries': len(annotations), 'query_vector_rows': len(qmeta),
                   'video_corpus': len(mapping), 'unique_ground_truth_videos': len(by_video),
                   'corpus_videos_without_positive_query': len(set(mapping)-set(by_video))},
        'video_duration_seconds': distribution(durations),
        'annotation_video_duration_seconds': distribution(list(annotation_duration_by_video.values())),
        'source_minus_annotation_duration_seconds': distribution(duration_gaps),
        'moment_duration_seconds': distribution(moments),
        'moment_fraction_of_annotation_duration': distribution([
            (a['time'][1]-a['time'][0])/a['duration'] for a in annotations]),
        'queries_per_video': distribution(list(by_video.values())),
        'query_word_counts': distribution([len(a['fig_desc'].split()) for a in annotations]),
        'sampling': {'frame_counts': distribution(frame_counts),
                     'configured_fps_and_max_frames': [{'fps': k[0], 'max_frames': k[1], 'videos': n}
                                                       for k, n in frame_targets.items()],
                     'source_videos_over_64_seconds': len(source_long),
                     'queries_on_source_videos_over_64_seconds': sum(a['video'] in source_long for a in annotations),
                     'stored_dtypes': dict(Counter(v['output_dtype'] for v in metadata.values())),
                     'source_last_timestamp_over_sequential_frame_last_index': distribution(time_ratios),
                     'time_axis_note': 'Source metadata span differs from nominal sequential frame positions; actual ActivityNet processor behavior must be checked in CPU preflight. No timestamp correction applied.'},
        'vectors': {'query_shape': list(qv.shape), 'video_shape': list(vv.shape),
                    'query_dtype': str(qv.dtype), 'video_dtype': str(vv.dtype),
                    'query_norms': distribution(np.linalg.norm(qv, axis=1)),
                    'video_norms': distribution(np.linalg.norm(vv, axis=1))},
        'data_observations': {
            'annotation_qid_mapping_mismatch_indices': mapping_mismatches,
            'duplicate_query_text_groups': len(duplicates),
            'duplicate_query_indices': list(duplicates.values()),
            'cross_video_duplicate_groups': sum(len({annotations[j]['video'] for j in ids}) > 1 for ids in duplicates.values()),
            'query_whitespace_indices': [j for j,a in enumerate(annotations) if a['fig_desc'] != a['fig_desc'].strip()],
            'gt_end_exceeds_annotation_duration_by_0p1_indices': [j for j,a in enumerate(annotations) if a['time'][1] > a['duration']+.1],
            'gt_end_exceeds_source_duration_by_0p1_indices': [j for j,a in enumerate(annotations)
                                                            if a['time'][1] > metadata[a['video']]['source_duration_sec']+.1],
            'videos_source_annotation_duration_differ_over_1_second': sum(abs(x)>1 for x in duration_gaps),
            'videos_source_annotation_duration_differ_over_10_seconds': sum(abs(x)>10 for x in duration_gaps),
            'videos_with_multiple_annotation_duration_values': [vid for vid,values in annotation_durations.items()
                                                                if len(values)>1],
            'policy': 'Report observations; preserve all original examples, ids, labels and pixels.'},
        'pixel_volume': {'raw_rgb_payload_bytes': payload_bytes, 'raw_rgb_payload_gib': payload_bytes/2**30,
                         'indexed_npy_files': len(indexed_members), 'indexed_npy_bytes': sum(indexed_members.values()),
                         'npy_files_present_at_snapshot': len(complete_pixels),
                         'note': 'Metadata payload excludes NPY headers; indexed size includes them. Present count is not final SHA/shape validation.'},
        'download_snapshot': state, 'source_metadata': sources,
        'input_sha256': hashes, 'example': annotations[0],
    }
    series = {'video_duration_seconds': durations, 'moment_duration_seconds': moments,
              'queries_per_video': list(by_video.values()),
              'query_word_counts': [len(a['fig_desc'].split()) for a in annotations]}
    for relative, sha in hashes.items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != sha:
            raise ValueError('EDA input changed during reading')
    output.mkdir(parents=True)
    save(output/'eda.json', result)
    save(output/'plot_data.json', series)
    save(report, result)
    with (output/'per_query.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['index', 'video', 'start', 'end', 'moment_seconds', 'annotation_duration', 'query_words'])
        writer.writerows((j,a['video'],*a['time'],a['time'][1]-a['time'][0],a['duration'],len(a['fig_desc'].split()))
                         for j,a in enumerate(annotations))
    print(json.dumps({k: result[k] for k in ('status','counts','video_duration_seconds','moment_duration_seconds',
                                           'queries_per_video','query_word_counts','sampling','vectors','pixel_volume')},
                     ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--report', type=Path, default=ROOT/'reports/21_activitynet_eda.json')
    ap.add_argument('--plot-from', type=Path)
    ap.add_argument('--plot-output', type=Path)
    args = ap.parse_args()
    if args.plot_from:
        if not args.plot_output or args.output:
            ap.error('Plot mode requires --plot-from and --plot-output only')
        plot(args.plot_from, args.plot_output)
    elif args.output:
        analyze(args.output, args.report)
    else:
        ap.error('Choose --output for fresh EDA, or --plot-from/--plot-output for figures')


if __name__ == '__main__':
    main()
