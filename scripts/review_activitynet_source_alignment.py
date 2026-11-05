#!/usr/bin/env python3
"""Offline orientation/time-vs-ordinal diagnostics for rejected source candidates.

This script proposes no automatic acceptance and never edits source videos,
annotations, accepted repairs or thresholds. Review evidence precedes scoring.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'cache/activitynet_repair_v1'
OUT = ROOT/'outputs/activitynet_repair_source_review_v1'


def analyze():
    import av
    meta = {r['video_id']: r for r in map(json.loads, (ROOT/'data/activitynet/test/video_npy_with_meta/meta.jsonl').open())}
    records = [json.loads(p.read_text()) for p in (WORK/'qualification').glob('*.json')]
    accepted = {r['video_id'] for r in records if r['status'] == 'QUALIFIED_REPAIRED'}
    todo = [r for r in records if r['video_id'] not in accepted]
    OUT.mkdir(exist_ok=True)
    output = []
    for candidate in todo:
        video = candidate['video_id']
        row = meta[video]
        baseline = np.load(ROOT/'data/activitynet'/row['output_npy_path'], mmap_mode='r', allow_pickle=False)
        probe_positions = np.linspace(0, len(baseline)-1, 8).round().astype(int)
        probes = np.stack([np.asarray(Image.fromarray(baseline[i]).resize((64, 48), Image.Resampling.BICUBIC)) for i in probe_positions])
        target_times = np.array([row['sampled_timestamps_sec'][i] for i in probe_positions])
        ordinals = [row['sampled_indices_src'][i] for i in probe_positions]
        thumbs, times = [], []
        raw_path = ROOT/candidate['raw_video_path']
        with av.open(str(raw_path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 2
            first = None
            for i, frame in enumerate(container.decode(stream)):
                t = float(frame.pts*frame.time_base)
                first = t if first is None else first
                relative = t-first
                rgb = frame.to_ndarray(format='rgb24')
                factor = 128/max(rgb.shape[:2])
                thumb = np.asarray(Image.fromarray(rgb).resize((max(1, int(rgb.shape[1]*factor)),
                                                               max(1, int(rgb.shape[0]*factor))), Image.Resampling.BICUBIC))
                times.append(relative)
                thumbs.append(thumb)
                if i >= max(ordinals) and relative > max(target_times)+.5:
                    break
        nearest = [int(np.abs(np.asarray(times)-t).argmin()) for t in target_times]
        modes = {'source_pts': nearest}
        if max(ordinals) < len(thumbs):
            modes['source_frame_ordinal'] = ordinals
        candidates = []
        for mode, indices in modes.items():
            for turns in range(4):
                images = np.stack([np.asarray(Image.fromarray(np.rot90(thumbs[i], turns)).resize((64, 48), Image.Resampling.BICUBIC)) for i in indices])
                errors = np.abs(images.astype(np.float32)-probes.astype(np.float32)).mean(axis=(1, 2, 3))
                native = thumbs[0]
                h, w = native.shape[:2]
                if turns % 2:
                    h, w = w, h
                candidates.append({'mode': mode, 'rotate_ccw_degrees': turns*90,
                                   'median_rgb_mae': float(np.median(errors)),
                                   'p90_rgb_mae': float(np.quantile(errors, .9)),
                                   'aspect_relative_difference': abs((w/h)/(row['output_width']/row['output_height'])-1),
                                   'indices': indices})
        best = min(candidates, key=lambda r: r['median_rgb_mae'])
        selected = np.stack([np.asarray(Image.fromarray(np.rot90(thumbs[i], best['rotate_ccw_degrees']//90)).resize((64, 48), Image.Resampling.BICUBIC)) for i in best['indices']])
        key = video+'__'+candidate['source']
        np.savez(OUT/(key+'.npz'), baseline=probes, selected=selected)
        output.append({'video_id': video, 'source': candidate['source'],
                       'raw_video_path': candidate['raw_video_path'],
                       'original_rejection_reasons': candidate['failure_reasons'],
                       'source_validation': candidate.get('validation'),
                       'best_alignment': best, 'all_alignments': candidates,
                       'images': str((OUT/(key+'.npz')).relative_to(ROOT)),
                       'decision': 'REVIEW_REQUIRED; diagnostic only'})
        print(video, candidate['source'], best, flush=True)
    (OUT/'diagnostics.json').write_text(json.dumps(output, indent=2)+'\n')


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = json.loads((OUT/'diagnostics.json').read_text())
    for page in range((len(records)+3)//4):
        rows = records[page*4:(page+1)*4]
        fig, axes = plt.subplots(len(rows)*2, 4, figsize=(12, len(rows)*3.1), squeeze=False)
        fig.subplots_adjust(left=.02, right=.99, top=.96, bottom=.02, hspace=.5, wspace=.06)
        for i, row in enumerate(rows):
            data = np.load(ROOT/row['images'])
            for offset, key in enumerate(('baseline', 'selected')):
                for ax, j in zip(axes[i*2+offset], (0, 2, 5, 7)):
                    ax.imshow(data[key][j]);ax.axis('off')
                    ax.set_title(f'{row["video_id"]} {key} {j}', fontsize=9)
        fig.savefig(OUT/f'review_{page+1:02d}.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    plot() if args.plot_only else analyze()
