#!/usr/bin/env python3
"""Post-run, read-only diagnostics for the frozen localization probe."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

from inspect_charades_localization_cases import ROOT, jsonl, save, sha, font


def span_key(condition):
    return tuple(condition['span']) if condition['span'] is not None else None


def main():
    result_path = ROOT / 'reports/49_charades_localization_probe_result.json'
    run = json.loads(result_path.read_text())
    assert run['status'] == 'complete' and run['resource_release']['passed']
    output = Path(run['output'])
    records = jsonl(output / 'paired_predictions.jsonl')
    assert sha(output / 'paired_predictions.jsonl') == run['comparison']['predictions_sha256']
    audits = {(r['index'], r['arm']): r for r in jsonl(output / 'generation_audit.jsonl')}
    diagnostics = {}
    for group in ['dev', 'holdout', 'all']:
        rows = [r for r in records if group == 'all' or r['group'] == group]
        part = {}
        for arm in ['ordinary', 'evidence']:
            executed = [r for r in rows if r['conditions'][arm]['executed']]
            evidence_notes = []
            for r in executed:
                c = r['conditions'][arm]
                thought = re.search(r'<think>(.*?)</think>', c['raw_output'], re.S)
                numbers = [float(s) for s in re.findall(r'(?<![\w.])\d+(?:\.\d+)?(?![\w.])', thought.group(1) if thought else '')]
                labels = [float(s) for s in audits[r['index'], arm]['time_labels']]
                boundaries = list(r['conditions']['baseline']['span']) + (c['span'] or [])
                independent = [v for v in numbers if any(abs(v - t) < .051 for t in labels)
                               and all(abs(v - b) >= .051 for b in boundaries)]
                evidence_notes.append(dict(index=r['index'], nonboundary_display_labels_mentioned=independent))
            part[arm] = dict(
                executed=len(executed), not_executed=len(rows) - len(executed),
                executed_fallbacks=sum(r['conditions'][arm]['fallback'] for r in executed),
                effective_span_equal_to_baseline=sum(span_key(r['conditions'][arm]) == span_key(r['conditions']['baseline']) for r in executed),
                outputs_with_nonboundary_display_label_mentions=sum(bool(x['nonboundary_display_labels_mentioned']) for x in evidence_notes),
                evidence_notes=evidence_notes)
        diagnostics[group] = part
    assets = json.loads((ROOT / 'reports/47_charades_localization_probe_prepare.json').read_text())
    by_index = {r['index']: r for r in assets['selection']}
    predicted = {r['index']: r for r in records}
    figure = Image.new('RGB', (1560, 890), 'white')
    draw = ImageDraw.Draw(figure)
    draw.text((15, 10), 'Post-run diagnostic examples | published frames / source time | no new inference', fill='black', font=font(21))
    selections = [(11823, [8., 9., 10., 11., 16., 18.]), (9609, [1., 3., 5.5, 8.5, 11.5, 15.5])]
    frame_evidence = []
    for row_number, (i, requested_times) in enumerate(selections):
        row, prediction = by_index[i], predicted[i]
        array = np.load(row['pixel_path'], allow_pickle=False)
        times = row['metadata']['sampled_timestamps_sec']
        indices = [min(range(len(times)), key=lambda j: abs(times[j] - t)) for t in requested_times]
        y = 55 + row_number * 415
        draw.text((15, y), f'{row["group"]} index {i} / {row["video_id"]} / GT {row["gt"]}', fill='black', font=font(22))
        text = ' | '.join(f'{arm}: {prediction["conditions"][arm]["span"]}' for arm in ['baseline', 'ordinary', 'evidence'])
        draw.text((15, y + 35), text, fill='#a137aa', font=font(16))
        for col, j in enumerate(indices):
            x = col * 260 + 5
            frame = Image.fromarray(array[j])
            frame.thumbnail((250, 255), Image.Resampling.LANCZOS)
            figure.paste(frame, (x + (250 - frame.width) // 2, y + 70))
            color = '#258445' if row['gt'][0] <= times[j] <= row['gt'][1] else '#555555'
            draw.text((x, y + 335), f'i={j} / source {times[j]:.2f}s', fill=color, font=font(16))
            frame_evidence.append(dict(index=i, frame_index=j, source_seconds=times[j]))
    figure_path = output / 'postrun_examples.png'
    figure.save(figure_path)
    summary = dict(status='POSTRUN_DIAGNOSTICS_COMPLETE', output=str(output),
                   result_report_sha256=sha(result_path), diagnostics=diagnostics,
                   figure=str(figure_path), figure_sha256=sha(figure_path), frame_evidence=frame_evidence,
                   limitations=['Timestamp mention counts are a heuristic adherence check, not proof of visual grounding.',
                                'Only numeric labels other than prior/current interval endpoints are counted.',
                                'Examples are post-hoc diagnostic illustrations, not used to select or tune this run.',
                                'Frames come from published sampled NPYs, not newly decoded raw videos.'])
    save(output / 'postrun_diagnostics.json', summary)
    save(ROOT / 'reports/50_charades_localization_probe_diagnostics.json', summary)
    progress = json.loads((output / 'progress.json').read_text())
    progress.update(status='complete', completed_at=run['completed_at'], resource_release_passed=True)
    save(output / 'progress.json', progress)
    compact = {g: {a: {k: v for k, v in s.items() if k != 'evidence_notes'}
                   for a, s in d.items()} for g, d in diagnostics.items()}
    print(json.dumps(dict(status=summary['status'], diagnostics=compact)))


if __name__ == '__main__':
    main()
