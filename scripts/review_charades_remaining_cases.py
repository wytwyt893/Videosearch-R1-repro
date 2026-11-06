#!/usr/bin/env python3
"""Package the human visual review; no inference, relabeling or Markdown."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from inspect_charades_localization_cases import ROOT, DATA, font, jsonl, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    import json
    reconstruction = json.loads((directory / 'analysis.json').read_text())
    assert reconstruction['status'] == 'REMAINING_CASE_INPUT_RECONSTRUCTION_PASSED'
    cases = {c['index']: c for c in reconstruction['cases']}
    annotation_path = DATA / 'raw_annotation/test.jsonl'
    annotations = jsonl(annotation_path)
    author_path = ROOT / 'cache/official_eval/charades-stage2/c78dc49c6b2cffdf9ae36bd01b738f9b9cdce283/external_verified_test_temporal_grounding.check.jsonl'
    author = jsonl(author_path)
    assert annotations[81]['cog_desc'] == 'a person sits at a table.'
    assert annotations[3506]['video'] == cases[81]['video_id']
    assert annotations[3506]['cog_desc'] == 'person pours a cup of coffee.'
    assert annotations[3506]['time'] == [8.9, 23.2]
    assert [author[81]['final_pred_start'], author[81]['final_pred_end']] == cases[81]['spans']['B_source']

    # Manually reviewed observations, not automatic action labels or new GT.
    notes = {
        2233: [(10, 'Glass at mouth'), (12, 'Glass at mouth'),
               (16, 'Looking at phone; glass lowered'), (19, 'Phone / holding glass')],
        81: [(7, 'Approaching / sitting at table'), (10, 'Pot begins to tilt'),
             (15, 'Pot tilted toward cup'), (19, 'Pot upright again')],
    }
    width, row_h, margin, cell_w = 1500, 380, 14, 368
    figure = Image.new('RGB', (width, row_h * 2 + 65), 'white')
    draw = ImageDraw.Draw(figure)
    draw.text((margin, 10), 'Actual processor RGB | source-frame times | human visual review', fill='black', font=font(23))
    draw.text((margin, 42), 'Display resized only. Observations are diagnostic; official annotations and scores remain unchanged.', fill='#555555', font=font(16))
    reviewed = []
    for row, i in enumerate([2233, 81]):
        case = cases[i]
        y = 70 + row * row_h
        gt, prediction = case['spans']['GT'], case['spans']['B_source']
        draw.text((margin, y), f'Case {i} / {case["video_id"]} / {"DRINKING" if i == 2233 else "POURING"}', fill='black', font=font(24))
        draw.text((margin, y + 35), f'GT {gt[0]:g}-{gt[1]:g}s', fill='#258445', font=font(20))
        draw.text((330, y + 35), f'Source-clock prediction {prediction[0]:g}-{prediction[1]:g}s; IoU {case["B_iou"]:.3f}', fill='#a137aa', font=font(20))
        evidence = []
        for col, (frame_index, observation) in enumerate(notes[i]):
            frame_path = Path(case['directory']) / 'frames' / f'{frame_index:03d}.png'
            frame = Image.open(frame_path).convert('RGB')
            frame.thumbnail((cell_w - 8, 195), Image.Resampling.LANCZOS)
            x = margin + col * cell_w
            figure.paste(frame, (x + (cell_w - 8 - frame.width) // 2, y + 75))
            time = case['source_times'][frame_index]
            draw.text((x, y + 275), f'frame {frame_index:02d} / source {time:.2f}s', fill='black', font=font(18))
            draw.text((x, y + 305), observation, fill='black', font=font(16))
            evidence.append({'frame_index': frame_index, 'source_seconds': time,
                             'manual_observation': observation, 'path': str(frame_path),
                             'sha256': sha(frame_path)})
        related = [j for j, a in enumerate(annotations) if a['video'] == case['video_id']]
        reviewed.append({'index': i, 'video_id': case['video_id'], 'frames': evidence,
                         'annotation': annotations[i],
                         'same_video_annotations': [{'index': j, **annotations[j]} for j in related],
                         'source_clock_span': prediction, 'source_clock_iou': case['B_iou'],
                         'author_span': [author[i]['final_pred_start'], author[i]['final_pred_end']],
                         'author_iou': author[i]['final_iou_r1']})
    figure_path = directory / 'evidence_overview.png'
    figure.save(figure_path)
    review = {
        'status': 'MANUAL_VISUAL_REVIEW_COMPLETED',
        'reconstruction_report_sha256': sha(directory / 'analysis.json'),
        'annotation_sha256': sha(annotation_path), 'author_predictions_sha256': sha(author_path),
        'figure': str(figure_path), 'figure_sha256': sha(figure_path), 'cases': reviewed,
        'interpretation': {
            '2233': 'Action-evidence mismatch: mouth-contact frames are present, but the predicted interval shows mainly phone/holding behavior. Scene/action confusion is a hypothesis, not an attention diagnosis.',
            '81': 'The row describes sitting in cog_desc but pouring in fig_desc under the same time label. A separate pouring row in this video has a longer label. This is strong evidence of query/label inconsistency; full construction provenance was not audited. The prediction also includes pre-action context and is not established as an ideal boundary.',
        },
        'limits': [
            'Two purposively selected cases do not estimate error-category prevalence.',
            'Known nominal/source timestamp transport mismatch is corrected; original-video PTS and annotation provenance were not reverified.',
            'Official GT, model weights, existing predictions, full-run reports and metrics are unchanged.',
            'No GPU, model generation, new labels or Markdown were produced.',
        ],
        'primary_sources': {
            'QD-DETR': 'https://arxiv.org/abs/2303.13874',
            'HawkEye': 'https://arxiv.org/html/2403.10228v1',
            'BERT passage reranking': 'https://arxiv.org/html/1901.04085v5',
            'MILL': 'https://aclanthology.org/2024.naacl-long.138/',
        },
    }
    save(directory / 'visual_review.json', review)
    print(json.dumps({'status': review['status'], 'figure': str(figure_path),
                      'review': str(directory / 'visual_review.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
