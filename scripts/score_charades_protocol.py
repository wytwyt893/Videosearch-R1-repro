#!/usr/bin/env python3
"""Score saved Charades trajectories; primary VER uses the last checked video.

No model imports or inference. VR/VCMR use the official scoring helpers without
altering the official predictions or report. First-turn VER is opt-in only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def official_helpers():
    path = ROOT / 'upstream/scripts/eval/write_temporal_result.py'
    spec = importlib.util.spec_from_file_location('official_temporal_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verification(pairs, expected_count):
    counts = Counter()
    for answer, is_gt in pairs:
        if answer not in {'matched', 'not_matched'}:
            counts['invalid'] += 1
            continue
        label = ('TP' if is_gt else 'FP') if answer == 'matched' else ('FN' if is_gt else 'TN')
        counts[label] += 1
    correct = counts['TP'] + counts['TN']
    valid = expected_count - counts['invalid']
    return {
        'N': valid, 'N_expected': expected_count, 'correct': correct,
        'accuracy': correct / valid if valid else None,
        'invalid_answers': counts['invalid'], 'coverage': valid / expected_count,
        'accuracy_all_examples': correct / expected_count,
        'confusion': {key: counts[key] for key in ('TP', 'TN', 'FP', 'FN')},
    }


def score_rows(rows, expected_count, include_first_turn=False):
    if expected_count < 1:
        raise ValueError('Expected count must be positive')
    indices = [row.get('index') for row in rows]
    if (len(rows) != expected_count or any(type(i) is not int for i in indices)
            or len(set(indices)) != len(indices)
            or set(indices) != set(range(expected_count))):
        raise ValueError('Require every original index exactly once; do not deduplicate by qid')

    last_pairs, first_pairs, native_pairs, final_ranks, final_ious = [], [], [], [], []
    terminal = Counter()
    for row in rows:
        turns = row.get('turns')
        if (row.get('max_turn') != 2 or not isinstance(turns, list) or not 1 <= len(turns) <= 2
                or [t.get('turn') for t in turns] != list(range(1, len(turns) + 1))):
            raise ValueError(f'Invalid T=2 trajectory at index {row["index"]}')
        gt = row.get('gt_pos_doc_id')
        if not isinstance(gt, str) or not gt:
            raise ValueError('Ground-truth video id is required')
        for turn in turns:
            doc = turn.get('top1_doc_id')
            if not isinstance(doc, str) or not doc or 'answer' not in turn:
                raise ValueError('Checked video and answer must be recorded in the same turn')
            if type(turn.get('top1_is_gt')) is not bool or turn['top1_is_gt'] != (doc == gt):
                raise ValueError('Checked-video label disagrees with its actual video id')
        first, last = turns[0], turns[-1]
        if row.get('final_answer') != last['answer']:
            raise ValueError('Final answer does not describe the last actual verification')
        rank = row.get('final_rank')
        if type(rank) is not int or not 1 <= rank <= 1334 or rank != last.get('policy_rank'):
            raise ValueError('Final VR must use the saved final policy ranking')
        final_doc = row.get('final_top1_doc_id')
        if not isinstance(final_doc, str) or not final_doc or final_doc != last.get('policy_top1_doc_id'):
            raise ValueError('Final retrieved video disagrees with the final policy')
        final_is_gt = final_doc == gt
        if type(row.get('final_top1_is_gt')) is not bool or row['final_top1_is_gt'] != final_is_gt:
            raise ValueError('Final retrieved-video label disagrees with its actual video id')
        iou = row.get('final_iou_r1')
        if (type(iou) not in (int, float) or not math.isfinite(iou) or not 0 <= iou <= 1
                or not math.isclose(iou, last.get('iou_r1', -1), abs_tol=1e-9)):
            raise ValueError('VCMR requires the last verified video\'s valid IoU@R1')
        if iou > 0 and not (last['answer'] == 'matched' and last['top1_is_gt']
                            and last.get('time_parse_ok') is True):
            raise ValueError('A rejected, wrong, or unlocalized video cannot receive VCMR credit')
        last_pairs.append((last['answer'], last['top1_is_gt']))
        first_pairs.append((first['answer'], first['top1_is_gt']))
        native_pairs.append((row['final_answer'], final_is_gt))
        final_ranks.append(rank)
        final_ious.append(iou)
        if last['turn'] == 2 and last.get('refine_applied'):
            terminal['refined_after_second_verification'] += 1
            terminal['changed_top1_video'] += int(final_doc != last['top1_doc_id'])
            terminal['changed_video_label'] += int(final_is_gt != last['top1_is_gt'])

    helpers = official_helpers()
    output = {
        'protocol': {
            'primary_ver_turn': 'last_verified',
            'primary_ver_pairing': 'turns[-1].answer versus turns[-1].top1_is_gt',
            'ver_denominator': 'valid_answers; invalid count and all-example accuracy reported explicitly',
            'terminal_refinement': 'keep_official_when_triggered; no extra verification beyond T=2',
            'vr_ranking': 'final_rank after any terminal refinement',
            'vcmr': 'official final_iou_r1 from the actual verified video',
            'paper_ver_turn_confirmed': False,
        },
        'num_examples': len(rows),
        'primary': {
            'VER': verification(last_pairs, expected_count),
            'VR': helpers.rank_metrics(final_ranks),
            'VCMR': helpers.temporal_metrics(final_ious),
        },
        'diagnostics': {
            'terminal_refinement': {key: terminal[key] for key in (
                'refined_after_second_verification', 'changed_top1_video', 'changed_video_label')},
            'official_final_field_pairing_VER': verification(native_pairs, expected_count),
            'official_pairing_is_primary': False,
        },
    }
    if include_first_turn:
        output['supplementary'] = {'first_turn_VER': verification(first_pairs, expected_count)}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--expected-count', type=int, default=3720)
    parser.add_argument('--include-first-turn', action='store_true')
    args = parser.parse_args()
    if os.environ.get('VIDEOSEARCH_VER_TURN', 'last_verified') != 'last_verified':
        parser.error('Primary VER is frozen to last_verified; use --include-first-turn for a supplement')
    if os.environ.get('VIDEOSEARCH_VER_DENOMINATOR', 'valid_answers') != 'valid_answers':
        parser.error('The selected baseline retains the official valid-answer denominator')
    if os.environ.get('VIDEOSEARCH_TERMINAL_REFINEMENT', 'keep_official') != 'keep_official':
        parser.error('This scorer implements the selected baseline with official terminal refinement')
    if args.output.resolve() == args.predictions.resolve() or args.output.exists():
        parser.error('Choose a new output file; existing predictions and results are preserved')
    raw = args.predictions.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    annotations = [json.loads(line) for line in
                   (ROOT / 'data/charades-sta/raw_annotation/test.jsonl').read_text().splitlines()
                   if line.strip()]
    for row in rows:
        i = row.get('index')
        if type(i) is not int or not 0 <= i < len(annotations):
            parser.error('Predicted index is outside the official test set')
        annotation = annotations[i]
        if (row.get('gt_pos_doc_id') != annotation['video']
                or row.get('query') != annotation['fig_desc']
                or row.get('gt_start') != annotation['time'][0]
                or row.get('gt_end') != annotation['time'][1]):
            parser.error(f'Prediction does not match official annotation index {i}')
    result = score_rows(rows, args.expected_count, args.include_first_turn)
    result['source_jsonl'] = str(args.predictions.resolve())
    result['source_sha256'] = hashlib.sha256(raw).hexdigest()
    result['scorer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'output': str(args.output), 'primary': result['primary'],
                      'diagnostics': result['diagnostics']['terminal_refinement']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
