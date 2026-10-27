#!/usr/bin/env python3
"""CPU-only analysis of frozen Charades trajectories and released predictions.

No model, CUDA, network or configuration changes. Requires already verified
reference files and creates a fresh analysis directory, never a new prediction.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '69234fd9783b7235164d9f92b8903d9781dc284e'
REVISION = 'c78dc49c6b2cffdf9ae36bd01b738f9b9cdce283'
BASE = 'external_verified_test_temporal_grounding_VideoSearchR1__charades-stage2'
THRESHOLDS = (0.3, 0.5, 0.7)


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_scorer():
    path = ROOT / 'scripts/score_charades_protocol.py'
    spec = importlib.util.spec_from_file_location('frozen_protocol', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_parser():
    """Compile only official regexes and four pure functions, not model imports."""
    path = ROOT / 'upstream/scripts/internal/stage1/eval_verified_test_rerank_temporal_grounding.py'
    tree = ast.parse(path.read_text())
    functions = {'_parse_answer', '_parse_float', '_parse_temporal_span', '_compute_iou'}
    constants = {'ANSWER_RE', 'START_RE', 'END_RE', 'FLOAT_RE'}
    body = [node for node in tree.body if
            (isinstance(node, ast.FunctionDef) and node.name in functions) or
            (isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in constants for target in node.targets))]
    if len(body) != len(functions) + len(constants):
        raise ValueError('Official parser layout changed')
    namespace = {'re': re, 'np': np, 'Optional': Optional, 'Tuple': Tuple}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def near(a, b):
    return a is b if a is None or b is None else math.isclose(a, b, rel_tol=0, abs_tol=1e-9)


def validate_rows(rows, annotations, parser):
    if [row['index'] for row in rows] != list(range(len(annotations))):
        raise ValueError('Require every original index in order, exactly once')
    problems, turn_count = [], 0
    for row, annotation in zip(rows, annotations):
        if (row['query'] != annotation['fig_desc'] or row['gt_pos_doc_id'] != annotation['video']
                or [row['gt_start'], row['gt_end']] != annotation['time']):
            raise ValueError(f'Annotation mismatch at index {row["index"]}')
        for turn in row['turns']:
            turn_count += 1
            answer = parser['_parse_answer'](turn['raw_output'])
            span = parser['_parse_temporal_span'](turn['raw_output'], annotation['duration'])
            raw_iou = (parser['_compute_iou'](annotation['time'], span)
                       if answer == 'matched' and span is not None else 0.0)
            scored_iou = raw_iou if turn['top1_doc_id'] == annotation['video'] else 0.0
            checks = {'answer': answer == turn['answer'],
                      'time_parse_ok': (span is not None) == turn['time_parse_ok'],
                      'start': near(span[0] if span else None, turn['pred_start']),
                      'end': near(span[1] if span else None, turn['pred_end']),
                      'iou_raw': near(raw_iou, turn['iou_raw']),
                      'iou_r1': near(scored_iou, turn['iou_r1'])}
            for check, passed in checks.items():
                if not passed:
                    problems.append({'index': row['index'], 'turn': turn['turn'], 'check': check})
    return {'examples': len(rows), 'annotations_aligned': True, 'turns_reparsed': turn_count,
            'parser_and_iou_mismatch_count': len(problems), 'mismatches': problems}


def failure_category(row, threshold):
    turn = row['turns'][-1]
    if turn['top1_doc_id'] != row['gt_pos_doc_id']:
        return 'wrong_checked_video'
    if turn['answer'] == 'not_matched':
        return 'correct_video_rejected'
    if turn['answer'] != 'matched':
        return 'correct_video_invalid_answer'
    if not turn['time_parse_ok']:
        return 'correct_video_accepted_time_parse_failed'
    return 'hit' if row['final_iou_r1'] >= threshold else 'correct_video_localization_below_threshold'


def distribution(values):
    if not values:
        return {'N': 0}
    arr = np.asarray(values, dtype=np.float64)
    return {'N': len(values), 'mean': float(arr.mean()), 'median': float(np.median(arr)),
            'p10': float(np.quantile(arr, 0.1)), 'p90': float(np.quantile(arr, 0.9)),
            'min': float(arr.min()), 'max': float(arr.max())}


def summarize(rows, scorer):
    protocol = scorer.score_rows(rows, len(rows), include_first_turn=True)
    first_pairs = [(r['turns'][0]['answer'], r['turns'][0]['top1_is_gt']) for r in rows]
    last_pairs = [(r['turns'][-1]['answer'], r['turns'][-1]['top1_is_gt']) for r in rows]
    transition = Counter()
    for (first, first_gt), (last, last_gt) in zip(first_pairs, last_pairs):
        first_ok = first in {'matched', 'not_matched'} and (first == 'matched') == first_gt
        last_ok = last in {'matched', 'not_matched'} and (last == 'matched') == last_gt
        transition[f'first_{"correct" if first_ok else "wrong"}__last_{"correct" if last_ok else "wrong"}'] += 1
    buckets = {}
    for threshold in THRESHOLDS:
        counts = Counter(failure_category(row, threshold) for row in rows)
        assert sum(counts.values()) == len(rows)
        buckets[str(threshold)] = dict(counts)
    selected = [r for r in rows if r['turns'][-1]['top1_is_gt']
                and r['turns'][-1]['answer'] == 'matched' and r['turns'][-1]['time_parse_ok']]
    start_errors = [r['final_pred_start'] - r['gt_start'] for r in selected]
    end_errors = [r['final_pred_end'] - r['gt_end'] for r in selected]
    iou_bins = Counter()
    for row in selected:
        value = row['final_iou_r1']
        label = '0' if value == 0 else next((f'[{lo},{hi})' for lo, hi in
            [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9)] if lo <= value < hi), '[0.9,1]')
        iou_bins[label] += 1
    turn_views = {}
    for turn_no in (1, 2):
        pairs = [(r['turns'][turn_no-1]['answer'], r['turns'][turn_no-1]['top1_is_gt'])
                 for r in rows if len(r['turns']) >= turn_no]
        turn_views[str(turn_no)] = scorer.verification(pairs, len(pairs))
    boundary = {}
    values = [r['final_iou_r1'] for r in rows]
    for threshold in THRESHOLDS:
        rounded = np.asarray(values, dtype=np.float32) >= threshold
        ids = [r['index'] for r, flag in zip(rows, rounded)
               if (r['final_iou_r1'] >= threshold) != bool(flag)]
        boundary[str(threshold)] = {'jsonl_float_hits': sum(v >= threshold for v in values),
                                   'float32_hits': int(rounded.sum()), 'different_indices': ids}
    before_terminal_ranks = [r['turns'][-1]['rank_before_refine'] if
                             r['turns'][-1]['turn'] == 2 and r['turns'][-1]['refine_applied']
                             else r['final_rank'] for r in rows]
    return {'protocol': protocol, 'actual_verification_by_turn': turn_views,
            'first_last_correctness_transition': dict(transition),
            'vcmr_failure_partition': buckets,
            'localization_on_correct_accepted_parsed_videos': {
                'N': len(selected), 'iou_bins': dict(iou_bins),
                'zero_length_predictions': sum(r['final_pred_start'] == r['final_pred_end'] for r in selected),
                'start_signed_error_seconds': distribution(start_errors),
                'end_signed_error_seconds': distribution(end_errors),
                'start_absolute_error_seconds': distribution([abs(v) for v in start_errors]),
                'end_absolute_error_seconds': distribution([abs(v) for v in end_errors]),
                'duration_ratio_pred_over_gt': distribution([
                    (r['final_pred_end']-r['final_pred_start'])/(r['gt_end']-r['gt_start']) for r in selected]),
            }, 'float_boundary': boundary,
            'pre_terminal_saved_ranking_VR': scorer.official_helpers().rank_metrics(before_terminal_ranks),
            'pre_terminal_scope': 'Saved ranking before terminal update; not a new inference or training ablation.'}


def pair_comparison(local, reference):
    n = len(local)
    shared = Counter()
    losses = {}
    for left, right in zip(local, reference):
        for key in ('qid', 'orig_rank', 'final_rank', 'final_top1_doc_id'):
            shared[key + '_equal'] += left[key] == right[key]
        for key in ('top1_doc_id', 'answer'):
            shared['first_' + key + '_equal'] += left['turns'][0][key] == right['turns'][0][key]
            shared['last_' + key + '_equal'] += left['turns'][-1][key] == right['turns'][-1][key]
    common = [(a, b) for a, b in zip(local, reference) if all(
        r['turns'][-1]['top1_is_gt'] and r['turns'][-1]['answer'] == 'matched'
        and r['turns'][-1]['time_parse_ok'] for r in (a, b))]
    for threshold in THRESHOLDS:
        lost = [(a, b) for a, b in zip(local, reference)
                if b['final_iou_r1'] >= threshold and a['final_iou_r1'] < threshold]
        gained = [(a, b) for a, b in zip(local, reference)
                  if a['final_iou_r1'] >= threshold and b['final_iou_r1'] < threshold]
        losses[str(threshold)] = {
            'reference_hits': sum(r['final_iou_r1'] >= threshold for r in reference),
            'local_hits': sum(r['final_iou_r1'] >= threshold for r in local),
            'reference_hit_local_miss': len(lost), 'local_hit_reference_miss': len(gained),
            'net_local_minus_reference': len(gained)-len(lost),
            'lost_local_failure_types': dict(Counter(failure_category(a, threshold) for a, _ in lost)),
            'gained_reference_failure_types': dict(Counter(failure_category(b, threshold) for _, b in gained)),
            'lost_indices': [a['index'] for a, _ in lost],
            'gained_indices': [a['index'] for a, _ in gained]}
        assert losses[str(threshold)]['local_hits']-losses[str(threshold)]['reference_hits'] == len(gained)-len(lost)
    return {'N': n, 'same_counts': dict(shared), 'vcmr_paired_differences': losses,
            'both_correct_video_accepted_and_parsed': {
                'N': len(common),
                'same_span': sum(a['final_pred_start'] == b['final_pred_start'] and
                                 a['final_pred_end'] == b['final_pred_end'] for a, b in common),
                'local_iou_lower': sum(a['final_iou_r1'] < b['final_iou_r1'] for a, b in common),
                'local_iou_higher': sum(a['final_iou_r1'] > b['final_iou_r1'] for a, b in common),
                'local_minus_reference_iou': distribution([a['final_iou_r1']-b['final_iou_r1'] for a, b in common]),
            }}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, default=ROOT / 'outputs/charades_full_20261027_161353_178368')
    ap.add_argument('--output', type=Path, required=True, help='New analysis directory; existing files refused')
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Choose a new analysis output directory')
    if subprocess.check_output(['git', '-C', str(ROOT/'upstream'), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise ValueError('Unexpected official source revision')
    if subprocess.check_output(['git', '-C', str(ROOT/'upstream'), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Official source is dirty')
    manifest = json.loads((args.run/'run_manifest.json').read_text())
    local_path = args.run/(BASE+'.jsonl')
    reference_dir = ROOT/'cache/official_eval/charades-stage2'/REVISION
    reference_path = reference_dir/'external_verified_test_temporal_grounding.check.jsonl'
    metadata_path = ROOT/'cache/manifests/model.json'
    metadata = json.loads(metadata_path.read_text())
    item = next(r for r in metadata['siblings'] if r['rfilename'] == 'eval/'+reference_path.name)
    assert metadata['sha'] == REVISION and digest(reference_path) == item['lfs']['sha256']
    assert reference_path.stat().st_size == item['size']
    assert digest(local_path) == manifest['prediction_sha256']
    assert digest(Path(manifest['primary_result'])) == manifest['primary_result_sha256']
    frozen_paths = [local_path, Path(manifest['primary_result']), args.run/'run_manifest.json',
                    ROOT/'configs/charades_stage2_eval.bash', reference_path,
                    reference_dir/'result.json', metadata_path,
                    ROOT/'data/charades-sta/raw_annotation/test.jsonl',
                    ROOT/'reports/08_cpu_preflight.json']
    before = {str(p.relative_to(ROOT)): digest(p) for p in frozen_paths}
    local, reference = load_jsonl(local_path), load_jsonl(reference_path)
    annotations = load_jsonl(ROOT/'data/charades-sta/raw_annotation/test.jsonl')
    if len(annotations) != 3720:
        raise ValueError('Unexpected Charades test count')
    parser = official_parser()
    local_validation = validate_rows(local, annotations, parser)
    reference_validation = validate_rows(reference, annotations, parser)
    if local_validation['parser_and_iou_mismatch_count'] or reference_validation['parser_and_iou_mismatch_count']:
        raise ValueError(json.dumps({'local': local_validation, 'reference': reference_validation}))
    scorer = load_scorer()
    local_summary, reference_summary = summarize(local, scorer), summarize(reference, scorer)
    assert local_summary['protocol']['primary'] == json.loads(Path(manifest['primary_result']).read_text())['primary']
    reference_result = json.loads((reference_dir/'result.json').read_text())
    aggregation_differences = []
    for key, saved_key in [('VR', 'final_retrieval'), ('VCMR', 'final_temporal')]:
        actual, saved = reference_summary['protocol']['primary'][key], reference_result[saved_key]
        assert actual.keys() == saved.keys()
        for metric, value in actual.items():
            # Aggregation order can change MRR/mIoU in the last binary digits.
            assert math.isclose(value, saved[metric], rel_tol=0, abs_tol=1e-12)
            if value != saved[metric]:
                aggregation_differences.append({'task': key, 'metric': metric,
                    'recomputed': value, 'released': saved[metric], 'delta': value-saved[metric]})
    comparison = pair_comparison(local, reference)
    preflight = json.loads((ROOT/'reports/08_cpu_preflight.json').read_text())
    duplicate_ids = set(preflight['mappings']['official_text_map_qid_overrides'])
    over_ids = {i for i, a in enumerate(annotations) if a['time'][1] > a['duration']+0.1}
    subgroup = {}
    for name, ids in [('duplicate_text_overrides', duplicate_ids), ('annotation_end_exceeds_duration', over_ids)]:
        subgroup[name] = {'N': len(ids), 'local_iou07_hits': sum(local[i]['final_iou_r1']>=0.7 for i in ids),
                         'reference_iou07_hits': sum(reference[i]['final_iou_r1']>=0.7 for i in ids),
                         'net_iou07_hits': sum(int(local[i]['final_iou_r1']>=0.7)-int(reference[i]['final_iou_r1']>=0.7) for i in ids),
                         'scope': 'Descriptive subgroup; no label edits and no causal attribution.'}
    assert before == {str(p.relative_to(ROOT)): digest(p) for p in frozen_paths}
    result = {'created_at': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'status': 'OFFLINE_MINIMUM_PASSED', 'model_execution': False, 'gpu_access': False,
              'source_sha256': before, 'analysis_script_sha256': digest(Path(__file__)),
              'reference_provenance': json.loads((reference_dir/'download_receipts.json').read_text()),
              'validation': {'local': local_validation, 'reference': reference_validation,
                             'original_sources_unchanged': True, 'official_result_reproduced': True,
                             'official_result_absolute_tolerance': 1e-12,
                             'aggregation_roundoff': aggregation_differences},
              'local': local_summary, 'reference': reference_summary, 'comparison': comparison,
              'data_subgroups': subgroup,
              'limitations': ['No new inference or visual inspection; labels define video correctness.',
                              'Reference inference backend and exact original input budget are not recorded by these predictions.',
                              'Observed trajectory/span differences do not isolate their backend or preprocessing cause.',
                              'Primary last-verification protocol remains unchanged.']}
    args.output.mkdir(parents=True)
    (args.output/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    with (args.output/'per_example.csv').open('w', newline='') as handle:
        fields = ['index', 'query', 'gt_video', 'gt_start', 'gt_end', 'annotation_duration',
                  'local_orig_rank', 'reference_orig_rank', 'local_final_rank', 'reference_final_rank',
                  'local_checked_video', 'reference_checked_video', 'local_answer', 'reference_answer',
                  'local_start', 'local_end', 'reference_start', 'reference_end', 'local_iou', 'reference_iou',
                  'local_failure_07', 'reference_failure_07', 'duplicate_text_override', 'gt_end_exceeds_duration']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for a, b, annotation in zip(local, reference, annotations):
            writer.writerow(dict(zip(fields, [a['index'], a['query'], a['gt_pos_doc_id'], a['gt_start'], a['gt_end'],
                annotation['duration'], a['orig_rank'], b['orig_rank'], a['final_rank'], b['final_rank'],
                a['turns'][-1]['top1_doc_id'], b['turns'][-1]['top1_doc_id'], a['final_answer'], b['final_answer'],
                a['final_pred_start'], a['final_pred_end'], b['final_pred_start'], b['final_pred_end'],
                a['final_iou_r1'], b['final_iou_r1'], failure_category(a, 0.7), failure_category(b, 0.7),
                a['index'] in duplicate_ids, a['index'] in over_ids])))
    print(json.dumps({'output': str(args.output), 'validation': result['validation'],
                      'local_primary': local_summary['protocol']['primary'],
                      'reference_primary': reference_summary['protocol']['primary'],
                      'same_counts': comparison['same_counts'],
                      'paired_differences': {k: {a:b for a,b in v.items() if not a.endswith('_indices')}
                                             for k,v in comparison['vcmr_paired_differences'].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
