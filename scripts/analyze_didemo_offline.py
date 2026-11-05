#!/usr/bin/env python3
"""CPU-only paired analysis of frozen DiDeMo and released author predictions.

Requires previously fetched official eval artifacts, verifies their published
hashes, and refuses to overwrite an analysis directory. No model or network use.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from analyze_charades_offline import (
    COMMIT, THRESHOLDS, digest, failure_category, load_jsonl,
    official_parser, pair_comparison, summarize, validate_rows,
)

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'a576c16040c35a5ec71468c66ec7628e7db71997'
BASE = 'external_verified_test_temporal_grounding_VideoSearchR1__didemo-stage2'


def load_scorer():
    path = ROOT / 'scripts/score_didemo_protocol.py'
    spec = importlib.util.spec_from_file_location('didemo_frozen_protocol', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleasedThreeTurnScorer:
    """Use the same metric definitions without falsely relabeling T=3 as T=2."""

    def __init__(self, frozen_scorer):
        self.verification = frozen_scorer.verification
        self.official_helpers = frozen_scorer.official_helpers

    def score_rows(self, rows, expected_count, include_first_turn=False):
        if [r['index'] for r in rows] != list(range(expected_count)):
            raise ValueError('Released reference indices are not complete/in order')
        first, last, native, ranks, ious = [], [], [], [], []
        terminal = Counter()
        for row in rows:
            turns = row['turns']
            if (row.get('max_turn') != 3 or not 1 <= len(turns) <= 3
                    or [t['turn'] for t in turns] != list(range(1, len(turns)+1))):
                raise ValueError('Unexpected released T=3 trajectory')
            gt = row['gt_pos_doc_id']
            for turn in turns:
                if (type(turn.get('top1_is_gt')) is not bool
                        or turn['top1_is_gt'] != (turn['top1_doc_id'] == gt)):
                    raise ValueError('Released checked-video label mismatch')
            end = turns[-1]
            rank, doc, iou = row['final_rank'], row['final_top1_doc_id'], row['final_iou_r1']
            if (type(rank) is not int or not 1 <= rank <= 1002 or rank != end['policy_rank']
                    or doc != end['policy_top1_doc_id'] or row['final_answer'] != end['answer']
                    or type(row['final_top1_is_gt']) is not bool
                    or row['final_top1_is_gt'] != (doc == gt)
                    or not math.isfinite(iou) or not 0 <= iou <= 1
                    or not math.isclose(iou, end['iou_r1'], rel_tol=0, abs_tol=1e-9)):
                raise ValueError('Released final fields disagree with actual saved trajectory')
            if iou > 0 and not (end['answer'] == 'matched' and end['top1_is_gt']
                                and end['time_parse_ok'] is True):
                raise ValueError('Released VCMR credits an incorrect/rejected/unlocalized video')
            first.append((turns[0]['answer'], turns[0]['top1_is_gt']))
            last.append((end['answer'], end['top1_is_gt']))
            native.append((row['final_answer'], row['final_top1_is_gt']))
            ranks.append(rank)
            ious.append(iou)
            if end['turn'] == 3 and end['refine_applied']:
                terminal['refined_after_third_verification'] += 1
                terminal['changed_top1_video'] += int(doc != end['top1_doc_id'])
                terminal['changed_video_label'] += int((doc == gt) != end['top1_is_gt'])
        helpers = self.official_helpers()
        result = {
            'protocol': {
                'primary_ver_turn': 'last_verified',
                'primary_ver_pairing': 'answer and actual video in same saved turn',
                'ver_denominator': 'valid_answers; invalid count and coverage reported',
                'maximum_turns': 3,
                'vr_ranking': 'saved final_rank after any terminal refinement',
                'vcmr': 'saved final_iou_r1 from last actual verified video',
                'is_local_T2_configuration': False,
                'paper_ver_turn_confirmed': False,
            },
            'num_examples': expected_count,
            'primary': {'VER': self.verification(last, expected_count),
                        'VR': helpers.rank_metrics(ranks), 'VCMR': helpers.temporal_metrics(ious)},
            'diagnostics': {'terminal_refinement': dict(terminal),
                            'official_final_field_pairing_VER': self.verification(native, expected_count),
                            'official_pairing_is_primary': False},
        }
        if include_first_turn:
            result['supplementary'] = {'first_turn_VER': self.verification(first, expected_count)}
        return result


def first_two_turns_snapshot(rows):
    """Project already saved fields; never claim this is a new T=2 inference."""
    output = []
    for row in rows:
        turns = row['turns'][:2]
        last = turns[-1]
        copied = dict(row, turns=turns, max_turn=2,
                      final_rank=last['policy_rank'], final_top1_doc_id=last['policy_top1_doc_id'],
                      final_top1_is_gt=last['policy_top1_is_gt'], final_answer=last['answer'])
        for key in ('time_parse_ok', 'pred_start', 'pred_end', 'iou_raw', 'iou_r1'):
            copied['final_' + key] = last[key]
        output.append(copied)
    return output


def verify_artifact(path, metadata):
    import hashlib
    raw = path.read_bytes()
    if len(raw) != metadata['size']:
        raise ValueError(f'Official artifact size mismatch: {path.name}')
    if metadata.get('lfs'):
        if digest(path) != metadata['lfs']['sha256']:
            raise ValueError(f'Official LFS SHA256 mismatch: {path.name}')
    else:
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if blob != metadata['blobId']:
            raise ValueError(f'Official Git blob mismatch: {path.name}')


def subgroup_summary(local, reference, indices):
    return {
        'N': len(indices),
        'thresholds': {
            str(t): {
                'local_hits': sum(local[i]['final_iou_r1'] >= t for i in indices),
                'reference_hits': sum(reference[i]['final_iou_r1'] >= t for i in indices),
                'net_local_minus_reference': sum(
                    int(local[i]['final_iou_r1'] >= t) - int(reference[i]['final_iou_r1'] >= t)
                    for i in indices),
            } for t in THRESHOLDS
        },
        'scope': 'Descriptive subgroup; no label edits or causal attribution.',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path,
                    default=ROOT / 'outputs/didemo_full_20261028_101412_373223')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Choose a new analysis directory; existing outputs are preserved')
    if subprocess.check_output(['git', '-C', str(ROOT/'upstream'), 'rev-parse', 'HEAD'],
                               text=True).strip() != COMMIT:
        raise ValueError('Unexpected official source revision')
    if subprocess.check_output(['git', '-C', str(ROOT/'upstream'), 'status', '--porcelain'],
                               text=True).strip():
        raise ValueError('Official source is dirty')
    manifest_path = args.run / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'complete' or manifest['expected_examples'] != 4013:
        raise ValueError('A completed 4013-query DiDeMo run is required')
    local_path = args.run / (BASE + '.jsonl')
    primary_path = Path(manifest['primary_result'])
    reference_dir = ROOT / 'cache/official_eval/didemo-stage2' / REVISION
    reference_path = reference_dir / 'external_verified_test_temporal_grounding.check.jsonl'
    reference_result_path = reference_dir / 'result.json'
    metadata_path = ROOT / 'cache/manifests/didemo/model.json'
    metadata = json.loads(metadata_path.read_text())
    if metadata['sha'] != REVISION:
        raise ValueError('Unexpected reference checkpoint revision')
    for path in (reference_path, reference_result_path):
        item = next(x for x in metadata['siblings'] if x['rfilename'] == 'eval/' + path.name)
        verify_artifact(path, item)
    if digest(local_path) != manifest['prediction_sha256']:
        raise ValueError('Local prediction hash differs from completed-run evidence')
    if digest(primary_path) != manifest['primary_result_sha256']:
        raise ValueError('Local primary result hash differs from completed-run evidence')
    annotation_path = ROOT / 'data/didemo/raw_annotation/test.jsonl'
    preflight_path = ROOT / 'reports/15_didemo_cpu_preflight.json'
    video_meta_path = ROOT / 'data/didemo/test/video_npy_with_meta/meta.jsonl'
    frozen_paths = [local_path, primary_path, manifest_path,
                    ROOT/'configs/didemo_stage2_eval.bash', reference_path,
                    reference_result_path, metadata_path, annotation_path,
                    preflight_path, video_meta_path,
                    ROOT/'scripts/score_didemo_protocol.py',
                    ROOT/'scripts/analyze_charades_offline.py',
                    ROOT/'upstream/scripts/eval/write_temporal_result.py',
                    ROOT/'upstream/scripts/internal/stage1/eval_verified_test_rerank_temporal_grounding.py']
    before = {str(p.relative_to(ROOT)): digest(p) for p in frozen_paths}
    annotations = load_jsonl(annotation_path)
    if len(annotations) != 4013:
        raise ValueError('Unexpected DiDeMo test count')
    # The official evaluator strips query text. Normalize a copied dictionary
    # solely for comparison; original annotation bytes remain untouched.
    normalized = [dict(a, fig_desc=a['fig_desc'].strip()) for a in annotations]
    local, reference = load_jsonl(local_path), load_jsonl(reference_path)
    parser = official_parser()
    validation = {name: validate_rows(rows, normalized, parser)
                  for name, rows in [('local', local), ('reference', reference)]}
    if any(v['parser_and_iou_mismatch_count'] for v in validation.values()):
        raise ValueError(json.dumps(validation))
    scorer = load_scorer()
    local_summary = summarize(local, scorer)
    reference_scorer = ReleasedThreeTurnScorer(scorer)
    reference_summary = summarize(reference, reference_scorer)
    reference_summary['actual_verification_by_turn'] = {
        str(t): scorer.verification([(r['turns'][t-1]['answer'], r['turns'][t-1]['top1_is_gt'])
                                     for r in reference if len(r['turns']) >= t],
                                    sum(len(r['turns']) >= t for r in reference))
        for t in (1, 2, 3)
    }
    reference_summary['pre_terminal_saved_ranking_VR'] = scorer.official_helpers().rank_metrics([
        r['turns'][-1]['rank_before_refine'] if len(r['turns']) == 3
        and r['turns'][-1]['refine_applied'] else r['final_rank'] for r in reference
    ])
    snapshot = first_two_turns_snapshot(reference)
    snapshot_summary = summarize(snapshot, scorer)
    snapshot_summary['scope'] = 'Only saved first-two-turn fields from T=3; not a fresh T=2 inference or a controlled GPU ablation.'
    if local_summary['protocol']['primary'] != json.loads(primary_path.read_text())['primary']:
        raise ValueError('Recomputed local scores differ from frozen primary result')
    reference_result = json.loads(reference_result_path.read_text())
    roundoff = []
    released_metric_differences = []
    for key, saved_key in [('VR', 'final_retrieval'), ('VCMR', 'final_temporal')]:
        actual = reference_summary['protocol']['primary'][key]
        saved = reference_result[saved_key]
        if actual.keys() != saved.keys():
            raise ValueError('Unexpected released result schema')
        for metric, value in actual.items():
            if value == saved[metric]:
                continue
            delta = {'task': key, 'metric': metric, 'recomputed': value,
                     'released': saved[metric], 'delta': value-saved[metric]}
            if math.isclose(value, saved[metric], rel_tol=0, abs_tol=1e-12):
                roundoff.append(delta)
            else:
                released_metric_differences.append(delta)
    comparison = pair_comparison(local, reference)
    preflight = json.loads(preflight_path.read_text())
    duplicate_ids = set(preflight['mappings']['official_text_map_qid_overrides'])
    whitespace_ids = {j for j, a in enumerate(annotations) if a['fig_desc'] != a['fig_desc'].strip()}
    over_ids = {j for j, a in enumerate(annotations) if a['time'][1] > a['duration'] + 0.1}
    video_metadata = {v['video_id']: v for v in load_jsonl(video_meta_path)}
    long_ids = {j for j, a in enumerate(annotations)
                if video_metadata[a['video']]['source_duration_sec'] > 64}
    subgroup = {name: subgroup_summary(local, reference, ids) for name, ids in [
        ('duplicate_text_overrides', duplicate_ids),
        ('query_whitespace_normalization', whitespace_ids),
        ('annotation_end_exceeds_duration', over_ids),
        ('source_video_longer_than_64_seconds', long_ids),
        ('source_video_at_most_64_seconds', set(range(4013))-long_ids),
    ]}
    candidates = [(a, b) for a, b in zip(local, reference)
                  if b['final_iou_r1'] >= 0.7 and a['final_iou_r1'] < 0.7
                  and all(r['turns'][-1]['top1_is_gt']
                          and r['turns'][-1]['answer'] == 'matched'
                          and r['turns'][-1]['time_parse_ok'] for r in (a, b))]
    candidates.sort(key=lambda pair: (pair[0]['final_iou_r1']-pair[1]['final_iou_r1'], pair[0]['index']))
    cases = [{'index': a['index'], 'query': a['query'], 'video': a['gt_pos_doc_id'],
              'gt': [a['gt_start'], a['gt_end']],
              'local_span': [a['final_pred_start'], a['final_pred_end']],
              'reference_span': [b['final_pred_start'], b['final_pred_end']],
              'local_iou': a['final_iou_r1'], 'reference_iou': b['final_iou_r1']}
             for a, b in candidates[:3]]
    if before != {str(p.relative_to(ROOT)): digest(p) for p in frozen_paths}:
        raise ValueError('A frozen input changed during analysis')
    if any(name in sys.modules for name in ('torch', 'transformers', 'vllm')):
        raise ValueError('Offline analyzer must not import model runtimes')
    validation.update(original_sources_unchanged=True,
                      official_result_reproduced=not released_metric_differences,
                      official_result_absolute_tolerance=1e-12,
                      aggregation_roundoff=roundoff,
                      released_metric_differences=released_metric_differences)
    result = {
        'created_at': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'status': 'OFFLINE_MINIMUM_PASSED' if not released_metric_differences else 'RELEASED_RESULT_DIFFERENCE',
        'model_execution': False, 'gpu_access': False, 'network_access': False,
        'model_runtime_modules_imported': False,
        'source_sha256': before, 'analysis_script_sha256': digest(Path(__file__)),
        'reference_provenance': json.loads((reference_dir/'download_receipts.json').read_text()),
        'validation': validation, 'local': local_summary, 'reference': reference_summary,
        'reference_configuration': {
            'declared_max_turns': dict(Counter(r['max_turn'] for r in reference)),
            'actual_turn_counts': dict(Counter(len(r['turns']) for r in reference)),
            'local_declared_max_turns': dict(Counter(r['max_turn'] for r in local)),
            'maximum_turns_match': False,
            'soft_token_shapes': dict(Counter(str(t.get('rollout_shape')) for r in reference
                                               for t in r['turns'] if t.get('refine_applied'))),
        },
        'reference_first_two_turns_snapshot': snapshot_summary,
        'reference_third_turn_increment': {
            'VR': {k: reference_summary['protocol']['primary']['VR'][k]
                      - snapshot_summary['protocol']['primary']['VR'][k]
                   for k in ('R@1', 'R@5', 'R@10', 'R@100')},
            'VCMR_hit_counts': {str(t): sum(r['final_iou_r1'] >= t for r in reference)
                                - sum(r['final_iou_r1'] >= t for r in snapshot)
                                for t in THRESHOLDS},
        },
        'comparison_to_reference_first_two_turns_snapshot': pair_comparison(local, snapshot),
        'comparison': comparison, 'data_subgroups': subgroup,
        'deterministic_examples': cases,
        'case_selection': 'Both accept and localize GT; reference hits IoU0.7, local misses; largest IoU loss, then index; first three.',
        'limitations': [
            'No new inference or visual inspection; labels define video correctness.',
            'Published predictions do not identify exact paper VER aggregation.',
            'Author release is T=3 whereas local/paper T=2; saved-prefix statistics do not establish an actual T=2 counterfactual.',
            'Reference backend and exact original pixel/time-axis settings are not established.',
            'Paired span/subgroup differences do not identify a causal backend or preprocessing factor.',
            'Primary last-verification protocol remains unchanged.',
        ],
    }
    args.output.mkdir(parents=True)
    (args.output/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    fields = ['index', 'query', 'gt_video', 'gt_start', 'gt_end', 'annotation_duration',
              'source_duration', 'local_orig_rank', 'reference_orig_rank',
              'local_final_rank', 'reference_final_rank', 'local_checked_video',
              'reference_checked_video', 'local_answer', 'reference_answer',
              'local_start', 'local_end', 'reference_start', 'reference_end',
              'local_iou', 'reference_iou', 'local_failure_07', 'reference_failure_07',
              'duplicate_text_override', 'gt_end_exceeds_duration', 'source_longer_than_64s']
    with (args.output/'per_example.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for a, b, annotation in zip(local, reference, annotations):
            j = a['index']
            writer.writerow(dict(zip(fields, [
                j, a['query'], a['gt_pos_doc_id'], a['gt_start'], a['gt_end'],
                annotation['duration'], video_metadata[a['gt_pos_doc_id']]['source_duration_sec'],
                a['orig_rank'], b['orig_rank'], a['final_rank'], b['final_rank'],
                a['turns'][-1]['top1_doc_id'], b['turns'][-1]['top1_doc_id'],
                a['final_answer'], b['final_answer'], a['final_pred_start'], a['final_pred_end'],
                b['final_pred_start'], b['final_pred_end'], a['final_iou_r1'], b['final_iou_r1'],
                failure_category(a, 0.7), failure_category(b, 0.7),
                j in duplicate_ids, j in over_ids, j in long_ids,
            ])))
    print(json.dumps({
        'output': str(args.output), 'status': result['status'], 'validation': validation,
        'local_primary': local_summary['protocol']['primary'],
        'reference_primary': reference_summary['protocol']['primary'],
        'local_first_VER': local_summary['protocol']['supplementary']['first_turn_VER'],
        'reference_first_VER': reference_summary['protocol']['supplementary']['first_turn_VER'],
        'reference_configuration': result['reference_configuration'],
        'reference_first_two_turns_snapshot_primary': snapshot_summary['protocol']['primary'],
        'reference_third_turn_increment': result['reference_third_turn_increment'],
        'same_counts': comparison['same_counts'],
        'paired_differences': {k: {a:b for a,b in v.items() if not a.endswith('_indices')}
                               for k,v in comparison['vcmr_paired_differences'].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
