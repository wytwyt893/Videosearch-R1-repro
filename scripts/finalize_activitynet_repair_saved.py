#!/usr/bin/env python3
"""Finish the saved full run offline after the erroneous smoke-only gate.

No model construction, CUDA initialization, prediction rewriting or inference.
The original failed manifest, log and launcher exit code remain available.
"""
from __future__ import annotations
import datetime
import json
from pathlib import Path
import re
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np

from activitynet_repair_time import FIELD
from compare_activitynet_full_to_paper import compare
from run_activitynet_full import ROOT, RUNTIME, BASE, write_json
from run_activitynet_repair_smoke import DATA, digest, validate_ver_coverage, verify_inputs

OUTPUT = ROOT/'outputs/activitynet_repair_full_20261030_004132_840892'
REPORT = ROOT/'reports/36_activitynet_repair_full.json'
RECOVERY = ROOT/'reports/36a_activitynet_full_postprocess_recovery.json'
TZ = ZoneInfo('Asia/Shanghai')


def main():
    start = time.monotonic()
    if REPORT.exists() or RECOVERY.exists():
        raise ValueError('Preserve completed recovery; do not repeat it')
    failure_path = OUTPUT/'run_manifest.postprocess_failed_20261031.json'
    failure = json.loads(failure_path.read_text())
    if (failure['status'] != 'failed' or failure.get('error') != 'Unparseable VER answers in smoke'
            or failure.get('returncode') != 0 or not failure.get('resource_release_passed')):
        raise ValueError('This offline recovery is limited to the known postprocessing gate failure')
    verify_inputs()
    summary = json.loads((OUTPUT/f'{BASE}.json').read_text())
    predictions = OUTPUT/f'{BASE}.jsonl'
    before_sha = digest(predictions)
    primary = OUTPUT/f'{BASE}.primary.result.json'
    original_score = json.loads(primary.read_text())
    if original_score['source_sha256'] != before_sha:
        raise ValueError('Saved primary report does not describe these predictions')
    rows = [json.loads(x) for x in predictions.read_text().splitlines() if x.strip()]
    if ([r['index'] for r in rows] != list(range(16966)) or summary['rows_valid'] != 16966
            or any(summary['skip'].values()) or summary['video_meta_stats']['meta_payload_misses']):
        raise ValueError('Incomplete predictions or input metadata')
    rechecked = OUTPUT/f'{BASE}.primary.rechecked.result.json'
    if rechecked.exists():
        raise ValueError('Preserve prior re-scoring output')
    with (OUTPUT/'offline_recheck_score.log').open('w') as log:
        subprocess.run([str(RUNTIME),'-B',str(ROOT/'scripts/score_activitynet_protocol.py'),
                        str(predictions),str(rechecked),'--expected-count','16966'],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    rescored = json.loads(rechecked.read_text())
    if rescored != original_score:
        raise ValueError('Fresh annotation-validated re-scoring disagrees with the saved primary result')
    validate_ver_coverage(rescored['primary']['VER'], smoke=False)
    expected_arguments = json.loads((DATA/'evaluation_plans.json').read_text())['plans']['source_time_repaired_subset_full']['arguments']
    if any(failure['arguments'][k] != v for k,v in expected_arguments.items()
           if k not in {'--output_json','--output_jsonl'}):
        raise ValueError('The run did not use the frozen full-evaluation plan')
    if summary['settings']['max_turn'] != 2 or summary['settings']['refine_rollout_depth'] != 8:
        raise ValueError('T/N settings differ from the qualified plan')
    refined = [t for r in rows for t in r['turns'] if t.get('refine_applied')]
    if not refined or any(t.get('rollout_depth_actual')!=8 or t.get('rollout_shape')!=[8,2048] for t in refined):
        raise ValueError('Saved SQR dimensions differ from N=8')
    cohort = json.loads((DATA/'cohort_manifest.json').read_text())
    metadata = {r['video_id']:r for r in map(json.loads,Path(failure['arguments']['--video_meta_path']).read_text().splitlines())}
    audit_path = OUTPUT/'input_audit.jsonl'
    inputs = [json.loads(x) for x in audit_path.read_text().splitlines() if x.strip()]
    checked = [t['top1_doc_id'] for r in rows for t in r['turns']]
    if [i['video_id'] for i in inputs] != checked or len(inputs)!=summary['video_meta_stats']['meta_payload_hits']:
        raise ValueError('Actual input audit differs from saved trajectories')
    repaired_inputs = []
    for item in inputs:
        row = metadata[item['video_id']]
        times = row[FIELD]
        expected = [(times[i]+times[i+1])/2 for i in range(0,len(times),2)]
        observed = item['model_timestamp_labels_seconds']
        if (item['do_sample_frames'] is not False or len(observed)!=len(expected) or
                any(abs(a-b)>1e-6 for a,b in zip(observed,expected))):
            raise ValueError('Input time labels do not match frozen source timestamps')
        if item['video_id'] in cohort['accepted_source_records']:
            record = cohort['accepted_source_records'][item['video_id']]
            if item['resolved_path'] != str((ROOT/record['npy_path']).resolve()):
                raise ValueError('Actual repaired-video input resolved to the wrong file')
            repaired_inputs.append(item['video_id'])
    log_text = (OUTPUT/'inference.log').read_text(errors='replace')

    def marker(text):
        for line in log_text.splitlines():
            if text in line:
                match = re.match(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)',line)
                if match:
                    return datetime.datetime.fromisoformat(match.group(1).replace(',','.')).replace(tzinfo=TZ)
        raise ValueError(f'Missing official completion marker: {text}')

    loop_start, loop_end = marker('Streaming detail jsonl:'), marker('Saved summary json:')
    timing = {'process_seconds':failure['elapsed_seconds'],
              'process_hours':failure['elapsed_seconds']/3600,
              'official_loop_started_at':loop_start.isoformat(),
              'official_loop_completed_at':loop_end.isoformat(),
              'official_loop_seconds':(loop_end-loop_start).total_seconds(),
              'official_loop_mean_seconds_per_query':(loop_end-loop_start).total_seconds()/16966,
              'measured_full_run':True, 'per_query_timing_available':False,
              'scope':'Process elapsed_seconds comes from the original monotonic timer; loop times come from original log markers. Per-query events were not saved before the postprocessing failure and are not fabricated.'}
    write_json(OUTPUT/'timing.json',timing)
    boundary = {str(t):[r['index'] for r in rows if (float(r['final_iou_r1'])>=t)!=bool(np.float32(r['final_iou_r1'])>=np.float32(t))] for t in (.3,.5,.7)}
    precision = {'runtime_summary_uses_float32':True,
                 'jsonl_official_offline_scorer_is_primary':True,
                 'threshold_disagreement_indices':boundary,
                 'policy':'Preserve saved double-precision IoUs and official offline scoring; do not round predictions or change thresholds.'}
    invalid = [r['index'] for r in rows if r['turns'][-1]['answer'] not in {'matched','not_matched'}]
    recovery = {'created_at':datetime.datetime.now(TZ).isoformat(),'status':'OFFLINE_FINALIZATION_VALIDATED',
                'gpu_model_execution':False,'prediction_rows_modified':False,
                'original_launcher_exitcode':1,'inference_returncode':0,
                'original_error':failure['error'],'original_failure_manifest':str(failure_path),
                'original_failure_manifest_sha256':digest(failure_path),
                'original_failed_launcher_source':str(ROOT/'cache/activitynet_repair_v1/full_launcher_before_coverage_fix_20261101.py'),
                'original_launcher_source_sha256':failure['launcher_sha256'],
                'prediction_sha256':before_sha,'full_examples':len(rows),'invalid_last_ver_indices':invalid,
                'ver_valid_denominator':rescored['primary']['VER']['N'],
                'ver_all_example_denominator':16966,
                'annotation_validated_rescore_exactly_matches_original':True,
                'rechecked_score':str(rechecked),'rechecked_score_sha256':digest(rechecked),
                'actual_inputs_verified':len(inputs),'refine_applications_verified':len(refined),
                'distinct_repaired_videos_actually_used':len(set(repaired_inputs)),
                'metric_precision_observation':precision,
                'offline_validation_seconds':round(time.monotonic()-start,2)}
    if digest(predictions)!=before_sha:
        raise ValueError('Predictions changed during offline finalization')
    write_json(RECOVERY,recovery)
    manifest = dict(failure)
    manifest.pop('error',None)
    manifest.update(status='complete',predictions=str(predictions),prediction_sha256=before_sha,
                    primary_result=str(primary),primary_result_sha256=digest(primary),
                    primary=rescored['primary'],timing=timing,skip=summary['skip'],
                    video_meta_stats=summary['video_meta_stats'],refine=summary['refine'],
                    input_audit=str(audit_path),input_audit_sha256=digest(audit_path),
                    actual_repaired_video_inputs=repaired_inputs,
                    input_time_labels_and_no_resampling_verified=True,
                    inference_completed_at=loop_end.isoformat(),
                    postprocessing_recovery={'report':str(RECOVERY),'sha256':digest(RECOVERY),
                                             'original_launcher_exitcode':1,'offline_finalize_exitcode':0},
                    metric_precision_observation=precision)
    write_json(REPORT,manifest)
    write_json(OUTPUT/'run_manifest.json',manifest)
    comparison = compare(REPORT)
    manifest.update(paper_comparison_outputs=comparison,
                    offline_finalizer_sha256=digest(Path(__file__)),
                    comparison_script_sha256=digest(ROOT/'scripts/compare_activitynet_full_to_paper.py'))
    write_json(REPORT,manifest)
    write_json(OUTPUT/'run_manifest.json',manifest)
    (ROOT/'cache/activitynet_repair_v1/full_20261030.offline_finalize.exitcode').write_text('0\n')
    print(json.dumps({'status':'complete','predictions':len(rows),'primary':manifest['primary'],
                      'timing':timing,'recovery_report':str(RECOVERY),
                      'paper_comparison_outputs':comparison},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
