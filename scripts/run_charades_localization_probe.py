#!/usr/bin/env python3
"""Single-GPU supervised experiment runner and paired exploratory summary."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np

from charades_localization_probe_worker import PLAN, verify_plan
from inspect_charades_localization_cases import ROOT, jsonl, save, sha
from run_charades_full import RUNTIME, load_environment, wait_for_gpu_release

REPORT = ROOT / 'reports/49_charades_localization_probe_result.json'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def summarize(output):
    records = jsonl(output / 'paired_predictions.jsonl')
    assets = json.loads((ROOT / 'reports/47_charades_localization_probe_prepare.json').read_text())
    assert [r['index'] for r in records] == [r['index'] for r in assets['selection']]
    assert len({r['video_id'] for r in records}) == len(records)
    assert not ({r['video_id'] for r in records if r['group'] == 'dev'} &
                {r['video_id'] for r in records if r['group'] == 'holdout'})
    audits = jsonl(output / 'generation_audit.jsonl')
    assert len(audits) == sum(c['executed'] for r in records for c in r['conditions'].values())
    cohorts, examples = {}, []
    for group in ['dev', 'holdout', 'all']:
        selected = [r for r in records if group == 'all' or r['group'] == group]
        cohort = dict(N=len(selected), baseline_matched=sum(r['frozen_verification'] == 'matched' for r in selected), conditions={})
        for arm in ['baseline', 'ordinary', 'evidence']:
            values = np.array([r['iou'][arm] for r in selected], dtype=float)
            generated = [r['conditions'][arm] for r in selected if r['conditions'][arm]['executed']]
            cohort['conditions'][arm] = dict(
                mIoU=float(values.mean()),
                recall_at_iou={str(t): float(np.mean(values >= t)) for t in [.3, .5, .7]},
                executed_calls=len(generated),
                fallback_count=sum(r['conditions'][arm].get('fallback', False) for r in selected),
                generation_seconds=sum(r['elapsed_seconds'] for r in generated),
                input_tokens=sum(r['input_tokens'] for r in generated),
                output_tokens=sum(r['output_tokens'] for r in generated))
        cohort['comparisons'] = {}
        for left, right in [('ordinary', 'baseline'), ('evidence', 'baseline'), ('evidence', 'ordinary')]:
            a, b = np.array([r['iou'][left] for r in selected]), np.array([r['iou'][right] for r in selected])
            delta = a - b
            rng = np.random.default_rng(4207)
            samples = rng.integers(0, len(delta), size=(5000, len(delta)))
            bootstrap = delta[samples].mean(axis=1)
            cohort['comparisons'][left + '_vs_' + right] = dict(
                mean_iou_delta=float(delta.mean()),
                bootstrap_95_interval=np.quantile(bootstrap, [.025, .975]).tolist(),
                iou_improved=int(np.sum(delta > 1e-9)), iou_worsened=int(np.sum(delta < -1e-9)),
                iou_unchanged=int(np.sum(np.abs(delta) <= 1e-9)),
                point7_gained=int(np.sum((a >= .7) & (b < .7))),
                point7_lost=int(np.sum((a < .7) & (b >= .7))))
        cohorts[group] = cohort
    for row in records:
        examples.append({k: row[k] for k in ['index', 'group', 'video_id', 'query', 'gt', 'frozen_verification', 'iou']})
    holdout = cohorts['holdout']
    comparison = holdout['comparisons']['evidence_vs_ordinary']
    advantage = (comparison['mean_iou_delta'] > 0 and comparison['bootstrap_95_interval'][0] > 0
                 and holdout['conditions']['evidence']['recall_at_iou']['0.7'] >= holdout['conditions']['ordinary']['recall_at_iou']['0.7'])
    report = dict(status='EXPLORATORY_PROBE_COMPLETE', cohorts=cohorts,
                  interpretation='Evidence prompt has a positive paired holdout mIoU interval with no lower point7 recall; still exploratory.' if advantage else 'No clear holdout evidence that the evidence-guided prompt beats an ordinary second look; do not promote automatically to full test.',
                  automatic_full_benchmark_started=False, paired_examples=examples,
                  selection_scope=assets['sampling_scope'], evaluation_scope=assets['evaluation_scope'],
                  holdout_limit=assets['holdout_limit'],
                  runtime_input_audit=dict(all_calls_covered=True, calls=len(audits),
                      per_case_visual_signature_consistent=all(
                          len({json.dumps({k: a[k] for k in ['time_labels','video_shapes','pixel_tensor_sha256']}, sort_keys=True)
                               for a in audits if a['index'] == row['index']}) == 1 for row in records)),
                  predictions_sha256=sha(output / 'paired_predictions.jsonl'),
                  limitations=['Training-derived correct-top1 probe is not a full VCMR/VER/VR benchmark.',
                               'Original checkpoint may already have trained on these videos.',
                               f'Holdout has only {cohorts["holdout"]["N"]} video-disjoint examples; bootstrap intervals are exploratory.',
                               'Condition C requests evidence but does not establish that generated rationales are true.',
                               'Identical visual and maximum-generation budgets; actual text/output token costs differ and are recorded.',
                               'No training, new frames, test-label edits or MD documents.'])
    assert report['runtime_input_audit']['per_case_visual_signature_consistent']
    save(output / 'comparison.json', report)
    return report


def execute(gpu):
    if REPORT.exists():
        raise ValueError('Preserve previous GPU attempt; do not overwrite its report')
    plan = verify_plan()
    output = Path(plan['output'])
    with (ROOT / 'cache/charades_full.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        query = ['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid,memory.free,memory.used,utilization.gpu', '--format=csv,noheader,nounits']
        uuid, free, used, util = subprocess.check_output(query, text=True).strip().split(', ')
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits'], text=True)
        if int(free) < 70000 or int(used) >= 1000 or int(util) >= 5 or uuid in apps:
            raise RuntimeError('Requested GPU is not idle; no running task will be interrupted')
        env = load_environment()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   PROBE_GPU_UUID=uuid, VLLM_WORKER_MULTIPROC_METHOD='spawn')
        # Verify UUID in a short subprocess before model loading.
        seen = subprocess.check_output([str(RUNTIME), '-B', '-c', 'import torch;print(torch.cuda.get_device_properties(0).uuid)'], env=env, text=True).strip()
        if seen.removeprefix('GPU-') != uuid.removeprefix('GPU-'):
            raise ValueError('Torch/NVML physical GPU mismatch')
        argv = [str(RUNTIME), '-B', str(ROOT / 'scripts/charades_localization_probe_worker.py'), '--run']
        state = dict(status='running', started_at=now(), output=str(output), gpu_index=gpu, gpu_uuid=uuid,
                     command=argv, plan_sha256=sha(PLAN), runner_sha256=sha(Path(__file__)),
                     conditions=['baseline', 'ordinary', 'evidence'], records=plan['expected_records'])
        def persist():
            save(REPORT, state); save(output / 'run_manifest.json', state)
        persist()
        process = None
        started, peak = time.monotonic(), 0
        try:
            with (output / 'inference.log').open('x') as log, (output / 'gpu_memory.csv').open('x') as memory:
                memory.write('elapsed_seconds,used_mib,utilization_percent\n')
                process = subprocess.Popen(argv, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    used, util = subprocess.check_output(['nvidia-smi', f'--id={uuid}', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
                    peak = max(peak, int(used))
                    memory.write(f'{time.monotonic() - started:.3f},{used},{util}\n'); memory.flush()
                    time.sleep(2)
            state.update(returncode=process.returncode, elapsed_seconds=round(time.monotonic() - started, 2), peak_gpu_used_mib=peak)
            state['resource_release'] = wait_for_gpu_release(query, uuid)
            if process.returncode:
                raise RuntimeError('Worker failed; inspect independent inference log')
            if not state['resource_release']['passed']:
                raise RuntimeError('GPU release check failed')
            verify_plan()
            state['comparison'] = summarize(output)
            state.update(status='complete', completed_at=now())
            persist()
            print(json.dumps(dict(status='complete', output=str(output), report=str(REPORT),
                                 holdout=state['comparison']['cohorts']['holdout'],
                                 interpretation=state['comparison']['interpretation']), ensure_ascii=False), flush=True)
        except BaseException as error:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
            state.update(status='failed', error=str(error), completed_at=now())
            if 'resource_release' not in state:
                state['resource_release'] = wait_for_gpu_release(query, uuid)
            persist()
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    execute(args.gpu)
