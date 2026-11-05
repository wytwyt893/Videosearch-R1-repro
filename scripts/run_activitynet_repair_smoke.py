#!/usr/bin/env python3
"""Run a guarded ten-question smoke on a frozen repaired ActivityNet view."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from zoneinfo import ZoneInfo

from activitynet_repair_time import CLOCK_HZ, FIELD
from run_activitynet_full import (ROOT, RUNTIME, BASE, load_environment, verify_preflight,
                                 wait_for_gpu_release, analyze_timing, write_json)

DATA = ROOT/'data/activitynet_repair_v1'
INTEGRITY = ROOT/'reports/34_activitynet_repair_integrity.json'
DEFAULT_REPORT = ROOT/'reports/35_activitynet_repair_smoke.json'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def validate_ver_coverage(metric, *, smoke):
    """Smoke exercises parsing strictly; full evaluation retains invalid answers."""
    valid, invalid, expected = metric['N'], metric['invalid_answers'], metric['N_expected']
    if expected < 1 or valid < 1 or invalid < 0 or valid+invalid != expected:
        raise ValueError('Inconsistent or entirely invalid VER counts')
    if abs(metric['coverage']-valid/expected) > 1e-12:
        raise ValueError('VER coverage differs from its explicit denominator')
    if smoke and invalid:
        raise ValueError('Unparseable VER answers in smoke')


def verify_inputs():
    verify_preflight()
    integrity = json.loads(INTEGRITY.read_text())
    if not integrity.get('passed') or integrity['status'] != 'FULL_REPAIR_INTEGRITY_PASSED':
        raise ValueError('A passing fresh full integrity check is required')
    for key in ('file_states', 'view_links'):
        if digest(integrity[key]) != integrity[key+'_sha256']:
            raise ValueError('Integrity evidence changed')
    states = json.loads(Path(integrity['file_states']).read_text())
    for rel, previous in states.items():
        current = (ROOT/rel).stat()
        if (current.st_size, current.st_mtime_ns) != (previous['bytes'], previous['mtime_ns']):
            raise ValueError(f'Input changed after full hashing: {rel}')
    for link in json.loads(Path(integrity['view_links']).read_text()):
        path = ROOT/link['path']
        if not path.is_symlink() or str(path.resolve(strict=True)) != link['target']:
            raise ValueError(f'Input symlink changed: {path}')
    plans = json.loads((DATA/'evaluation_plans.json').read_text())
    if (plans['manifest_sha256'] != digest(DATA/'cohort_manifest.json') or
            plans['cpu_preflight_sha256'] != digest(ROOT/'reports/32_activitynet_repair_cpu_preflight.json')):
        raise ValueError('Plan is not bound to the frozen cohort and CPU preflight')
    return integrity, plans


def verify_qualified_smoke(variant, selected, plans):
    path = ROOT/'reports/35_activitynet_repair_smoke.json'
    smoke = json.loads(path.read_text())
    if (smoke.get('status') != 'complete' or smoke.get('variant') != variant or
            smoke.get('expected_examples') != 10 or not smoke.get('resource_release_passed') or
            not smoke.get('input_time_labels_and_no_resampling_verified') or
            smoke.get('manifest_sha256') != plans['manifest_sha256'] or
            smoke.get('integrity_report_sha256') != digest(INTEGRITY)):
        raise ValueError('A passing smoke bound to these repaired inputs is required')
    for key, hash_key in (('predictions','prediction_sha256'),
                          ('primary_result','primary_result_sha256'), ('input_audit','input_audit_sha256')):
        if digest(smoke[key]) != smoke[hash_key]:
            raise ValueError('Qualified smoke evidence changed')
    snapshot = ROOT/'cache/activitynet_repair_v1/smoke_launcher_qualified_20261030.py'
    if digest(snapshot) != smoke['launcher_sha256']:
        raise ValueError('Qualified smoke launcher snapshot changed')
    excluded = {'--max_samples', '--output_json', '--output_jsonl'}
    if (set(selected['arguments']) != set(smoke['arguments']) or
            any(v != smoke['arguments'][k] for k, v in selected['arguments'].items() if k not in excluded) or
            selected['argv'][:4] != smoke['command'][:4]):
        raise ValueError('Full inference settings differ from the successful smoke')
    return {'report': str(path), 'sha256': digest(path), 'qualified_variant': variant,
            'inference_differences': ['max_samples 10 -> 0 (all 16966); fresh output paths']}


def execute(gpu, variant, report, full=False):
    if Path(os.sys.prefix).resolve() != ROOT/'envs/videosearchr1':
        raise ValueError('Use the project interpreter')
    if report.exists():
        raise ValueError('Preserve previous result; choose a new --report path')
    with (ROOT/'cache/activitynet_full.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        integrity, plans = verify_inputs()
        selected = plans['plans'][variant+('_full' if full else '_smoke')]
        qualification = verify_qualified_smoke(variant, selected, plans) if full else None
        expected_count = 16966 if full else 10
        kind = 'full' if full else 'smoke'
        env = load_environment()
        query = ['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid,memory.free,memory.used,utilization.gpu',
                 '--format=csv,noheader,nounits']
        values = subprocess.check_output(query, text=True).strip().split(', ')
        uuid, free, used, utilization = values
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                       '--format=csv,noheader,nounits'], text=True)
        if int(free)<70000 or int(used)>=1000 or int(utilization)>=5 or uuid in apps:
            raise RuntimeError(f'GPU {gpu} is not idle: {values}')
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID')
        cuda_uuid = subprocess.check_output([str(RUNTIME), '-B', '-c',
            'import torch; print(torch.cuda.get_device_properties(0).uuid)'], env=env, text=True).strip()
        if cuda_uuid.removeprefix('GPU-') != uuid.removeprefix('GPU-'):
            raise ValueError('CUDA and NVML refer to different physical GPUs')
        stamp = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        output = ROOT/f'outputs/activitynet_repair_{kind}_{stamp}'
        output.mkdir()
        arguments = dict(selected['arguments'])
        arguments.update({'--output_json': str(output/f'{BASE}.json'),
                          '--output_jsonl': str(output/f'{BASE}.jsonl')})
        if arguments['--max_samples'] != ('0' if full else '10') or arguments['--max_turn'] != '2' or arguments['--refine_token_count'] != '8':
            raise ValueError('Unexpected smoke protocol')
        argv = list(selected['argv'][:4]) + [x for pair in arguments.items() for x in pair]
        audit_path = output/'input_audit.jsonl'
        env.update(VIDEOSEARCH_REPAIR_INPUT_AUDIT=str(audit_path), TZ='Asia/Shanghai')
        manifest = {'status': 'running', 'created_at': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                    'run_kind': 'repaired_full' if full else 'repaired_final_smoke', 'variant': variant, 'model_execution': True,
                    'output_dir': str(output), 'gpu_index': gpu, 'gpu_uuid': uuid, 'cuda_uuid': cuda_uuid,
                    'gpu_name': subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=name',
                        '--format=csv,noheader'], text=True).strip(),
                    'arguments': arguments, 'command': argv, 'expected_examples': expected_count,
                    'expected_corpus': 4866, 'original_indices': list(range(expected_count)),
                    'integrity_report_sha256': digest(INTEGRITY), 'manifest_sha256': plans['manifest_sha256'],
                    'ver_turn': 'last_verified', 'terminal_refinement': 'keep_official',
                    'launcher_sha256': digest(Path(__file__)), 'full_evaluation_started': full,
                    'smoke_qualification': qualification,
                    'paper_comparison': 'Declared repaired-input variant; source sampling/time transport differ from released assets.'}
        write_json(output/'run_manifest.json', manifest)
        print(f'[{kind}] GPU {gpu}, {variant}, output {output}', flush=True)
        start, epoch, peak = time.monotonic(), time.time(), 0
        events, position, pending = [], 0, ''
        predictions = output/f'{BASE}.jsonl'

        def observe():
            nonlocal position, pending
            if not predictions.exists():
                return
            with predictions.open() as handle:
                handle.seek(position)
                pending += handle.read()
                position = handle.tell()
            lines = pending.split('\n')
            pending = lines.pop()
            for line in lines:
                if line.strip():
                    row = json.loads(line)
                    events.append({'index': row['index'], 'elapsed_seconds': time.monotonic()-start,
                                   'turns': len(row['turns']), 'refines': row['refine_applied_turns']})
                    print(f'[{kind}] {len(events)}/{expected_count} index={row["index"]}', flush=True)

        process = None
        try:
            with (output/'inference.log').open('w') as log, (output/'gpu_memory.csv').open('w') as monitor:
                monitor.write('elapsed_seconds,used_mib,utilization_percent\n')
                process = subprocess.Popen(argv, cwd=ROOT/'upstream', env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                last_poll = -10
                while process.poll() is None:
                    now = time.monotonic()-start
                    if now-last_poll>=2:
                        current = subprocess.check_output(['nvidia-smi', f'--id={uuid}',
                            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
                        used_now, util = current.strip().split(', ')
                        peak = max(peak, int(used_now))
                        monitor.write(f'{now:.2f},{used_now},{util}\n')
                        monitor.flush()
                        last_poll = now
                    observe()
                    time.sleep(.1)
                observe()
            write_json(output/'timing_events.json', {'process_start_epoch': epoch, 'events': events})
            manifest.update(returncode=process.returncode, elapsed_seconds=round(time.monotonic()-start, 2),
                            peak_gpu_used_mib=peak)
            if process.returncode:
                raise RuntimeError('Inference failed; inspect inference.log')
            summary = json.loads((output/f'{BASE}.json').read_text())
            rows = [json.loads(x) for x in predictions.read_text().splitlines() if x.strip()]
            if (summary['rows_valid'] != expected_count or any(summary['skip'].values()) or
                    summary['video_meta_stats']['meta_payload_misses'] or
                    [r['index'] for r in rows] != list(range(expected_count)) or
                    [e['index'] for e in events] != list(range(expected_count))):
                raise ValueError('Missing, skipped or reordered predictions/metadata')
            refined = [t for r in rows for t in r['turns'] if t.get('refine_applied')]
            if not refined or any(t.get('rollout_depth_actual')!=8 or t.get('rollout_shape')!=[8,2048] for t in refined):
                raise ValueError('Smoke must exercise the eight-token SQR path')
            frozen = json.loads((DATA/'cohort_manifest.json').read_text())
            meta_file = Path(arguments['--video_meta_path'])
            metadata = {r['video_id']: r for r in map(json.loads, meta_file.read_text().splitlines())}
            inputs = [json.loads(x) for x in audit_path.read_text().splitlines() if x.strip()]
            checked_videos = [t['top1_doc_id'] for r in rows for t in r['turns']]
            if [i['video_id'] for i in inputs] != checked_videos:
                raise ValueError('Input audit does not match actual checked-video trajectories')
            repaired_inputs = []
            for item in inputs:
                row = metadata[item['video_id']]
                values = row[FIELD] if frozen['variants'][variant]['source_timestamps'] else list(range(row['output_num_frames']))
                expected = [(values[i]+values[i+1])/2 for i in range(0,len(values),2)]
                observed = item['model_timestamp_labels_seconds']
                if item['do_sample_frames'] is not False or len(observed)!=len(expected) or any(
                        abs(a-b)>1e-6 for a,b in zip(observed,expected)):
                    raise ValueError('Actual model-input time labels differ from the chosen data view')
                if item['video_id'] in frozen['accepted_source_records']:
                    source = frozen['accepted_source_records'][item['video_id']]
                    if frozen['variants'][variant]['repaired_pixels'] and item['resolved_path'] != str((ROOT/source['npy_path']).resolve()):
                        raise ValueError('Repaired sample did not resolve to its new NPY')
                    repaired_inputs.append(item['video_id'])
            if frozen['variants'][variant]['repaired_pixels'] and not repaired_inputs:
                raise ValueError('Smoke did not actually exercise any repaired pixels')
            with (output/'report.log').open('w') as log:
                subprocess.run(['bash', str(ROOT/'upstream/scripts/inference/report.bash'), str(predictions)],
                               env=env, cwd=ROOT/'upstream', stdout=log, stderr=subprocess.STDOUT, check=True)
            primary = output/f'{BASE}.primary.result.json'
            with (output/'primary_score.log').open('w') as log:
                subprocess.run([str(RUNTIME), '-B', str(ROOT/'scripts/score_activitynet_protocol.py'),
                                str(predictions), str(primary), '--expected-count', str(expected_count)],
                               env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            scored = json.loads(primary.read_text())
            validate_ver_coverage(scored['primary']['VER'], smoke=not full)
            timing = analyze_timing((output/'inference.log').read_text(errors='replace'), events, epoch, manifest['elapsed_seconds'])
            if full:
                timing['measured_full_run'] = True
            else:
                timing['estimate_caveat'] = 'Ten fixed first queries; same repaired view and GPU only. Not a confidence interval or measured full-run duration.'
            write_json(output/'timing.json', timing)
            verify_inputs()
            manifest.update(status='complete', predictions=str(predictions), prediction_sha256=digest(predictions),
                            primary_result=str(primary), primary_result_sha256=digest(primary),
                            primary=scored['primary'], timing=timing, skip=summary['skip'],
                            video_meta_stats=summary['video_meta_stats'], refine=summary['refine'],
                            input_audit=str(audit_path), input_audit_sha256=digest(audit_path),
                            actual_repaired_video_inputs=repaired_inputs,
                            input_time_labels_and_no_resampling_verified=True)
        except BaseException as error:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            manifest.update(status='failed', error=str(error))
            raise
        finally:
            release = wait_for_gpu_release(query, uuid)
            manifest.update(resource_check=release, resource_release_passed=release['passed'])
            if manifest['status']=='complete' and not release['passed']:
                manifest['status']='resource_check_failed'
            write_json(output/'run_manifest.json', manifest)
        if manifest['status'] != 'complete':
            raise RuntimeError('Smoke did not pass all checks')
        write_json(report, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--variant', choices=('original_subset','pixels_repaired_subset','source_time_repaired_subset'),
                        default='source_time_repaired_subset')
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error('GPU index must be nonnegative')
    execute(args.gpu, args.variant, args.report)


if __name__ == '__main__':
    main()
