#!/usr/bin/env python3
"""Full Charades entry; default is a CPU argument dry run, never inference.

Actual inference requires --execute --gpu N and a passing bound CPU preflight.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'envs/videosearchr1/bin/python'
ENTRY = ROOT / 'upstream/scripts/inference/inference.bash'
PROFILE = ROOT / 'configs/charades_stage2_eval.bash'
BASE = 'external_verified_test_temporal_grounding_VideoSearchR1__charades-stage2'
COMMIT = '69234fd9783b7235164d9f92b8903d9781dc284e'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def load_environment():
    # Capture internally only: inherited secrets are never written to manifests.
    result = subprocess.run(
        ['bash', '--noprofile', '--norc', '-c',
         'set -e; source "$1"; source "$2"; env -0', 'charades-profile',
         str(ROOT / 'scripts/env.sh'), str(PROFILE)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    env = dict(item.split('=', 1) for item in result.stdout.decode().split('\0') if item)
    env.update(PYTHON_BIN=str(RUNTIME), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_TOKEN='', HUGGING_FACE_HUB_TOKEN='',
               VIDEOSEARCH_REPO_ROOT=str(ROOT / 'upstream'),
               VIDEOSEARCH_TMPDIR=str(ROOT / 'cache/tmp'), CUDA_VISIBLE_DEVICES='', TZ='Asia/Shanghai')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    return env


def command(env):
    return ['bash', str(ENTRY), 'charades', '--checkpoint', env['VIDEOSEARCH_STAGE2_CHECKPOINT']]


def check_source():
    commit = subprocess.check_output(['git', '-C', str(ROOT / 'upstream'), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT / 'upstream'), 'status', '--porcelain'], text=True).strip()
    if commit != COMMIT or dirty:
        raise ValueError('Official source changed; repeat the CPU preflight')


def dry_run(smoke=False):
    check_source()
    env = load_environment()
    if smoke:
        env['MAX_SAMPLES'] = '10'
    with tempfile.TemporaryDirectory(prefix='charades_full_entry_', dir=ROOT / 'cache/tmp') as temporary:
        temp = Path(temporary)
        recorder = temp / 'record_python'
        recorder.write_text(
            f'#!{sys.executable}\nimport json,os,pathlib,sys\n'
            'pathlib.Path(os.environ["CHARADES_CAPTURE"]).write_text(json.dumps('
            '{"argv":sys.argv[1:],"offline":os.environ.get("HF_HUB_OFFLINE"),'
            '"ver_turn":os.environ.get("VIDEOSEARCH_VER_TURN")}))\n'
        )
        recorder.chmod(0o700)
        env.update(PYTHON_BIN=str(recorder), CHARADES_CAPTURE=str(temp / 'args.json'),
                   LOGS_DIR=str(temp / 'logs'), VIDEOSEARCH_OUTPUT_ROOT=str(temp / 'output'), EVAL_GPUS='0')
        result = subprocess.run(command(env), env=env, cwd=ROOT / 'upstream',
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        captured = json.loads((temp / 'args.json').read_text())
        argv = captured['argv']
        if Path(argv[0]).resolve() != ROOT / 'upstream/scripts/internal/stage1/eval_verified_test_rerank_temporal_grounding.py':
            raise ValueError('Unexpected evaluation module')
        actual = dict(zip(argv[1::2], argv[2::2]))
        reference = json.loads((ROOT / 'reports/05_alignment_checks.json').read_text())['worker_arguments']
        for key, value in reference.items():
            if smoke and key == '--max_samples':
                value = '10'
            if actual.get(key) != value:
                raise ValueError(f'Entry argument changed: {key}')
        if captured['offline'] != '1' or captured['ver_turn'] != 'last_verified':
            raise ValueError('Offline / last-verification policy was lost')
        return {'passed': True, 'model_execution': False, 'gpu_access': False,
                'command': command(env),
                'worker_arguments': {k: v for k, v in actual.items() if k not in (
                    '--output_json', '--output_jsonl', '--save_query_vectors_dir')},
                'real_python_bin': str(RUNTIME), 'offline_models': True,
                'scoring_entry': str(ROOT / 'scripts/score_charades_protocol.py'),
                'run_kind': 'final_smoke' if smoke else 'full',
                'expected_examples': 10 if smoke else 3720, 'expected_videos': 1334,
                'output_policy': 'Fresh independent directory; no smoke outputs or lazy network downloads.'}


def verify_preflight():
    report = json.loads((ROOT / 'reports/08_cpu_preflight.json').read_text())
    if not report.get('passed') or report.get('status') != 'CPU_PREFLIGHT_PASSED':
        raise ValueError('Complete scripts/preflight_charades_full.py before execution')
    signature = hashlib.sha256(json.dumps(report['file_states'], sort_keys=True).encode()).hexdigest()
    if signature != report['input_signature']:
        raise ValueError('Preflight fingerprint is inconsistent')
    import importlib.metadata as metadata
    installed = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in metadata.distributions()}
    if installed != report['runtime']['installed']:
        raise ValueError('Python environment changed since preflight')
    check_source()
    for relative, state in report['file_states'].items():
        current = (ROOT / relative).stat()
        if current.st_size != state['bytes'] or current.st_mtime_ns != state['mtime_ns']:
            raise ValueError(f'Preflight input changed: {relative}; repeat CPU preflight')
    return report


def experiment_signature(report):
    # Launcher/checker maintenance is tracked by input_signature separately.
    # Bind smoke qualification to effective inference inputs and environment.
    tooling = {'scripts/run_charades_full.py', 'scripts/preflight_charades_full.py',
               'scripts/check_charades_alignment.py'}
    files = {p: state['sha256'] for p, state in report['file_states'].items()
             if p not in tooling and not p.startswith('tests/')}
    payload = {'files': files, 'official_commit': COMMIT,
               'worker_arguments': report['full_entry']['worker_arguments'],
               'runtime': report['runtime']['installed']}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def wait_for_gpu_release(query, uuid, timeout=15):
    start = time.monotonic()
    samples, consecutive = [], 0
    while True:
        raw = subprocess.check_output(query, text=True).strip()
        fields = raw.split(', ')
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                        '--format=csv,noheader,nounits'], text=True).strip()
        active = [line for line in apps.splitlines() if uuid in line]
        idle = int(fields[2]) < 1000 and int(fields[3]) < 5 and not active
        samples.append({'elapsed_seconds': round(time.monotonic() - start, 3),
                        'gpu': raw, 'compute_apps': active, 'idle': idle})
        consecutive = consecutive + 1 if idle else 0
        if consecutive >= 2 or time.monotonic() - start >= timeout:
            return {'passed': consecutive >= 2, 'samples': samples,
                    'method': 'Two consecutive idle samples, >=1s apart, without compute processes.'}
        time.sleep(1)


def verify_final_smoke(preflight):
    result = json.loads((ROOT / 'reports/09_final_smoke.json').read_text())
    if (result.get('status') != 'complete' or result.get('run_kind') != 'final_smoke'
            or result.get('expected_examples') != 10 or not result.get('resource_release_passed')
            or result.get('experiment_signature') != experiment_signature(preflight)):
        raise ValueError('A passing final-entry smoke for the current inputs is required before full execution')
    for path_key, hash_key in [('predictions', 'prediction_sha256'), ('primary_result', 'primary_result_sha256')]:
        with Path(result[path_key]).open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != result[hash_key]:
                raise ValueError('Final smoke evidence changed')
    return result


def analyze_timing(log_text, events, start_epoch, process_seconds, target_count=3720):
    def stamp(needle):
        for line in log_text.splitlines():
            if needle in line:
                match = re.match(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)', line)
                if match:
                    return datetime.datetime.fromisoformat(match.group(1).replace(',', '.')).replace(
                        tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
        raise ValueError(f'Missing official timing marker: {needle}')
    loop_start = stamp('Streaming detail jsonl:')
    loop_end = stamp('Saved summary json:')
    startup = loop_start - start_epoch
    if not events or startup < 0 or loop_end < loop_start:
        raise ValueError('Invalid timing observations')
    previous = startup
    durations = []
    for event in events:
        duration = event['elapsed_seconds'] - previous
        if duration < -0.2:
            raise ValueError('Non-monotonic sample timing')
        durations.append(max(0, duration))
        previous = event['elapsed_seconds']
    steady = durations[1:] or durations
    mean = statistics.mean(steady)
    ordered = sorted(steady)
    def percentile(q):
        index = (len(ordered) - 1) * q
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] * (hi - index) + ordered[hi] * (index - lo) if hi != lo else ordered[lo]
    shutdown = max(0, process_seconds - startup - (loop_end - loop_start))
    result = {'startup_seconds': startup, 'official_loop_seconds': loop_end - loop_start,
              'process_seconds': process_seconds, 'shutdown_seconds': shutdown,
              'first_query_seconds': durations[0], 'steady_mean_seconds_per_query': mean,
              'per_query_seconds': durations, 'sample_timing_poll_seconds': 0.1,
              'events': events}
    if len(events) < target_count:
        fixed = startup + shutdown + max(0, durations[0] - mean) + 60
        result['full_estimate'] = {
            'target_queries': target_count, 'point_seconds': fixed + target_count * mean,
            'scenario_lower_seconds': fixed + target_count * min(mean, percentile(0.1)),
            'scenario_upper_seconds': fixed + target_count * max(mean, percentile(0.9)),
            'scoring_allowance_seconds': 60,
            'method': f'Initialization once, first-query excess once, later-query mean times {target_count}. Scenario range uses observed p10/p90; not a confidence interval.',
            'assumptions': 'Same single GPU type, batch1, BF16/vLLM/eager, full local corpus and fixed settings. Query lengths, video sizes and number of turns vary.'}
    return result


def execute(gpu, smoke=False):
    if Path(sys.prefix).resolve() != ROOT / 'envs/videosearchr1':
        raise ValueError('Source scripts/env.sh to use the project interpreter')
    preflight = verify_preflight()
    if not smoke:
        verify_final_smoke(preflight)
    env = load_environment()
    expected = 10 if smoke else 3720
    if smoke:
        env['MAX_SAMPLES'] = '10'
    import fcntl
    with (ROOT / 'cache/charades_full.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        query = ['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid,memory.free,memory.used,utilization.gpu',
                 '--format=csv,noheader,nounits']
        values = subprocess.check_output(query, text=True).strip().split(', ')
        uuid, free, used, utilization = values
        if int(free) < 70000 or int(used) >= 1000 or int(utilization) >= 5:
            raise RuntimeError(f'GPU {gpu} is not idle: {values}')
        probe_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID')
        cuda_uuid = subprocess.check_output([str(RUNTIME), '-B', '-c',
            'import torch;print(torch.cuda.get_device_properties(0).uuid)'], env=probe_env, text=True).strip()
        if cuda_uuid.removeprefix('GPU-') != uuid.removeprefix('GPU-'):
            raise ValueError('CUDA ordinal and NVML UUID disagree')
        stamp = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        output = ROOT / f'outputs/charades_{"final_smoke" if smoke else "full"}_{stamp}'
        output.mkdir()
        env.update(EVAL_GPUS=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID', LOGS_DIR=str(output))
        manifest = {'status': 'running', 'model_execution': True, 'gpu_uuid': uuid,
                    'cuda_uuid': cuda_uuid, 'command': command(env), 'output_dir': str(output),
                    'preflight_signature': preflight['input_signature'], 'expected_examples': expected,
                    'experiment_signature': experiment_signature(preflight),
                    'run_kind': 'final_smoke' if smoke else 'full', 'gpu_index': gpu,
                    'gpu_name': subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=name',
                        '--format=csv,noheader'], text=True).strip(),
                    'ver_turn': 'last_verified', 'terminal_refinement': 'keep_official'}
        write_json(output / 'run_manifest.json', manifest)
        start, start_epoch, peak = time.monotonic(), time.time(), 0
        events, reader, pending = [], None, ''
        predictions = output / f'{BASE}.jsonl'
        def read_events():
            nonlocal reader, pending
            if reader is None and predictions.exists():
                reader = predictions.open()
            if reader is None:
                return
            pending += reader.read()
            lines = pending.split('\n')
            pending = lines.pop()
            for line in lines:
                if line.strip():
                    row = json.loads(line)
                    events.append({'index': row['index'], 'elapsed_seconds': time.monotonic() - start,
                                   'turns': len(row['turns']), 'refines': row['refine_applied_turns']})
        with (output / 'inference.log').open('w') as log, (output / 'gpu_memory.csv').open('w') as monitor:
            monitor.write('elapsed_seconds,used_mib,utilization_percent\n')
            process = subprocess.Popen(command(env), cwd=ROOT / 'upstream', env=env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                last_gpu_poll = -10.0
                while process.poll() is None:
                    now = time.monotonic() - start
                    if now - last_gpu_poll >= 2:
                        used_now, util_now = subprocess.check_output([
                            'nvidia-smi', f'--id={uuid}', '--query-gpu=memory.used,utilization.gpu',
                            '--format=csv,noheader,nounits'], text=True).strip().split(', ')
                        peak = max(peak, int(used_now))
                        monitor.write(f'{now:.1f},{used_now},{util_now}\n')
                        monitor.flush()
                        last_gpu_poll = now
                    read_events()
                    time.sleep(0.1)
                read_events()
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                manifest.update(status='interrupted', returncode=process.returncode)
                write_json(output / 'run_manifest.json', manifest)
                raise
            finally:
                if reader is not None:
                    reader.close()
        manifest.update(returncode=process.returncode, elapsed_seconds=round(time.monotonic()-start, 2),
                        peak_gpu_used_mib=peak, status='checking_results')
        write_json(output / 'run_manifest.json', manifest)
        try:
            if process.returncode:
                raise RuntimeError('Inference failed; inspect inference.log')
            evaluation = json.loads((output / f'{BASE}.json').read_text())
            if (evaluation['rows_valid'] != expected or any(evaluation['skip'].values())
                    or evaluation['video_meta_stats']['meta_payload_misses']):
                raise ValueError('Full-run completeness or metadata checks failed')
            rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
            if len(events) != expected or [e['index'] for e in events] != list(range(expected)):
                raise ValueError('Missing, duplicate, or reordered prediction/timing rows')
            if any(t.get('rollout_depth_actual') != 8 or t.get('rollout_shape') != [8, 2048]
                   for row in rows for t in row['turns'] if t.get('refine_applied')):
                raise ValueError('Applied SQR does not have the selected 8-token shape')
            with (output / 'report.log').open('w') as log:
                subprocess.run(['bash', str(ROOT / 'upstream/scripts/inference/report.bash'), str(predictions)],
                               env=env, cwd=ROOT / 'upstream', stdout=log, stderr=subprocess.STDOUT, check=True)
            primary = output / f'{BASE}.primary.result.json'
            scoring_start = time.monotonic()
            with (output / 'primary_score.log').open('w') as log:
                subprocess.run([str(RUNTIME), '-B', str(ROOT / 'scripts/score_charades_protocol.py'),
                                str(predictions), str(primary), '--expected-count', str(expected)], env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            primary_result = json.loads(primary.read_text())
            if smoke and (primary_result['primary']['VER']['coverage'] != 1
                          or not any(t.get('refine_applied') for row in rows for t in row['turns'])):
                raise ValueError('Smoke requires parseable answers and actual SQR coverage')
            timing = analyze_timing((output / 'inference.log').read_text(errors='replace'), events,
                                    start_epoch, manifest['elapsed_seconds'])
            write_json(output / 'timing.json', timing)
            manifest.update(status='complete', primary_result=str(primary), predictions=str(predictions),
                            primary=primary_result['primary'], refine=evaluation['refine'],
                            timing=timing, scoring_seconds=round(time.monotonic()-scoring_start, 2))
            manifest['prediction_sha256'] = primary_result['source_sha256']
            manifest['primary_result_sha256'] = hashlib.sha256(primary.read_bytes()).hexdigest()
        except BaseException as error:
            manifest.update(status='failed', error=str(error))
            write_json(output / 'run_manifest.json', manifest)
            raise
        finally:
            release = wait_for_gpu_release(query, uuid)
            manifest['resource_check'] = release
            manifest['gpu_after'] = release['samples'][-1]['gpu']
            manifest['resource_release_passed'] = release['passed']
            if manifest['status'] == 'complete' and not manifest['resource_release_passed']:
                manifest['status'] = 'resource_check_failed'
            write_json(output / 'run_manifest.json', manifest)
            if smoke:
                write_json(ROOT / 'reports/09_final_smoke.json', manifest)
        if manifest['status'] != 'complete':
            raise RuntimeError('Smoke results exist, but resource release was not verified')
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Same entry/settings, fixed original indices 0-9 only')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--output', type=Path, help='JSON destination for the CPU dry run')
    args = parser.parse_args()
    if args.execute:
        if args.gpu is None or args.gpu < 0 or args.output is not None:
            parser.error('--execute requires --gpu N; --output is for dry runs only')
        execute(args.gpu, smoke=args.smoke)
    else:
        result = dry_run(smoke=args.smoke)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
