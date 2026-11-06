#!/usr/bin/env python3
"""Prepare and run the smoke-qualified, single-GPU full clock A/B overnight.

The qualified 70-question entry, time adapter and upstream remain unchanged.
Only the question-exclusion list is removed. Each condition retains all 3720
questions, the original order, and the complete 1334-video retrieval corpus.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from charades_timestamp_eval import source_payload
from inspect_charades_localization_cases import ROOT, LOCAL, REFERENCE, jsonl, sha
from run_charades_full import BASE, RUNTIME, dry_run, load_environment, verify_preflight, wait_for_gpu_release
from run_charades_timestamp_ab import verify_plan as verify_small_plan
from score_charades_protocol import score_rows

PLAN = ROOT/'reports/43_charades_timestamp_full_cpu.json'
REPORT = ROOT/'reports/44_charades_timestamp_full.json'
ADAPTER = ROOT/'scripts/charades_timestamp_eval.py'
SUMMARY = ROOT/'scripts/summarize_charades_timestamp_full.py'
N = 3720


def now():
    return datetime.datetime.now(ZoneInfo('Asia/Shanghai'))


def write(path, value):
    """Readers never see a half-written progress/manifest JSON."""
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    os.replace(temporary,path)


def full_arguments(qualified):
    args = dict(qualified)
    args['--resume_skip_jsonl'] = ''
    required = {'--max_samples':'0','--eval_limit_ratio':'1.0','--num_shards':'1',
                '--shard_id':'0','--resume_from_jsonl':'False','--max_turn':'2',
                '--refine_token_count':'8','--vllm_tensor_parallel_size':'1',
                '--vllm_eval_batch_size':'1','--seed':'42'}
    if any(args.get(k)!=v for k,v in required.items()):
        raise ValueError('Full entry must retain every question, T=2/N=8, and single-GPU settings')
    return args


def qualification():
    small_path = ROOT/'reports/40_charades_timestamp_cpu.json'
    small = verify_small_plan(small_path)
    run_path = ROOT/'reports/41_charades_timestamp_ab.json'
    run = json.loads(run_path.read_text())
    repeat_path = ROOT/'reports/41a_charades_timestamp_nominal_repeat.json'
    repeat = json.loads(repeat_path.read_text())
    summary_path = ROOT/'reports/42_charades_timestamp_summary.json'
    summary = json.loads(summary_path.read_text())
    if (run['status']!='complete' or run['plan_sha256']!=sha(small_path)
            or run['expected_queries_per_condition']!=70
            or repeat['status']!='complete' or repeat['ab_run_sha256']!=sha(run_path)
            or not repeat['resource_release']['passed']
            or summary['status']!='TIMESTAMP_PAIRED_EXPLORATORY_CHECK_COMPLETE'
            or summary['run_report_sha256']!=sha(run_path)
            or summary['repeat_report_sha256']!=sha(repeat_path)
            or summary['AA_core_outcome_mismatch_indices']
            or not summary['actual_T2_N8_verified_all_conditions']):
        raise ValueError('Require passing paired70, A/A repeat, and bound timestamp input checks')
    for condition in run['conditions'].values():
        d = Path(condition['output_dir'])
        if (condition['status']!='complete' or not condition['resource_release']['passed']
                or sha(d/(BASE+'.jsonl'))!=condition['prediction_sha256']
                or sha(d/'input_audit.jsonl')!=condition['audit_sha256']):
            raise ValueError('Qualified condition evidence changed')
    repeat_dir = Path(repeat['output_dir'])
    if (sha(repeat_dir/(BASE+'.jsonl'))!=repeat['prediction_sha256']
            or sha(repeat_dir/'input_audit.jsonl')!=repeat['input_audit_sha256']):
        raise ValueError('A/A evidence changed')
    if not all(run['comparison']['runtime_audit'][k] for k in [
            'all_generation_calls_covered','consistent_visual_tensors_for_every_seen_video',
            'source_and_nominal_labels_verified']):
        raise ValueError('Small-run visual/timestamp audit did not pass')
    return small, [small_path,run_path,repeat_path,summary_path]


def prepare(destination):
    if destination.exists():
        raise ValueError('Preserve existing full CPU plan')
    if Path(sys.prefix).resolve()!=ROOT/'envs/videosearchr1' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Source scripts/env.sh and keep CUDA hidden for preparation')
    preflight = verify_preflight()
    small, qualified_reports = qualification()
    args = full_arguments(small['arguments'])
    entry = dry_run()
    if args!=entry['worker_arguments']:
        raise ValueError('Full arguments differ from the frozen official full entry')
    annotations = jsonl(ROOT/'data/charades-sta/raw_annotation/test.jsonl')
    metadata = jsonl(ROOT/'data/charades-sta/test/video_npy_with_meta/meta.jsonl')
    if len(annotations)!=N or len(metadata)!=1334:
        raise ValueError('Require complete official Charades data')
    for row in metadata:
        source_payload(row)
    if sha(LOCAL)!='35cf85cc43a42d96c2bcc8fe47d25416014db7669b09c5b3bd4e06b4faa3f425':
        raise ValueError('Historical predictions changed')
    if sha(REFERENCE)!='ac2256b6a5d882df706cc8b0730b22d29594b5bcb2d38894ac782b3def01ac2f':
        raise ValueError('Pinned author predictions changed')
    files = [Path(__file__),SUMMARY,ADAPTER,ROOT/'scripts/run_charades_timestamp_ab.py',
             ROOT/'scripts/score_charades_protocol.py',ROOT/'scripts/run_charades_full.py',
             ROOT/'scripts/summarize_charades_timestamp_ab.py',LOCAL,REFERENCE] + qualified_reports
    report = {'status':'FULL_TIMESTAMP_CPU_QUALIFIED','created_at':now().isoformat(),
        'model_execution':False,'gpu_access':False,'expected_queries_per_condition':N,
        'expected_corpus_videos':1334,'question_order':'Original test indices 0..3719; no deduplication or filtering',
        'condition_order':['nominal','source'],'single_gpu_sequential':True,
        'arguments':args,'preflight_input_signature':preflight['input_signature'],
        'qualified_small_run':'Paired A/B70 plus same-order A/A70; all 70 core outcomes reproduced.',
        'differences_from_qualified_small_run':['resume_skip_jsonl cleared: 70 -> 3720; fresh outputs'],
        'time_semantics':small['timestamp_semantics'],'held_fixed':small['held_fixed'],
        'file_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},
        'disk_free_bytes':shutil.disk_usage(ROOT).free}
    if report['disk_free_bytes']<5*1024**3:
        raise ValueError('Insufficient space for independent outputs')
    write(destination,report)
    print(json.dumps({'status':report['status'],'plan':str(destination),'queries_each':N,
                      'corpus':1334,'single_gpu':True},ensure_ascii=False),flush=True)
    return report


def verify(path):
    report = json.loads(path.read_text())
    preflight = verify_preflight()
    if (report['status']!='FULL_TIMESTAMP_CPU_QUALIFIED'
            or report['preflight_input_signature']!=preflight['input_signature']
            or full_arguments(report['arguments'])!=report['arguments']):
        raise ValueError('Full plan is not qualified for current inputs')
    for relative,digest in report['file_sha256'].items():
        if sha(ROOT/relative)!=digest:
            raise ValueError(f'Full experiment input changed: {relative}')
    return report


def validate_rows(rows,expected=N):
    if [r['index'] for r in rows]!=list(range(expected)):
        raise ValueError('Missing, repeated, or reordered original indices')
    for row in rows:
        if row['max_turn']!=2:
            raise ValueError('T changed')
        for t in row['turns']:
            if t.get('refine_applied') and (t.get('rollout_depth_actual')!=8 or t.get('rollout_shape')!=[8,2048]):
                raise ValueError('Applied SQR does not retain N=8')
    # Invalid answers retain their explicit coverage and are never dropped.
    return score_rows(rows,expected,include_first_turn=True)


def execute(gpu,plan_path,report_path):
    if Path(sys.prefix).resolve()!=ROOT/'envs/videosearchr1':
        raise ValueError('Use the project interpreter')
    if report_path.exists():
        raise ValueError('Preserve existing experiment; choose a new report')
    plan = verify(plan_path)
    with (ROOT/'cache/charades_full.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env = load_environment()
        query = ['nvidia-smi',f'--id={gpu}','--query-gpu=uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits']
        uuid,free,used,util = subprocess.check_output(query,text=True).strip().split(', ')
        apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
        if int(free)<70000 or int(used)>=1000 or int(util)>=5 or uuid in apps:
            raise RuntimeError('Selected GPU is not idle')
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID')
        cuda_uuid = subprocess.check_output([str(RUNTIME),'-B','-c',
            'import torch;print(torch.cuda.get_device_properties(0).uuid)'],env=env,text=True).strip()
        if cuda_uuid.removeprefix('GPU-')!=uuid.removeprefix('GPU-'):
            raise ValueError('CUDA/NVML physical GPU identity mismatch')
        output = ROOT/f'outputs/charades_timestamp_full_{now():%Y%m%d_%H%M%S_%f}'
        output.mkdir()
        manifest = {'status':'running','created_at':now().isoformat(),
            'run_kind':'charades_full_timestamp_paired','output_dir':str(output),
            'gpu_index':gpu,'gpu_uuid':uuid,'cuda_uuid':cuda_uuid,
            'gpu_name':subprocess.check_output(['nvidia-smi',f'--id={uuid}','--query-gpu=name','--format=csv,noheader'],text=True).strip(),
            'single_gpu_sequential':True,'condition_order':['nominal','source'],
            'expected_queries_per_condition':N,'expected_corpus_videos':1334,
            'plan':str(plan_path),'plan_sha256':sha(plan_path),'launcher_sha256':sha(Path(__file__)),
            'ver_policy':'last_verified/valid_answers; invalid and all-example accuracy explicit',
            'terminal_refinement':'keep_official','conditions':{},'original_baseline_overwritten':False}
        def persist():
            write(output/'run_manifest.json',manifest)
            write(report_path,manifest)
        persist()
        print(f'[full A/B] GPU{gpu}, single-card sequential, output={output}',flush=True)
        print(f'[full A/B] Both conditions: 3720 queries, full 1334-video corpus, T=2/N=8.',flush=True)
        current_process = None
        try:
            for clock in plan['condition_order']:
                verify(plan_path)
                if not wait_for_gpu_release(query,uuid)['passed']:
                    raise ValueError('GPU must be idle before each condition')
                condition = output/clock
                condition.mkdir()
                arguments = dict(plan['arguments'])
                arguments.update({'--output_json':str(condition/(BASE+'.json')),
                                  '--output_jsonl':str(condition/(BASE+'.jsonl'))})
                argv = [str(RUNTIME),'-B',str(ADAPTER),'--clock',clock,
                        '--audit',str(condition/'input_audit.jsonl')] + [x for pair in arguments.items() for x in pair]
                state = {'status':'running','started_at':now().isoformat(),'command':argv,
                    'arguments':arguments,'output_dir':str(condition),'completed_queries':0}
                manifest['conditions'][clock] = state
                manifest['current_condition'] = clock
                persist()
                print(f'[full A/B] START {clock}: 0/{N}, {now().isoformat()}',flush=True)
                start = time.monotonic()
                peak = 0
                reader = None
                pending = ''
                completed = 0
                last_progress = -60.0
                predictions = condition/(BASE+'.jsonl')
                def read_predictions(events):
                    nonlocal reader,pending,completed
                    if reader is None and predictions.exists():
                        reader = predictions.open()
                    if reader is None:return
                    pending += reader.read()
                    parts = pending.split('\n')
                    pending = parts.pop()
                    for line in parts:
                        if not line.strip():continue
                        row = json.loads(line)
                        if row['index']!=completed:
                            raise ValueError('Streaming predictions lost original order')
                        completed += 1
                        events.write(json.dumps({'index':row['index'],'elapsed_seconds':round(time.monotonic()-start,3),
                            'turns':len(row['turns']),'refines':row['refine_applied_turns']})+'\n')
                        events.flush()
                with (condition/'inference.log').open('w') as log,(condition/'gpu_memory.csv').open('w') as memory,(condition/'progress_events.jsonl').open('w') as events:
                    memory.write('elapsed_seconds,used_mib,utilization_percent\n')
                    current_process = subprocess.Popen(argv,cwd=ROOT/'upstream',env=env,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state['launcher_child_pid'] = current_process.pid
                    persist()
                    try:
                        while current_process.poll() is None:
                            elapsed = time.monotonic()-start
                            used_now,util_now = subprocess.check_output(['nvidia-smi',f'--id={uuid}',
                                '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
                            peak = max(peak,int(used_now))
                            memory.write(f'{elapsed:.2f},{used_now},{util_now}\n');memory.flush()
                            read_predictions(events)
                            if elapsed-last_progress>=30:
                                state.update(completed_queries=completed,elapsed_seconds=round(elapsed,2),peak_gpu_used_mib=peak,
                                             last_progress_at=now().isoformat())
                                persist()
                                write(output/'progress.json',{'status':'running','clock':clock,'completed':completed,'expected':N,
                                    'elapsed_seconds':round(elapsed,2),'updated_at':now().isoformat()})
                                print(f'[full A/B] {clock}: {completed}/{N}, elapsed {elapsed/3600:.2f}h',flush=True)
                                last_progress = elapsed
                            time.sleep(2)
                        read_predictions(events)
                    finally:
                        if reader is not None:reader.close()
                state.update(returncode=current_process.returncode,completed_queries=completed,
                    elapsed_seconds=round(time.monotonic()-start,2),peak_gpu_used_mib=peak)
                state['resource_release'] = wait_for_gpu_release(query,uuid)
                if current_process.returncode:
                    raise RuntimeError(f'{clock} inference exited {current_process.returncode}; inspect its inference.log')
                if not state['resource_release']['passed']:
                    raise RuntimeError('GPU resources were not released')
                state['status'] = 'checking_results'
                persist()
                evaluation = json.loads((condition/(BASE+'.json')).read_text())
                if evaluation['rows_valid']!=N or any(evaluation['skip'].values()) or evaluation['video_meta_stats']['meta_payload_misses']:
                    raise ValueError('Full condition is incomplete or has missing metadata')
                rows = jsonl(predictions)
                primary = validate_rows(rows)
                primary.update(source_jsonl=str(predictions),source_sha256=sha(predictions))
                primary_path = condition/(BASE+'.primary.result.json')
                write(primary_path,primary)
                cpu_env = dict(env,CUDA_VISIBLE_DEVICES='')
                with (condition/'report.log').open('w') as log:
                    native = subprocess.run(['bash',str(ROOT/'upstream/scripts/inference/report.bash'),str(predictions)],
                        env=cpu_env,cwd=ROOT/'upstream',stdout=log,stderr=subprocess.STDOUT)
                state.update(status='complete',completed_at=now().isoformat(),predictions=str(predictions),
                    prediction_sha256=primary['source_sha256'],primary_result=str(primary_path),
                    primary_result_sha256=sha(primary_path),input_audit=str(condition/'input_audit.jsonl'),
                    audit_sha256=sha(condition/'input_audit.jsonl'),primary=primary['primary'],
                    native_report_returncode=native.returncode)
                persist()
                print(f'[full A/B] COMPLETE {clock}: {N}/{N}, GPU released, {state["elapsed_seconds"]/3600:.2f}h',flush=True)
                current_process = None
            verify(plan_path)
            manifest.update(status='inference_complete',current_condition='postprocessing')
            persist()
            print('[full A/B] Both full conditions complete; starting CPU paired comparison.',flush=True)
            cpu_env = dict(env,CUDA_VISIBLE_DEVICES='')
            with (output/'postprocess.log').open('w') as log:
                result = subprocess.run([str(RUNTIME),'-B',str(SUMMARY),'--run',str(report_path)],
                    env=cpu_env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('Both prediction files are preserved. CPU comparison failed; see postprocess.log; do not rerun inference.')
            manifest.update(status='complete',completed_at=now().isoformat(),
                comparison_report=str(ROOT/'reports/45_charades_timestamp_full_summary.json'),
                comparison_file=str(output/'comparison.json'))
            persist()
            write(output/'progress.json',{'status':'complete','conditions':{'nominal':N,'source':N},'completed_at':now().isoformat()})
            print(f'[full A/B] COMPLETE both conditions; report={report_path}',flush=True)
        except BaseException as error:
            if current_process is not None and current_process.poll() is None:
                os.killpg(current_process.pid,signal.SIGTERM)
                try:current_process.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(current_process.pid,signal.SIGKILL);current_process.wait()
            manifest.update(status='failed',error=str(error),failed_at=now().isoformat())
            try:manifest['failure_resource_check'] = wait_for_gpu_release(query,uuid)
            except Exception as resource_error:manifest['failure_resource_check'] = {'error':str(resource_error)}
            persist()
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--execute',action='store_true')
    p.add_argument('--gpu',type=int)
    p.add_argument('--plan',type=Path,default=PLAN)
    p.add_argument('--report',type=Path,default=REPORT)
    args = p.parse_args()
    if args.prepare:
        prepare(args.plan)
    elif args.execute:
        if args.gpu is None or args.gpu<0:p.error('--execute requires --gpu N')
        signal.signal(signal.SIGTERM,lambda *_:sys.exit('SIGTERM'))
        execute(args.gpu,args.plan,args.report)
    else:
        r = verify(args.plan)
        print(json.dumps({'passed':True,'model_execution':False,'gpu_access':False,
            'condition_order':r['condition_order'],'queries_each':N,'single_gpu_sequential':True,
            'arguments':r['arguments']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
