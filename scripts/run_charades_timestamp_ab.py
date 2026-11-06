#!/usr/bin/env python3
"""Run a CPU-qualified, paired Charades clock ablation on one idle GPU."""
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

from inspect_charades_localization_cases import LOCAL, ROOT, jsonl, save, sha
from run_charades_full import BASE, RUNTIME, load_environment, verify_preflight, wait_for_gpu_release
from score_charades_protocol import score_rows

PLAN=ROOT/'reports/40_charades_timestamp_cpu.json'
REPORT=ROOT/'reports/41_charades_timestamp_ab.json'


def verify_plan(path):
    plan=json.loads(path.read_text())
    preflight=verify_preflight()
    if plan['status']!='TIMESTAMP_CPU_CHECK_PASSED' or plan['preflight_input_signature']!=preflight['input_signature']:
        raise ValueError('CPU qualification does not match frozen inputs')
    for relative,digest in plan['file_sha256'].items():
        if sha(ROOT/relative)!=digest:
            raise ValueError(f'Clock experiment input changed: {relative}')
    return plan


def subset_score(rows,indices):
    by_index={r['index']:r for r in rows}
    if len(by_index)!=len(rows) or not set(indices)<=set(by_index):
        raise ValueError('Missing or repeated original index')
    # The existing scorer requires contiguous indices. Only copies are remapped;
    # original prediction files, qids, labels and corpus are never changed.
    copies=[{**by_index[i],'index':j} for j,i in enumerate(indices)]
    result=score_rows(copies,len(copies),include_first_turn=True)
    result['original_indices']=indices
    return result


def semantic_record(row):
    return {k:row[k] for k in ['orig_rank','final_rank','final_top1_doc_id','final_answer',
        'final_pred_start','final_pred_end','final_iou_r1','refine_applied_turns','turns']}


def compare(output,plan):
    rows={clock:jsonl(output/clock/(BASE+'.jsonl')) for clock in ['nominal','source']}
    indices=plan['selection']['original_indices']
    historical=jsonl(LOCAL)
    for clock,group in rows.items():
        if [r['index'] for r in group]!=indices:
            raise ValueError('The two conditions must preserve every selected original index')
        for r in group:
            h=historical[r['index']]
            if any(r[k]!=h[k] for k in ['query','qid','gt_pos_doc_id','gt_start','gt_end','orig_rank','turn1_top1_doc_id']):
                raise ValueError('Question, GT, or initial retrieval changed')
    audit={clock:jsonl(output/clock/'input_audit.jsonl') for clock in rows}
    canonical={}
    for clock,events in audit.items():
        if len(events)!=sum(len(r['turns']) for r in rows[clock]):
            raise ValueError('Every actual generation must have an input audit')
        for event in events:
            vid=event['video_id']
            visual={k:event[k] for k in ['video_kwargs','video_shapes','video_tensor_sha256']}
            if vid in canonical and canonical[vid]!=visual:
                raise ValueError('A visual input changed between conditions or turns')
            canonical[vid]=visual
    metadata={r['video_id']:r for r in jsonl(ROOT/'data/charades-sta/test/video_npy_with_meta/meta.jsonl')}
    for clock,events in audit.items():
        for event in events:
            times=metadata[event['video_id']]['sampled_timestamps_sec'] if clock=='source' else list(range(sum(s[0] for s in event['video_shapes'])))
            if len(times)%2:times=times+[times[-1]]
            expected=[f'{(times[i]+times[i+1])/2:.1f}' for i in range(0,len(times),2)]
            if event['time_labels']!=expected:
                raise ValueError(f'Runtime clock labels disagree for {clock}/{event["video_id"]}')
    by={clock:{r['index']:r for r in group} for clock,group in rows.items()}
    paired=[]
    for i in indices:
        a,b=by['nominal'][i],by['source'][i]
        record={'index':i,'query':a['query'],'gt_video':a['gt_pos_doc_id'],
            'gt':[a['gt_start'],a['gt_end']],
            'nominal_span':[a['final_pred_start'],a['final_pred_end']],
            'source_span':[b['final_pred_start'],b['final_pred_end']],
            'nominal_iou':a['final_iou_r1'],'source_iou':b['final_iou_r1'],
            'delta_iou':b['final_iou_r1']-a['final_iou_r1'],
            'nominal_last_video':a['turns'][-1]['top1_doc_id'],
            'source_last_video':b['turns'][-1]['top1_doc_id'],
            'nominal_answer':a['final_answer'],'source_answer':b['final_answer'],
            'nominal_refines':a['refine_applied_turns'],'source_refines':b['refine_applied_turns'],
            'historical_iou':historical[i]['final_iou_r1'],
            'nominal_replays_historical_semantics':semantic_record(a)==semantic_record(historical[i])}
        record['same_verified_GT_and_matched']=all(r['turns'][-1]['top1_is_gt'] and r['final_answer']=='matched' and r['final_time_parse_ok'] for r in [a,b])
        paired.append(record)
    cohorts={}
    for name,selected in plan['selection']['cohorts'].items():
        examples=[p for p in paired if p['index'] in selected]
        cohorts[name]={'N':len(selected),'nominal':subset_score(rows['nominal'],selected),
            'source':subset_score(rows['source'],selected),
            'historical':subset_score(historical,selected),
            'iou_improved':sum(p['delta_iou']>1e-9 for p in examples),
            'iou_worsened':sum(p['delta_iou']< -1e-9 for p in examples),
            'iou_unchanged':sum(abs(p['delta_iou'])<=1e-9 for p in examples),
            'strict_0point7_gained':sum(p['nominal_iou']<.7<=p['source_iou'] for p in examples),
            'strict_0point7_lost':sum(p['source_iou']<.7<=p['nominal_iou'] for p in examples),
            'nominal_historical_semantic_replays':sum(p['nominal_replays_historical_semantics'] for p in examples)}
    result={'cohorts':cohorts,'paired_cases':paired,
        'runtime_audit':{'all_generation_calls_covered':True,'consistent_visual_tensors_for_every_seen_video':True,
            'seen_videos':len(canonical),'generation_calls':{k:len(v) for k,v in audit.items()},
            'source_and_nominal_labels_verified':True},
        'limits':'Small exploratory paired test, with diagnostic and random cohorts reported separately. No new full-benchmark result; source times inherit released metadata, not newly verified container PTS.'}
    save(output/'comparison.json',result)
    return result


def execute(gpu,plan_path,report_path):
    if Path(os.sys.prefix).resolve()!=ROOT/'envs/videosearchr1':raise ValueError('Use the project Python')
    if report_path.exists():raise ValueError('Preserve previous experiment report')
    plan=verify_plan(plan_path)
    with (ROOT/'cache/charades_full.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=load_environment()
        query=['nvidia-smi',f'--id={gpu}','--query-gpu=uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits']
        uuid,free,used,util=subprocess.check_output(query,text=True).strip().split(', ')
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
        if int(free)<70000 or int(used)>=1000 or int(util)>=5 or uuid in apps:raise RuntimeError('Requested GPU is not idle')
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID')
        cuda_uuid=subprocess.check_output([str(RUNTIME),'-B','-c','import torch;print(torch.cuda.get_device_properties(0).uuid)'],env=env,text=True).strip()
        if cuda_uuid.removeprefix('GPU-')!=uuid.removeprefix('GPU-'):raise ValueError('Physical GPU identity mismatch')
        stamp=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        output=ROOT/f'outputs/charades_timestamp_ab_{stamp}'
        output.mkdir()
        manifest={'status':'running','run_kind':'timestamp_only_paired70','output_dir':str(output),
            'created_at':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'gpu_index':gpu,'gpu_uuid':uuid,'cuda_uuid':cuda_uuid,
            'plan':str(plan_path),'plan_sha256':sha(plan_path),'selection':plan['selection'],
            'expected_queries_per_condition':70,'corpus_videos':1334,'conditions':{},
            'launcher_sha256':sha(Path(__file__)),'full_benchmark_started':False}
        def persist():save(output/'run_manifest.json',manifest);save(report_path,manifest)
        persist()
        try:
            for clock in ['nominal','source']:
                verify_plan(plan_path)
                condition=output/clock;condition.mkdir()
                arguments=dict(plan['arguments'])
                arguments.update({'--output_json':str(condition/(BASE+'.json')),'--output_jsonl':str(condition/(BASE+'.jsonl'))})
                argv=[str(RUNTIME),'-B',str(ROOT/'scripts/charades_timestamp_eval.py'),'--clock',clock,'--audit',str(condition/'input_audit.jsonl')]+[x for p in arguments.items() for x in p]
                state={'status':'running','command':argv,'arguments':arguments,'output_dir':str(condition)}
                manifest['conditions'][clock]=state;manifest['current_condition']=clock;persist()
                print(f'[A/B] {clock}: GPU {gpu}, 70 queries, output {condition}',flush=True)
                start=time.monotonic();peak=0
                with (condition/'inference.log').open('w') as log,(condition/'gpu_memory.csv').open('w') as memory:
                    memory.write('elapsed_seconds,used_mib,utilization_percent\n')
                    process=subprocess.Popen(argv,cwd=ROOT/'upstream',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        while process.poll() is None:
                            now=time.monotonic()-start
                            u,v=subprocess.check_output(['nvidia-smi',f'--id={uuid}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
                            peak=max(peak,int(u));memory.write(f'{now:.2f},{u},{v}\n');memory.flush()
                            time.sleep(2)
                    except BaseException:
                        if process.poll() is None:
                            os.killpg(process.pid,signal.SIGTERM)
                            try:process.wait(timeout=30)
                            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                        raise
                state.update(returncode=process.returncode,elapsed_seconds=round(time.monotonic()-start,2),peak_gpu_used_mib=peak)
                release=wait_for_gpu_release(query,uuid)
                state['resource_release']=release
                if process.returncode:raise RuntimeError(f'{clock} inference failed; inspect its log')
                if not release['passed']:raise RuntimeError('GPU did not return to idle between conditions')
                evaluation=json.loads((condition/(BASE+'.json')).read_text())
                if evaluation['rows_valid']!=70 or any(evaluation['skip'].values()) or evaluation['video_meta_stats']['meta_payload_misses']:
                    raise ValueError('Incomplete inference or metadata miss')
                predictions=condition/(BASE+'.jsonl')
                state.update(status='complete',prediction_sha256=sha(predictions),audit_sha256=sha(condition/'input_audit.jsonl'))
                persist();print(f'[A/B] {clock} complete in {state["elapsed_seconds"]} seconds; GPU released.',flush=True)
            verify_plan(plan_path)
            manifest['comparison']=compare(output,plan)
            manifest.update(status='complete',completed_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),comparison_file=str(output/'comparison.json'))
            persist();print(json.dumps({'status':'complete','report':str(report_path),'output':str(output)},ensure_ascii=False),flush=True)
        except BaseException as error:
            manifest.update(status='failed',error=str(error));persist();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--gpu',type=int)
    parser.add_argument('--plan',type=Path,default=PLAN)
    parser.add_argument('--report',type=Path,default=REPORT)
    args=parser.parse_args()
    if args.execute:
        if args.gpu is None or args.gpu<0:parser.error('--execute requires --gpu N')
        execute(args.gpu,args.plan,args.report)
    else:
        p=verify_plan(args.plan)
        print(json.dumps({'passed':True,'model_execution':False,'gpu_access':False,'selection':p['selection'],'arguments':p['arguments']},ensure_ascii=False,indent=2))
