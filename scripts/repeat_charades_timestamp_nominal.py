#!/usr/bin/env python3
"""Add an A/A repeat control after the A/B run, without changing its inputs.

This control was added after noticing historical baseline trajectory variation.
The same 70 questions and order are replayed, not a newly selected subset.
"""
from __future__ import annotations

import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from zoneinfo import ZoneInfo

from inspect_charades_localization_cases import ROOT, jsonl, save, sha
from run_charades_full import BASE, RUNTIME, load_environment, wait_for_gpu_release
from run_charades_timestamp_ab import PLAN, REPORT as AB_REPORT, verify_plan, subset_score

REPORT=ROOT/'reports/41a_charades_timestamp_nominal_repeat.json'


def main():
    if REPORT.exists():raise ValueError('Preserve existing A/A result')
    while True:
        run=json.loads(AB_REPORT.read_text())
        if run['status']=='complete':break
        if run['status']!='running':raise ValueError('Parent A/B did not complete')
        time.sleep(5)
    plan=verify_plan(PLAN)
    gpu=run['gpu_index']
    with (ROOT/'cache/charades_full.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        query=['nvidia-smi',f'--id={gpu}','--query-gpu=uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits']
        uuid,free,used,util=subprocess.check_output(query,text=True).strip().split(', ')
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
        if uuid!=run['gpu_uuid'] or int(free)<70000 or int(used)>=1000 or int(util)>=5 or uuid in apps:
            raise ValueError('Original GPU is not idle')
        env=load_environment();env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID')
        cuda_uuid=subprocess.check_output([str(RUNTIME),'-B','-c','import torch;print(torch.cuda.get_device_properties(0).uuid)'],env=env,text=True).strip()
        if cuda_uuid.removeprefix('GPU-')!=uuid.removeprefix('GPU-'):raise ValueError('GPU identity mismatch')
        stamp=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        output=ROOT/f'outputs/charades_timestamp_nominal_repeat_{stamp}';output.mkdir()
        argv=list(run['conditions']['nominal']['command'])
        replacements={'--audit':str(output/'input_audit.jsonl'),'--output_json':str(output/(BASE+'.json')),
                      '--output_jsonl':str(output/(BASE+'.jsonl'))}
        for key,value in replacements.items():argv[argv.index(key)+1]=value
        report={'status':'running','reason':'A/A repeat added after noticing historical baseline trajectory variation; no new sample selection.',
            'created_at':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'gpu_index':gpu,'gpu_uuid':uuid,'cuda_uuid':cuda_uuid,'output_dir':str(output),
            'command':argv,'clock':'nominal','expected_queries':70,'plan_sha256':sha(PLAN),
            'ab_run_sha256':sha(AB_REPORT),'script_sha256':sha(Path(__file__))}
        def persist():save(REPORT,report);save(output/'run_manifest.json',report)
        persist();start=time.monotonic();peak=0
        print(f'[A/A] Replaying nominal clock, GPU {gpu}, output {output}',flush=True)
        process=None
        try:
            with (output/'inference.log').open('w') as log,(output/'gpu_memory.csv').open('w') as memory:
                memory.write('elapsed_seconds,used_mib,utilization_percent\n')
                process=subprocess.Popen(argv,cwd=ROOT/'upstream',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while process.poll() is None:
                    u,v=subprocess.check_output(['nvidia-smi',f'--id={uuid}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
                    peak=max(peak,int(u));memory.write(f'{time.monotonic()-start:.2f},{u},{v}\n');memory.flush();time.sleep(2)
            report.update(returncode=process.returncode,elapsed_seconds=round(time.monotonic()-start,2),peak_gpu_used_mib=peak)
            report['resource_release']=wait_for_gpu_release(query,uuid)
            if process.returncode or not report['resource_release']['passed']:raise ValueError('A/A process or GPU release check failed')
            verify_plan(PLAN)
            evaluation=json.loads((output/(BASE+'.json')).read_text())
            if evaluation['rows_valid']!=70 or any(evaluation['skip'].values()) or evaluation['video_meta_stats']['meta_payload_misses']:
                raise ValueError('Incomplete A/A result')
            rows=jsonl(output/(BASE+'.jsonl'));audits=jsonl(output/'input_audit.jsonl')
            if [r['index'] for r in rows]!=plan['selection']['original_indices'] or len(audits)!=sum(len(r['turns']) for r in rows):
                raise ValueError('A/A prediction or audit coverage mismatch')
            original_audits=jsonl(Path(run['conditions']['nominal']['output_dir'])/'input_audit.jsonl')
            seen={a['video_id']:a for a in original_audits}
            for event in audits:
                if event['video_id'] in seen:
                    old=seen[event['video_id']]
                    if any(event[k]!=old[k] for k in ['video_kwargs','video_shapes','video_tensor_sha256','time_labels']):
                        raise ValueError('A/A visual input or time labels differ')
            report.update(status='complete',prediction_sha256=sha(output/(BASE+'.jsonl')),
                input_audit_sha256=sha(output/'input_audit.jsonl'),
                cohorts={name:subset_score(rows,ids) for name,ids in plan['selection']['cohorts'].items()},
                completed_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
            persist();print(json.dumps({'status':'complete','report':str(REPORT),'output':str(output)},ensure_ascii=False),flush=True)
        except BaseException as error:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            report.update(status='failed',error=str(error));persist();raise


if __name__=='__main__':main()
