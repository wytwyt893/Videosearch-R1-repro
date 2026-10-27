#!/usr/bin/env python3
"""Run five official queries on an idle physical GPU; audit output completeness."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1)
args = parser.parse_args()
for filename in ['03_runtime_validation.json', '03_data_validation.json']:
    assert json.loads((ROOT/'reports'/filename).read_text())['passed'], filename
assert json.loads((ROOT/'reports/03_models.json').read_text())['verified']
gpu_query = ['nvidia-smi', f'--id={args.gpu}',
             '--query-gpu=uuid,memory.free,memory.used,utilization.gpu', '--format=csv,noheader,nounits']
gpu = subprocess.check_output(gpu_query, text=True).strip().split(', ')
uuid, free, used, utilization = gpu
assert int(free) >= 70000 and int(used) < 1000 and int(utilization) < 5, f'GPU not idle: {gpu}'
# vLLM 0.11 parses CUDA_VISIBLE_DEVICES entries as integers. Verify that the
# numeric CUDA ordinal maps to the expected NVML UUID before loading models.
probe_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID')
cuda_uuid = subprocess.check_output([sys.executable, '-B', '-c',
    'import torch; print(torch.cuda.get_device_properties(0).uuid)'], env=probe_env, text=True).strip()
assert cuda_uuid.removeprefix('GPU-') == uuid.removeprefix('GPU-'), f'CUDA/NVML device mapping mismatch: {cuda_uuid} != {uuid}'
stamp = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
output = ROOT/f'outputs/charades_smoke_{stamp}'
output.mkdir()
env = os.environ.copy()
env.update(EVAL_GPUS=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID', NUM_PROCESSES_PER_GPU='1', MAX_SAMPLES='5', MAX_TURN='2',
           REFINE_TOKEN_COUNT='8', SEED='42', RESUME_FROM_JSONL='False',
           USE_VLLM='True', VLLM_MAX_NUM_SEQS='1', VLLM_EVAL_BATCH_SIZE='1',
           VLLM_GPU_MEMORY_UTILIZATION='0.45', VLLM_ENFORCE_EAGER='True',
           QUERY_EMBEDDER_MODEL_PATH=str(ROOT/'checkpoints/Qwen__Qwen3-VL-Embedding-2B'),
           LOGS_DIR=str(output), PYTHON_BIN=str(ROOT/'tools/bin/smoke_python'),
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           HF_TOKEN='', HUGGING_FACE_HUB_TOKEN='')
checkpoint = ROOT/'checkpoints/VideoSearchR1__charades-stage2'
command = ['bash', 'scripts/inference/inference.bash', 'charades', '--checkpoint', str(checkpoint)]
manifest = {'started': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'gpu_index': args.gpu, 'gpu_uuid': uuid, 'cuda_uuid_verified': cuda_uuid, 'gpu_before': gpu,
            'command': command, 'output_dir': str(output),
            'parameters': {k:env[k] for k in ['MAX_SAMPLES', 'MAX_TURN', 'REFINE_TOKEN_COUNT', 'SEED',
                           'USE_VLLM', 'VLLM_GPU_MEMORY_UTILIZATION', 'VLLM_ENFORCE_EAGER']},
            'data_policy': 'Full official test embedding corpus; lazy fetch of official video tensors only; no query/corpus substitution.'}
(output/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
start = time.monotonic()
peak = 0
print(f'[smoke] {output} on {uuid}', flush=True)
with (output/'inference.log').open('w') as log, (output/'gpu_memory.csv').open('w') as monitor:
    monitor.write('elapsed_seconds,used_mib,utilization_percent\n')
    process = subprocess.Popen(command, cwd=ROOT/'upstream', env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        while process.poll() is None:
            used_now, util_now = subprocess.check_output([
                'nvidia-smi', f'--id={uuid}', '--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True).strip().split(', ')
            peak = max(peak, int(used_now))
            monitor.write(f'{time.monotonic()-start:.1f},{used_now},{util_now}\n')
            monitor.flush()
            time.sleep(2)
    except BaseException:
        process.terminate()
        process.wait(timeout=30)
        raise
manifest.update(returncode=process.returncode, elapsed_seconds=round(time.monotonic()-start, 2),
                peak_gpu_used_mib=peak, completed=False)
(output/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
assert process.returncode == 0, f'Smoke failed; inspect {output}/inference.log'
base = 'external_verified_test_temporal_grounding_VideoSearchR1__charades-stage2'
predictions = output/f'{base}.jsonl'
rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
assert len(rows) == 5 and len({r['index'] for r in rows}) == 5
assert {r['index'] for r in rows} == set(range(5))
evaluation = json.loads((output/f'{base}.json').read_text())
assert evaluation['rows_valid'] == 5 and not any(evaluation['skip'].values()), evaluation['skip']
assert evaluation['video_meta_stats']['meta_payload_misses'] == 0
with (output/'report.log').open('w') as log:
    subprocess.run(['bash', 'scripts/inference/report.bash', str(predictions)], cwd=ROOT/'upstream',
                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
manifest.update(completed=True, predictions=str(predictions), evaluation=str(output/f'{base}.json'),
                refine=evaluation.get('refine'), sample_count=5,
                results=[str(p) for p in output.glob('*.result.json')])
(output/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
(ROOT/'reports/03_smoke_result.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest, indent=2), flush=True)
