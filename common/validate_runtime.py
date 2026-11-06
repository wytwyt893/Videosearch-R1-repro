#!/usr/bin/env python3
"""CPU-only runtime validation, separate from the historical Stage 1 report."""
import datetime
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve() == ROOT/'envs/videosearchr1'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
installed = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in metadata.distributions()}
pins = dict(line.strip().split('==') for line in (ROOT/'upstream/requirements.txt').read_text().splitlines()
            if line.strip() and not line.startswith('#'))
missing = {k: {'expected': v, 'actual': installed.get(k.lower().replace('_', '-'))}
           for k, v in pins.items() if installed.get(k.lower().replace('_', '-'), '').split('+')[0] != v}
def run(command):
    r = subprocess.run(command, capture_output=True, text=True, timeout=120)
    return {'returncode': r.returncode, 'stdout': r.stdout[-12000:], 'stderr': r.stderr[-12000:]}
checks = {}
for name in ['torch', 'torchvision', 'transformers', 'accelerate', 'vllm', 'deepspeed', 'flash_attn',
             'faiss', 'av', 'decord', 'torchcodec', 'qwen_vl_utils']:
    checks[name] = run([sys.executable, '-B', '-c',
                       f'import {name}; import torch; assert not torch.cuda.is_initialized(); print("CPU_IMPORT_OK")'])
checks['torch_build'] = run([sys.executable, '-B', '-c',
    'import torch; assert torch.version.cuda == "12.8" and torch._C._GLIBCXX_USE_CXX11_ABI; print(torch.__version__,torch.version.cuda)'])
checks['pip_check'] = run([sys.executable, '-m', 'pip', 'check'])
passed = not missing and 'videosearch-r1' in installed and all(r['returncode'] == 0 for r in checks.values())
result = {'timestamp': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'passed': passed,
          'python': sys.version, 'pins_total': len(pins), 'mismatches': missing, 'checks': checks,
          'installed': installed, 'gpu_model_created': False}
(ROOT/'reports/03_runtime_validation.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'passed': passed, 'mismatches': missing,
                  'failed_checks': {k:v for k,v in checks.items() if v['returncode']}}, indent=2))
sys.exit(0 if passed else 1)
