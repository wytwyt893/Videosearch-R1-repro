#!/usr/bin/env python3
"""CPU-only reproducible engineering audit; never invokes inference/training."""
from __future__ import annotations
import argparse
import ast
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'upstream'
EXPECTED_SHA = '69234fd9783b7235164d9f92b8903d9781dc284e'


def run(command, timeout=60):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return {'command': command, 'exit_code': result.returncode,
            'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def test_helpers():
    """Demonstrate actual helper behavior, not findings in undownloaded data."""
    path = UPSTREAM / 'scripts/eval/write_temporal_result.py'
    spec = importlib.util.spec_from_file_location('official_metric_helpers', path)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    assert metrics.temporal_metrics([0, .5])['IoU@0.5@R1'] == .5
    assert metrics.temporal_metrics([0, None])['N'] == 2
    assert metrics.rank_metrics([1, 20])['R@1'] == .5
    # This is an observed denominator behavior requiring an integrity gate.
    missing_rank = metrics.rank_metrics([1, None])
    assert missing_rank['N'] == 1 and missing_rank['R@1'] == 1
    with tempfile.TemporaryDirectory(prefix='stage1_fixture_', dir=ROOT / 'cache/tmp') as scratch:
        fixture = Path(scratch) / 'predictions.jsonl'
        fixture.write_text('{"index":1,"final_rank":2}\nnot-json\n{"index":1,"final_rank":1}\n')
        loaded = metrics.load_rows(fixture)
        assert len(loaded) == 1 and loaded[0]['final_rank'] == 1
    source = UPSTREAM / 'scripts/internal/stage1/eval_verified_test_rerank_temporal_grounding.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_load_query_meta')
    fixture_rows = [{'query': 'same question', 'qid': 'q1', 'pos_doc_id': 'video_a'},
                    {'query': 'same question', 'qid': 'q2', 'pos_doc_id': 'video_b'}]
    namespace = {'Dict': dict, 'Tuple': tuple, '_iter_jsonl': lambda _: iter(fixture_rows)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    mapped = namespace['_load_query_meta']('fixture')
    assert len(mapped) == 1 and mapped['same question'] == (1, 'q2', 'video_b')
    return {'passed': True, 'checks': 6,
            'missing_rank_fixture': {'input_rows': 2, 'metric_denominator': missing_rank['N']},
            'malformed_and_duplicate_fixture': {'input_lines': 3, 'retained_rows': 1, 'last_duplicate_kept': True},
            'duplicate_question_mapping_fixture': {'input_rows': 2, 'mapped_questions': 1, 'last_qid': 'q2'},
            'scope': 'synthetic CPU fixtures, NOT evidence that actual benchmark data contains these anomalies'}


def check_setr():
    before = json.loads((ROOT / 'reports/setr_isolation_before.json').read_text())
    old = Path('/data/setr_repro')
    changed = [name for name, value in before['source_hashes'].items()
               if not (old / name).is_file() or digest(old / name) != value]
    versions = {}
    for name in before['environment_versions']:
        check = run([str(old / name / 'bin/python'), '-B', '-c',
                     'import json,importlib.metadata as m; a={d.metadata["Name"].lower():d.version for d in m.distributions()}; print(json.dumps({k:a[k] for k in ["torch","transformers","vllm","deepspeed"] if k in a}))'])
        if check['exit_code']:
            raise RuntimeError(check)
        versions[name] = json.loads(check['stdout'])
    result = {'passed': not changed and versions == before['environment_versions'],
              'checked_code_files': len(before['source_hashes']), 'changed_code_files': changed,
              'versions_before': before['environment_versions'], 'versions_after': versions,
              'scope': 'code hashes and core package versions; active SetR predictions/logs intentionally not frozen or stopped'}
    assert result['passed'], result
    return result


def import_probe():
    names = ['torch', 'transformers', 'vllm', 'deepspeed', 'flash_attn', 'faiss', 'av', 'decord', 'qwen_vl_utils']
    result = {}
    for name in names:
        code = f'import {name}; import torch; assert not torch.cuda.is_initialized(); print("CPU_IMPORT_OK")'
        result[name] = run([sys.executable, '-B', '-c', code], timeout=120)
    return {'passed': all(r['exit_code'] == 0 for r in result.values()), 'details': result,
            'gpu_engine_instantiated': False, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imports', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicitly hide GPUs for stage1'
    assert Path(sys.prefix).resolve() == ROOT / 'envs/videosearchr1', 'Wrong Python environment'
    head = run(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'])['stdout']
    status = run(['git', '-C', str(UPSTREAM), 'status', '--porcelain'])['stdout']
    assert head == EXPECTED_SHA and not status, 'Upstream commit or files changed'
    paths = [p for p in UPSTREAM.rglob('*.py') if '.git' not in p.parts]
    syntax_errors = []
    for path in paths:
        try:
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except Exception as exc:
            syntax_errors.append({'path': str(path.relative_to(ROOT)), 'error': str(exc)})
    shell = [p for p in UPSTREAM.rglob('*') if p.suffix in {'.bash', '.sh'} and '.git' not in p.parts]
    bash_errors = [r for p in shell if (r := run(['bash', '-n', str(p)]))['exit_code']]
    distributions = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    frozen = run([sys.executable, '-m', 'pip', 'freeze', '--all'])
    (ROOT / 'reports/environment.pip.freeze.txt').write_text(frozen['stdout'] + '\n')
    usage = shutil.disk_usage(ROOT)
    result = {'timestamp': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'scope': 'stage1 only; no benchmark data/weights downloaded, no inference/training',
              'commit_sha': head, 'upstream_clean': not status,
              'provenance': json.loads((ROOT / 'logs/01_upstream_provenance.json').read_text()),
              'python': {'executable': sys.executable, 'version': sys.version, 'prefix': sys.prefix},
              'disk': dict(zip(['total_bytes', 'used_bytes', 'free_bytes'], usage)),
              'gpu_snapshot': run(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu', '--format=csv,noheader']),
              'gpu_processes': run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader']),
              'topology': run(['nvidia-smi', 'topo', '-m']),
              'mount': run(['findmnt', '-T', str(ROOT), '-o', 'TARGET,SOURCE,FSTYPE,OPTIONS']),
              'host_memory': run(['free', '-h']), 'cuda_compiler': run(['/usr/local/cuda-12.8/bin/nvcc', '--version']),
              'package_versions': distributions,
              'syntax_checks': {'python_files': len(paths), 'bash_files': len(shell),
                                'python_errors': syntax_errors, 'bash_errors': bash_errors},
              'helper_fixtures': test_helpers(), 'setr_isolation': check_setr(),
              'bucket_cli_probe': run([str(ROOT / 'tools/bin/hf'), 'buckets', '--help']),
              'runtime_install_pause': json.loads((ROOT / 'logs/01_runtime_pause.json').read_text()),
              'project_script_sha256': {str(p.relative_to(ROOT)): digest(p) for folder in ['scripts', 'tools/bin']
                                        for p in (ROOT / folder).rglob('*') if p.is_file()},
              'benchmark_data_downloaded': False, 'checkpoint_downloaded': False,
              'gpu_model_or_engine_created': False,
              'source_sha256': {str(p.relative_to(UPSTREAM)): digest(p) for p in UPSTREAM.rglob('*')
                                if p.is_file() and '.git' not in p.parts and p.suffix in {'.py', '.bash', '.sh', '.toml', '.txt', '.json'}}}
    result['pip_check'] = run([sys.executable, '-m', 'pip', 'check'])
    if args.imports:
        result['import_probe'] = import_probe()
    result['engineering_checks_passed'] = not syntax_errors and not bash_errors and result['helper_fixtures']['passed'] and result['setr_isolation']['passed']
    # A clean skeleton is NOT a ready runtime; require every official pin.
    pins = [line.strip().split('==') for line in (UPSTREAM / 'requirements.txt').read_text().splitlines()
            if line.strip() and not line.strip().startswith('#') and '==' in line]
    lower = {k.lower().replace('_', '-'): v for k, v in distributions.items()}
    missing = {name: {'expected': version, 'actual': lower.get(name.lower().replace('_', '-'))}
               for name, version in pins if lower.get(name.lower().replace('_', '-'), '').split('+')[0] != version}
    result['runtime_requirements_mismatch'] = missing
    result['runtime_ready_cpu'] = not missing and result['pip_check']['exit_code'] == 0 and result.get('import_probe', {}).get('passed', False)
    save(ROOT / 'reports/01_engineering_audit.json', result)
    print(json.dumps({k: result[k] for k in ['timestamp', 'commit_sha', 'engineering_checks_passed', 'runtime_ready_cpu', 'runtime_requirements_mismatch', 'syntax_checks']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
