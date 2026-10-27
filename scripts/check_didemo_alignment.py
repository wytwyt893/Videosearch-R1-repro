#!/usr/bin/env python3
"""Check the real shell argument chain with a recorder replacing model Python.

Uses the standard library only. No GPU probing, downloads, model imports, or
inference. Historical smoke reports and official source files remain intact.
"""
from __future__ import annotations

import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PINNED_COMMIT = '69234fd9783b7235164d9f92b8903d9781dc284e'
PROFILE = ROOT / 'configs/didemo_stage2_eval.bash'
EVALUATOR = ROOT / 'upstream/scripts/internal/stage1/eval_verified_test_rerank_temporal_grounding.py'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def constant(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -constant(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return constant(node.left) * constant(node.right)
    raise ValueError(ast.dump(node))


def parse_worker_argv(captured):
    argv = captured['argv']
    if Path(argv[0]).resolve() != EVALUATOR:
        raise ValueError(f'Unexpected evaluator: {argv[0]}')
    if (len(argv) - 1) % 2:
        raise ValueError('Expected name/value argument pairs')
    out = dict(zip(argv[1::2], argv[2::2]))
    if len(out) != (len(argv) - 1) // 2:
        raise ValueError('Duplicate worker arguments')
    return out


def main():
    commit = subprocess.check_output(['git', '-C', str(ROOT / 'upstream'),
                                      'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT / 'upstream'),
                                     'status', '--porcelain'], text=True).strip()
    if commit != PINNED_COMMIT or dirty:
        raise ValueError('Official checkout changed; re-audit before using this profile')
    subprocess.run(['bash', '-n', str(PROFILE)], check=True)

    # Intentionally hostile inherited values prove the profile wins over stale
    # interactive-shell settings. The interpreter below only records argv.
    env = dict(os.environ, MAX_TURN='17', REFINE_TOKEN_COUNT='1', MAX_SAMPLES='5',
               USE_UPDATED_QUERY='True', QFINAL_POOLING='mean',
               NUM_PROCESSES_PER_GPU='7', RESUME_FROM_JSONL='True',
               VIDEOSEARCH_VER_TURN='first', VIDEOSEARCH_VER_DENOMINATOR='unexpected',
               VIDEOSEARCH_TERMINAL_REFINEMENT='skip',
               QUERY_EMBEDDER_INPUT_PREFIX='unexpected inherited prefix')
    with tempfile.TemporaryDirectory(prefix='didemo_config_', dir='/tmp') as temporary:
        temp = Path(temporary)
        capture = temp / 'worker_args.json'
        recorder = temp / 'record_python'
        recorder.write_text(
            f'#!{sys.executable}\n'
            'import json, os, pathlib, sys\n'
            'pathlib.Path(os.environ["CHARADES_ARGV_CAPTURE"]).write_text('
            'json.dumps({"argv": sys.argv[1:], '
            '"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), '
            '"local_policy": {k: os.environ.get(k) for k in '
            '("VIDEOSEARCH_VER_TURN", "VIDEOSEARCH_VER_DENOMINATOR", '
            '"VIDEOSEARCH_TERMINAL_REFINEMENT")}}))\n'
        )
        recorder.chmod(0o700)
        env['CHARADES_ARGV_CAPTURE'] = str(capture)
        # All paths are positional arguments, not interpolated shell text.
        shell = '''
set -euo pipefail
source "$1"
source "$2"
export PYTHON_BIN="$3"
export VIDEOSEARCH_OUTPUT_ROOT="$4/output"
export LOGS_DIR="$4/logs"
export EVAL_GPUS=0
exec bash "$5" didemo --checkpoint "$VIDEOSEARCH_STAGE2_CHECKPOINT"
'''
        process = subprocess.run(
            ['bash', '--noprofile', '--norc', '-c', shell, 'alignment-check',
             str(ROOT / 'scripts/env.sh'), str(PROFILE), str(recorder), str(temp),
             str(ROOT / 'upstream/scripts/inference/inference.bash')],
            cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=30,
        )
        if process.returncode:
            raise RuntimeError(process.stdout)
        captured = json.loads(capture.read_text())
        actual = parse_worker_argv(captured)

    data = ROOT / 'data/didemo'
    expected = {
        '--model_path': str(ROOT / 'checkpoints/VideoSearchR1__didemo-stage2'),
        '--processor_path': str(ROOT / 'checkpoints/VideoSearchR1__didemo-stage2'),
        '--query_embedder_model_path': str(ROOT / 'checkpoints/Qwen__Qwen3-VL-Embedding-2B'),
        '--verified_test_jsonl': str(data / 'raw_annotation/test.jsonl'),
        '--video_root': str(data / 'test/video_npy_with_meta'),
        '--video_meta_path': str(data / 'test/video_npy_with_meta/meta.jsonl'),
        '--query_embeddings_path': str(data / 'test/query_embedding/query_embeddings.test.npy'),
        '--query_meta_path': str(data / 'test/query_embedding/query_meta.test.jsonl'),
        '--video_embeddings_path': str(data / 'test/video_embedding_1fps/segment_embeds.npy'),
        '--video_docid2row_path': str(data / 'test/video_embedding_1fps/docid2row.json'),
        '--max_turn': '2', '--refine_token_count': '8', '--temporal': 'True',
        '--topk': '1,5,10,100', '--iou_thresholds': '0.3,0.5,0.7',
        '--max_samples': '0', '--eval_limit_ratio': '1.0', '--rerank_topk': 'all',
        '--use_updated_query': 'False', '--use_query_embedder_path': 'True',
        '--use_refine_gate': 'False', '--refine_token': '<REFINE>',
        '--qfinal_pooling': 'latent_last', '--qfinal_normalize': 'True',
        '--query_embedder_max_length': '128',
        '--query_embedder_input_prefix': "Represent the user's input",
        '--model_max_length': '20384', '--max_new_tokens': '1024',
        '--bf16': 'True', '--seed': '42', '--use_vllm': 'True',
        '--vllm_model_path': '', '--vllm_tensor_parallel_size': '1',
        '--vllm_max_num_seqs': '1', '--vllm_eval_batch_size': '1',
        '--vllm_disable_custom_all_reduce': 'False',
        '--vllm_gpu_memory_utilization': '0.45', '--vllm_enforce_eager': 'True',
        '--save_query_vectors': 'False', '--profile_rollout_depths': '',
    }
    mismatches = {key: {'expected': value, 'actual': actual.get(key)}
                  for key, value in expected.items() if actual.get(key) != value}
    if mismatches:
        raise ValueError(json.dumps(mismatches, indent=2))
    if (actual.get('--num_shards') != '1' or actual.get('--shard_id') != '0'
            or actual.get('--resume_from_jsonl') != 'False'):
        raise ValueError('Profile did not enforce a fresh, single-worker configuration')
    local_policy = {'VIDEOSEARCH_VER_TURN': 'last_verified',
                    'VIDEOSEARCH_VER_DENOMINATOR': 'valid_answers',
                    'VIDEOSEARCH_TERMINAL_REFINEMENT': 'keep_official'}
    if captured.get('local_policy') != local_policy:
        raise ValueError('Selected VER / terminal-refinement protocol was overridden')

    tree = ast.parse(EVALUATOR.read_text())
    visual_defaults = {}
    sampling = None
    explicit_vllm_seed = None
    visual_expected = {'--image_patch_size': 16, '--video_min_pixels': 128 * 28 * 28,
                       '--video_max_pixels': 768 * 28 * 28,
                       '--video_total_pixels': 115200 * 28 * 28}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in visual_expected):
            default = next(k.value for k in node.keywords if k.arg == 'default')
            visual_defaults[node.args[0].value] = constant(default)
        if isinstance(node.func, ast.Name) and node.func.id == 'SamplingParams':
            sampling = {k.arg: constant(k.value) for k in node.keywords
                        if k.arg in ('n', 'temperature', 'top_p', 'top_k', 'repetition_penalty')}
        if isinstance(node.func, ast.Name) and node.func.id == 'LLM':
            explicit_vllm_seed = any(k.arg == 'seed' for k in node.keywords)
    if visual_defaults != visual_expected:
        raise ValueError(f'Visual defaults changed: {visual_defaults}')
    if sampling != {'n': 1, 'temperature': 0.0, 'top_p': 1.0,
                    'top_k': -1, 'repetition_penalty': 1.0}:
        raise ValueError(f'Generation settings changed: {sampling}')
    if explicit_vllm_seed is not False:
        raise ValueError('vLLM seed handling changed; re-audit the policy')

    model = json.loads((Path(actual['--model_path']) / 'config.json').read_text())
    embedder = json.loads((Path(actual['--query_embedder_model_path']) / 'config.json').read_text())
    snapshots = {'verified': True, 'models': [
        json.loads((ROOT / 'cache/didemo_download/models.json').read_text()),
        next(s for s in json.loads((ROOT / 'reports/03_models.json').read_text())['models']
             if s['repo'] == 'Qwen/Qwen3-VL-Embedding-2B')]}
    revisions = {'VideoSearchR1/didemo-stage2': 'a576c16040c35a5ec71468c66ec7628e7db71997',
                 'Qwen/Qwen3-VL-Embedding-2B': '9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda'}
    if not snapshots['verified']:
        raise ValueError('Prior official snapshot verification did not pass')
    for snapshot in snapshots['models']:
        if snapshot['revision'] != revisions[snapshot['repo']]:
            raise ValueError(f'Unexpected snapshot revision: {snapshot["repo"]}')
        for entry in snapshot['files']:
            path = Path(snapshot['path']) / entry['path']
            if path.stat().st_size != entry['bytes']:
                raise ValueError(f'Snapshot file size changed: {path}')
            if entry['path'] != 'model.safetensors' and sha256(path) != entry['sha256']:
                raise ValueError(f'Snapshot configuration/tokenizer changed: {path}')
    architecture = {'num_hidden_layers': 28, 'hidden_size': 2048,
                    'intermediate_size': 6144, 'num_attention_heads': 16,
                    'num_key_value_heads': 8}
    for name, config in [('vlm', model), ('retriever', embedder)]:
        if any(config['text_config'].get(k) != v for k, v in architecture.items()):
            raise ValueError(f'{name} does not match the audited 2B architecture')
    metadata = [json.loads(line) for line in
                (data / 'test/video_npy_with_meta/meta.jsonl').read_text().splitlines() if line.strip()]
    annotations = [json.loads(line) for line in
                   (data / 'raw_annotation/test.jsonl').read_text().splitlines() if line.strip()]
    if len(metadata) != 1002 or len(annotations) != 4013:
        raise ValueError('Official test cardinality changed')
    if any(r['target_sample_fps'] != 1.0 or r['target_max_frames'] != 64
           or not 0 < r['output_num_frames'] <= 64 for r in metadata):
        raise ValueError('Official frame metadata no longer matches 1 FPS / max 64')
    stable_arguments = {k: v for k, v in actual.items()
                        if k not in ('--output_json', '--output_jsonl', '--save_query_vectors_dir')}
    report = {
        'timestamp': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'configuration_checks_passed': True,
        'claim': 'Paper-explicit inference settings match; unspecified choices are frozen to the official baseline, with declared uncertainties.',
        'not_a_claim': 'This is not proof of exact paper training lineage, complete data availability, or reproduced Table 2 scores.',
        'model_execution': False, 'network_access': False, 'gpu_access': False,
        'upstream_commit': commit, 'upstream_clean': not dirty,
        'profile': str(PROFILE.relative_to(ROOT)), 'profile_sha256': sha256(PROFILE),
        'official_evaluator_sha256': sha256(EVALUATOR),
        'snapshot_revisions': revisions,
        'snapshot_check_scope': 'Recheck configuration/tokenizer hashes and all sizes against verified DiDeMo and retriever receipts; current weight SHA256 is checked by CPU preflight.',
        'worker_arguments': stable_arguments, 'verified_argument_count': len(expected),
        'local_policy': local_policy,
        'primary_scoring_entry': 'scripts/score_didemo_protocol.py',
        'primary_scoring_entry_sha256': sha256(ROOT / 'scripts/score_didemo_protocol.py'),
        'experimental_configuration_decided': True,
        'visual_pixel_defaults': visual_defaults,
        'video_metadata_rows': len(metadata), 'annotation_rows': len(annotations),
        'visual_policy': 'Use released 1 FPS / max-64-frame tensors and official inference resizing; no inferred 4096-token inference cap.',
        'generation_sampling': sampling, 'vllm_seed_explicitly_passed': explicit_vllm_seed,
        'vllm_seed_observed_in_prior_smoke': 0,
        'resolved_policy_decisions': [{
            'topic': '4096 visual tokens at inference',
            'paper_scope': 'The paper sets 4096 visual tokens during training; it does not explicitly impose a 4096-token inference ceiling.',
            'decision': 'Keep the official inference resizing and processor; do not add a 4096-token cap.',
            'basis': 'User instruction on 2026-10-26: only impose the cap if explicitly required for inference by the paper.',
            'official_inference_modified': False,
        }, {
            'topic': 'Primary VER',
            'decision': 'Use each query\'s last actual verification, paired with that checked video. Retain official valid-answer denominator and report coverage.',
            'basis': 'User selected last verification as primary; first-turn VER is an optional supplementary analysis.',
        }, {
            'topic': 'Terminal refinement',
            'decision': 'Keep the official final query update and reranking when SQR is triggered at turn 2. Do not add a third verification.',
            'basis': 'User selected terminal refinement for the main experiment; skipping it is a possible later ablation.',
            'official_inference_modified': False,
        }, {
            'topic': 'Checkpoint size',
            'decision': 'Keep the verified released 2B-architecture checkpoint unchanged.',
            'basis': 'User accepted actual checkpoint architecture; model-card provenance discrepancy is retained as metadata only.',
        }],
        'open_questions': [],
        'paper_provenance_caveats': [
            'Selecting last-turn VER and terminal refinement is the declared reproduction protocol, not proof that the paper used precisely this scoring convention.',
            'Released model card says 4B while actual config/tensors match 2B; exact training lineage is not established.',
        ],
    }
    destination = ROOT / 'reports/14_didemo_alignment.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('configuration_checks_passed', 'model_execution',
                     'verified_argument_count', 'annotation_rows', 'video_metadata_rows', 'open_questions')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
