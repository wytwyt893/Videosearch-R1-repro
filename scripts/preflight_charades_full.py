#!/usr/bin/env python3
"""CPU-only full-evaluation readiness audit; no inference or network downloads."""
from __future__ import annotations

from collections import Counter, defaultdict
import datetime
import gc
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from run_charades_full import ROOT, RUNTIME, PROFILE, load_environment, check_source, write_json, experiment_signature

DATA = ROOT / 'data/charades-sta'


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def header(path):
    with path.open('rb') as f:
        length = int.from_bytes(f.read(8), 'little')
        return json.loads(f.read(length))


def main():
    if Path(sys.prefix).resolve() != ROOT / 'envs/videosearchr1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Source scripts/env.sh and use the project Python; CUDA must be hidden')
    env = load_environment()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    stamp = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    output = ROOT / f'outputs/charades_cpu_preflight_{stamp}'
    output.mkdir()
    report = {'passed': False, 'status': 'checking', 'model_execution': False,
              'cuda_initialized': False, 'network_downloads': False,
              'log_dir': str(output), 'file_states': {}}
    destination = ROOT / 'reports/08_cpu_preflight.json'
    start = time.monotonic()

    def phase(name):
        report['current_check'] = name
        print(f'[CPU preflight] {name}', flush=True)

    def run(name, argv):
        with (output / f'{name}.log').open('w') as log:
            result = subprocess.run(argv, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed; inspect {output / (name + ".log")}')

    def record(path, expected_sha=None):
        path = Path(path)
        before = path.stat()
        with path.open('rb') as f:
            sha = hashlib.file_digest(f, 'sha256').hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'File changed while hashing: {path}')
        if expected_sha is not None and sha != expected_sha:
            raise ValueError(f'Hash mismatch: {path.relative_to(ROOT)}')
        report['file_states'][str(path.relative_to(ROOT))] = {
            'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns, 'sha256': sha}
        return sha

    try:
        phase('source and effective entry parameters')
        check_source()
        run('alignment', [str(RUNTIME), '-B', str(ROOT / 'scripts/check_charades_alignment.py')])
        alignment = json.loads((ROOT / 'reports/05_alignment_checks.json').read_text())
        report['entry_alignment'] = {'passed': alignment['configuration_checks_passed'],
                                     'arguments_verified': alignment['verified_argument_count'],
                                     'local_policy': alignment['local_policy']}

        phase('runtime dependency pins, pip check, CPU imports')
        installed = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in metadata.distributions()}
        pins = dict(line.split('==') for line in (ROOT / 'upstream/requirements.txt').read_text().splitlines()
                    if line.strip() and not line.startswith('#'))
        mismatches = {name: installed.get(name.lower().replace('_', '-')) for name, version in pins.items()
                      if installed.get(name.lower().replace('_', '-'), '').split('+')[0] != version}
        if mismatches or 'videosearch-r1' not in installed:
            raise ValueError(f'Runtime package mismatch: {mismatches}')
        run('pip_check', [str(RUNTIME), '-m', 'pip', 'check'])
        modules = ['torchvision', 'transformers', 'vllm', 'flash_attn', 'faiss', 'decord', 'av', 'torchcodec']
        imports = ('import importlib,torch;assert not torch.cuda.is_initialized();'
                   f'[importlib.import_module(m) for m in {modules!r}];'
                   'assert not torch.cuda.is_initialized();print("CPU_IMPORTS_OK")')
        run('cpu_imports', [str(RUNTIME), '-B', '-c', imports])
        report['runtime'] = {'passed': True, 'pins_total': len(pins), 'mismatches': {},
                             'python': sys.version, 'installed': installed, 'imports': modules}

        phase('original model and vLLM-compatible snapshot hashes')
        snapshots = json.loads((ROOT / 'reports/03_models.json').read_text())['models']
        model_headers = {}
        for snapshot in snapshots:
            target = Path(snapshot['path'])
            for entry in snapshot['files']:
                record(target / entry['path'], entry['sha256'])
            model_headers[snapshot['repo']] = header(target / 'model.safetensors')
        original = model_headers['VideoSearchR1/charades-stage2']
        refined = {k for k in original if k.startswith('refine_')}
        if len(refined) != 16:
            raise ValueError('Expected 16 saved SQR tensors')
        compatible = ROOT / 'checkpoints/VideoSearchR1__charades-stage2/_vllm_compat'
        evidence = json.loads((ROOT / 'reports/03_smoke_integrity.json').read_text())['runtime_compat_sha256']
        for filename, sha in evidence.items():
            if not filename.endswith('.partial'):
                record(compatible / filename, sha)
        compat_header = header(compatible / 'model.safetensors')
        original_keys = set(original) - {'__metadata__'}
        compat_keys = set(compat_header) - {'__metadata__'}
        if original_keys - compat_keys != refined or compat_keys - original_keys:
            raise ValueError('vLLM snapshot does not differ only by the 16 SQR tensors')
        if any(original[k]['shape'] != compat_header[k]['shape'] for k in compat_keys):
            raise ValueError('vLLM compatible weight shapes changed')
        report['models'] = {'passed': True, 'snapshots': [
            {'repo': s['repo'], 'revision': s['revision']} for s in snapshots],
            'refine_tensors': len(refined), 'vllm_snapshot_verified': True,
            'weight_checks': 'Full current SHA256 comparison; no model objects constructed.'}

        phase('complete test assets, NPY arrays, and source receipts')
        import numpy as np
        import faiss
        receipts = {r['path']: r for r in json.loads((ROOT / 'reports/03_data_receipts.json').read_text())}
        receipts.update({r['path']: r for r in jsonl(ROOT / 'reports/07_charades_test_video_receipts.jsonl')})
        nonvideo = ['raw_annotation/test.jsonl', 'test/query_embedding/query_embeddings.test.npy',
                    'test/query_embedding/query_meta.test.jsonl', 'test/video_embedding_1fps/segment_embeds.npy',
                    'test/video_embedding_1fps/docid2row.json', 'test/video_embedding_1fps/config.json',
                    'test/index/index.faiss', 'test/index/id_map.json', 'test/video_npy_with_meta/meta.jsonl']
        for relative in nonvideo:
            if relative not in receipts:
                raise ValueError(f'Missing source receipt: {relative}')
            record(DATA / relative, receipts[relative]['sha256'])
        annotations = jsonl(DATA / 'raw_annotation/test.jsonl')
        queries = jsonl(DATA / 'test/query_embedding/query_meta.test.jsonl')
        videos = {r['video_id']: r for r in jsonl(DATA / 'test/video_npy_with_meta/meta.jsonl')}
        mapping = json.loads((DATA / 'test/video_embedding_1fps/docid2row.json').read_text())
        pixels_report = json.loads((ROOT / 'reports/07_charades_test_videos.json').read_text())
        if len(annotations) != 3720 or len(queries) != 3720 or len(videos) != 1334 or set(mapping) != set(videos):
            raise ValueError('Incomplete test corpus')
        pixel_bytes = 0
        for n, (vid, meta) in enumerate(videos.items(), 1):
            path = DATA / f'test/video_npy_with_meta/{vid}.npy'
            record(path, pixels_report['file_sha256'][str(path.relative_to(DATA))])
            a = np.load(path, allow_pickle=False, mmap_mode='r')
            expected = (meta['output_num_frames'], meta['output_height'], meta['output_width'], 3)
            if (a.shape != expected or a.dtype != np.uint8 or not a.flags.c_contiguous
                    or a.shape[0] % 2 or not 0 < a.shape[0] <= 64
                    or meta['target_sample_fps'] != 1.0 or meta['target_max_frames'] != 64):
                raise ValueError(f'Pixel array mismatch: {vid}')
            a._mmap.close()
            pixel_bytes += path.stat().st_size
            if n % 200 == 0:
                print(f'[CPU preflight] {n}/1334 video arrays and hashes verified', flush=True)
        if set(p.stem for p in (DATA / 'test/video_npy_with_meta').glob('*.npy')) != set(mapping):
            raise ValueError('Unexpected extra/missing pixel arrays')
        if list((DATA / 'test/video_npy_with_meta').glob('*.partial')):
            raise ValueError('Unfinished video downloads')
        q = np.load(DATA / 'test/query_embedding/query_embeddings.test.npy', allow_pickle=False)
        v = np.load(DATA / 'test/video_embedding_1fps/segment_embeds.npy', allow_pickle=False)
        if q.shape != (3720, 2048) or v.shape != (1334, 2048) or not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError('Invalid embedding corpus')
        if not (np.linalg.norm(q, axis=1) > 0).all() or not (np.linalg.norm(v, axis=1) > 0).all():
            raise ValueError('Zero embedding row')
        if set(mapping.values()) != set(range(1334)):
            raise ValueError('Video row mapping is not a permutation')
        faiss_index = faiss.read_index(str(DATA / 'test/index/index.faiss'))
        id_map = json.loads((DATA / 'test/index/id_map.json').read_text())
        if faiss_index.ntotal != 1334 or faiss_index.d != 2048 or id_map != sorted(mapping, key=mapping.get):
            raise ValueError('FAISS corpus / id map mismatch')
        report['data'] = {'passed': True, 'annotations': 3720, 'video_arrays': 1334,
                          'pixel_bytes': pixel_bytes, 'query_shape': list(q.shape), 'video_shape': list(v.shape)}

        phase('all question mappings and original-label boundary observations')
        by_qid = {r['qid']: r for r in queries}
        text_map = {r['query'].strip(): (i, r) for i, r in enumerate(queries)}
        if len(by_qid) != len(queries):
            raise ValueError('Original embedding qids are not unique')
        q_rows, overrides, video_mismatches, duration_excess = [], [], [], []
        for index, row in enumerate(annotations):
            original_qid = str(row['desc_id']) + '_fig'
            meta = by_qid[original_qid]
            if meta['query'] != row['fig_desc'] or meta['pos_doc_id'] != row['video'] or row['video'] not in mapping:
                raise ValueError(f'Original question identity mismatch: {index}')
            begin, end = row['time']
            if not all(math.isfinite(x) for x in (begin, end, row['duration'])) or not 0 <= begin <= end:
                raise ValueError('Invalid original temporal interval')
            q_row, selected = text_map[row['fig_desc']]
            q_rows.append(q_row)
            if selected['qid'] != original_qid:
                overrides.append(index)
            if selected['pos_doc_id'] != row['video']:
                video_mismatches.append(index)
            if end > row['duration'] + 0.1:
                duration_excess.append({'index': index, 'video': row['video'], 'gt_end': end,
                                        'duration': row['duration'], 'excess': end - row['duration']})
        counts = Counter(r['query'] for r in queries)
        report['mappings'] = {'passed': True, 'original_qid_and_video_matches': 3720,
                              'duplicate_query_groups': sum(n > 1 for n in counts.values()),
                              'official_text_map_qid_overrides': overrides,
                              'official_text_map_metadata_video_mismatches': video_mismatches,
                              'policy': 'Retain official text lookup; each original annotation supplies GT video/time. Score by original index.'}
        report['data_observations'] = {
            'annotation_end_after_duration_count': len(duration_excess), 'tolerance_seconds': 0.1,
            'max_excess_seconds': max((r['excess'] for r in duration_excess), default=0),
            'examples': sorted(duration_excess, key=lambda r: r['excess'], reverse=True)[:5],
            'policy': 'Preserve official labels and prediction-time clipping; do not repair or remove benchmark examples.'}
        qn = q.astype(np.float32); qn /= np.linalg.norm(qn, axis=1, keepdims=True)
        vn = v.astype(np.float32); vn /= np.linalg.norm(vn, axis=1, keepdims=True)
        scores = qn[np.array(q_rows)] @ vn.T
        reverse = {i: vid for vid, i in mapping.items()}
        first_docs = [reverse[int(i)] for i in scores.argmax(axis=1)]
        initial_hits = sum(doc == row['video'] for doc, row in zip(first_docs, annotations))
        report['cpu_initial_retrieval_diagnostic'] = {'hits_at_1': initial_hits, 'N': 3720,
            'R@1': initial_hits / 3720, 'stage2_model_executed': False,
            'note': 'Initial published vectors only; CPU float32 backend, not final Stage2 results.'}
        del scores, qn, vn

        phase('real processor on CPU: visual-token and prompt-length boundaries')
        import torch
        assert not torch.cuda.is_initialized()
        source = ROOT / 'upstream/scripts/internal/stage1/eval_verified_test_rerank_temporal_grounding.py'
        spec = importlib.util.spec_from_file_location('charades_cpu_official_eval', source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        processor = module.AutoProcessor.from_pretrained(
            str(ROOT / 'checkpoints/VideoSearchR1__charades-stage2'), padding_side='left', local_files_only=True)
        if len(processor.tokenizer) != 151670 or processor.tokenizer.convert_tokens_to_ids('<REFINE>') >= 151670:
            raise ValueError('Tokenizer is not aligned to the checkpoint embedding rows')
        meta_index = module.load_video_meta_index(str(DATA / 'test/video_npy_with_meta/meta.jsonl'))
        largest = max(videos, key=lambda vid: videos[vid]['output_num_frames'] *
                      round(videos[vid]['output_height'] / 32) * round(videos[vid]['output_width'] / 32))
        shortest = min(videos, key=lambda vid: videos[vid]['output_num_frames'])
        padded = next(vid for vid in videos if videos[vid].get('even_pad_applied'))
        longest_query = max(annotations, key=lambda row: len(row['fig_desc']))['fig_desc']
        selected = [('largest_visual_input', largest, longest_query),
                    ('fewest_frames', shortest, longest_query),
                    ('even_padding', padded, longest_query),
                    ('original_question_zero', first_docs[0], annotations[0]['fig_desc'])]
        processor_checks = []
        for role, vid, query in selected:
            path = DATA / f'test/video_npy_with_meta/{vid}.npy'
            payload = module.resolve_video_meta_for_video_path(str(path), meta_index)
            inputs = module._build_generation_inputs(
                processor, module._make_system_prompt('<REFINE>'), query, str(path), payload,
                20384, 16, 100352, 602112, 90316800)
            batch = processor(text=[inputs['text']], images=inputs['image_inputs'],
                              videos=inputs['video_inputs'], video_metadata=inputs['video_metadatas'],
                              return_tensors='pt', padding=True, truncation=False, **inputs['video_kwargs'])
            grid = batch['video_grid_thw'][0].tolist()
            visual_tokens = math.prod(grid) // 4
            prompt_tokens = int(batch['input_ids'].shape[1])
            video_token_count = int(batch['input_ids'].eq(151656).sum().item())
            if (visual_tokens != video_token_count or prompt_tokens > 20384
                    or any(t.device.type != 'cpu' for t in batch.values() if isinstance(t, torch.Tensor))):
                raise ValueError('Processor/token-budget mismatch')
            if inputs['video_kwargs'].get('do_sample_frames') is not False:
                raise ValueError('Unexpected runtime frame resampling')
            processor_checks.append({'role': role, 'video': vid, 'grid_thw': grid,
                                     'visual_tokens': visual_tokens, 'prompt_tokens': prompt_tokens,
                                     'generation_allowance': 1024, 'engine_context_limit': 21408,
                                     'truncation_enabled': False})
            del batch, inputs
            gc.collect()
        assert not torch.cuda.is_initialized()
        report['processor'] = {'passed': True, 'samples': processor_checks,
            'scope': 'Actual local HF processor on CPU; no visual encoder forward pass or vLLM engine execution.'}

        phase('full entry dry run, real smoke re-scoring, and metric fixtures')
        run('full_entry', [str(RUNTIME), '-B', str(ROOT / 'scripts/run_charades_full.py'), '--dry-run',
                           '--output', str(output / 'full_entry.json')])
        report['full_entry'] = json.loads((output / 'full_entry.json').read_text())
        run('final_smoke_entry', [str(RUNTIME), '-B', str(ROOT / 'scripts/run_charades_full.py'),
                                  '--dry-run', '--smoke', '--output', str(output / 'final_smoke_entry.json')])
        smoke_entry = json.loads((output / 'final_smoke_entry.json').read_text())
        differences = {k for k, v in report['full_entry']['worker_arguments'].items()
                       if smoke_entry['worker_arguments'].get(k) != v}
        if differences != {'--max_samples'} or smoke_entry['expected_examples'] != 10:
            raise ValueError('Final smoke must differ from full execution only in the 10-query limit')
        report['final_smoke_entry'] = {'passed': True, 'expected_examples': 10,
                                       'only_worker_argument_difference': '--max_samples'}
        run('metric_tests', [str(RUNTIME), '-B', '-m', 'unittest', 'discover', '-s', 'tests',
                             '-p', 'test_charades_protocol.py', '-v'])
        run('timing_tests', [str(RUNTIME), '-B', '-m', 'unittest', 'discover', '-s', 'tests',
                             '-p', 'test_runner_timing.py', '-v'])
        smoke = ROOT / ('outputs/charades_smoke_20261023_213015/'
                        'external_verified_test_temporal_grounding_VideoSearchR1__charades-stage2.jsonl')
        run('smoke_rescore', [str(RUNTIME), '-B', str(ROOT / 'scripts/score_charades_protocol.py'),
                              str(smoke), str(output / 'smoke_primary.json'), '--expected-count', '5'])
        scored = json.loads((output / 'smoke_primary.json').read_text())
        if scored['primary']['VER']['N'] != 5 or scored['primary']['VER']['accuracy'] != 0.6:
            raise ValueError('Last-verification smoke scoring changed')
        report['scoring'] = {'passed': True, 'fixture_tests': 6, 'smoke_examples': 5,
                             'smoke_VER': scored['primary']['VER'], 'new_inference': False,
                             'full_run_requires_all_indices_once': True}
        report['timing_checks'] = {'passed': True, 'tests': 4, 'initialization_counted_once': True,
                                   'resource_polling_tested': True}
        for relative in ['scripts/env.sh', 'scripts/network.sh', 'configs/charades_stage2_eval.bash',
                         'scripts/run_charades_full.py', 'scripts/preflight_charades_full.py',
                         'scripts/score_charades_protocol.py', 'scripts/check_charades_alignment.py',
                         'upstream/requirements.txt', 'tests/test_runner_timing.py',
                         'tests/test_charades_protocol.py']:
            record(ROOT / relative)
        report['input_signature'] = hashlib.sha256(
            json.dumps(report['file_states'], sort_keys=True).encode()).hexdigest()
        report['experiment_signature'] = experiment_signature(report)
        report.update(passed=True, status='CPU_PREFLIGHT_PASSED', current_check='complete',
                      elapsed_seconds=round(time.monotonic() - start, 2),
                      not_verified=['GPU availability and UUID mapping', 'Full 3720-query model execution and final scores'])
        write_json(destination, report)
        print(json.dumps({'passed': True, 'status': report['status'], 'report': str(destination),
                          'processor_samples': processor_checks, 'data_observations': report['data_observations'],
                          'elapsed_seconds': report['elapsed_seconds']}, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        report.update(passed=False, status='FAILED', error=f'{type(error).__name__}: {error}',
                      elapsed_seconds=round(time.monotonic() - start, 2))
        write_json(destination, report)
        raise


if __name__ == '__main__':
    main()
