#!/usr/bin/env python3
"""CPU-only qualification of frozen repaired-data views and actual processors."""
from __future__ import annotations
import datetime
import gc
import json
import math
import os
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

import numpy as np

from activitynet_repair_time import FIELD, timestamp_payload
from extract_video_frames_verified import sha256
from run_activitynet_full import verify_preflight, verify_final_smoke

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'data/activitynet_repair_v1'


def jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(report_path=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Source scripts/env.sh; CUDA must remain hidden')
    manifest_path = BASE/'cohort_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'COHORT_AND_PAIRED_DATA_VIEWS_FROZEN':
        raise ValueError('Freeze source decisions and cohort first')
    report_path = Path(report_path) if report_path else ROOT/'reports/32_activitynet_repair_cpu_preflight.json'
    if report_path.exists():
        raise ValueError('CPU preflight result already exists; preserve historical evidence')
    old = verify_preflight()
    verify_final_smoke(old)
    result = {'created_at': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'status': 'CHECKING', 'passed': False, 'model_execution': False, 'gpu_access': False,
              'manifest_sha256': sha256(manifest_path),
              'canonical_baseline_preflight_and_smoke_bindings_valid': True}
    for rel, digest in manifest['common_file_sha256'].items():
        if sha256(BASE/rel) != digest:
            raise ValueError(f'Common cohort file changed: {rel}')
    original = jsonl(ROOT/'data/activitynet/raw_annotation/test.jsonl')
    index_map = json.loads((BASE/'common/original_index_map.json').read_text())
    annotations = jsonl(BASE/'common/raw_annotation/test.jsonl')
    if len(index_map) != manifest['retained_queries'] or len(annotations) != len(index_map):
        raise ValueError('Question/index counts differ')
    for i, (entry, row) in enumerate(zip(index_map, annotations)):
        if (entry['subset_index'] != i or row != {**original[entry['original_index']], 'original_index': entry['original_index']}):
            raise ValueError('Original query text or labels changed in filtering')
    mapping = json.loads((BASE/'common/test/video_embedding_1fps/docid2row.json').read_text())
    if set(mapping.values()) != set(range(manifest['retained_videos'])):
        raise ValueError('Video mapping is not dense')
    qmeta = jsonl(BASE/'common/test/query_embedding/query_meta.test.jsonl')
    q = np.load(BASE/'common/test/query_embedding/query_embeddings.test.npy', allow_pickle=False)
    v = np.load(BASE/'common/test/video_embedding_1fps/segment_embeds.npy', allow_pickle=False)
    if q.shape != (len(annotations), 2048) or v.shape != (len(mapping), 2048) or not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError('Invalid filtered vectors')
    canonical_qmeta = jsonl(ROOT/'data/activitynet/test/query_embedding/query_meta.test.jsonl')
    old_q_rows = {r['qid']: i for i, r in enumerate(canonical_qmeta)}
    old_q = np.load(ROOT/'data/activitynet/test/query_embedding/query_embeddings.test.npy', mmap_mode='r', allow_pickle=False)
    if not np.array_equal(q, old_q[[old_q_rows[r['qid']] for r in qmeta]]):
        raise ValueError('Retained query vectors changed')
    old_mapping = json.loads((ROOT/'data/activitynet/test/video_embedding_1fps/docid2row.json').read_text())
    old_v = np.load(ROOT/'data/activitynet/test/video_embedding_1fps/segment_embeds.npy', mmap_mode='r', allow_pickle=False)
    ordered_videos = sorted(mapping, key=mapping.get)
    if not np.array_equal(v, old_v[[old_mapping[x] for x in ordered_videos]]):
        raise ValueError('Retained video vectors changed')
    import faiss
    index = faiss.read_index(str(BASE/'common/test/index/index.faiss'))
    if index.ntotal != len(mapping) or index.d != 2048:
        raise ValueError('Filtered FAISS index/cardinality mismatch')
    if json.loads((BASE/'common/test/index/id_map.json').read_text()) != ordered_videos:
        raise ValueError('Filtered FAISS id mapping mismatch')
    for video, record in manifest['accepted_source_records'].items():
        path = ROOT/record['npy_path']
        if sha256(path) != record['npy_sha256']:
            raise ValueError(f'Repaired NPY hash changed: {video}')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if (array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3 or
                not 2 <= array.shape[0] <= 64 or array.shape[0] % 2 or not array.flags.c_contiguous):
            raise ValueError(f'Invalid repaired pixel tensor: {video}')
    all_metadata = {}
    for name, view in manifest['variants'].items():
        data = Path(view['data_root'])
        rows = jsonl(data/'test/video_npy_with_meta/meta.jsonl')
        if {r['video_id'] for r in rows} != set(mapping):
            raise ValueError('Variant metadata corpus mismatch')
        if sha256(data/'test/video_npy_with_meta/meta.jsonl') != view['metadata_sha256']:
            raise ValueError('Variant metadata changed')
        for row in rows:
            path = data/'test/video_npy_with_meta'/(row['video_id']+'.npy')
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if list(array.shape) != [row['output_num_frames'], row['output_height'], row['output_width'], 3]:
                raise ValueError('Variant NPY shape/metadata mismatch')
            if view['source_timestamps']:
                timestamp_payload(row)
            elif FIELD in row:
                raise ValueError('Source-time flag leaked into official-timing comparison')
        all_metadata[name] = {r['video_id']: r for r in rows}
    result['data'] = {'queries': len(annotations), 'videos': len(mapping),
                      'repaired_arrays_sha_verified': len(manifest['accepted_source_records']),
                      'paired_query_and_corpus_identity_verified': True,
                      'original_labels_and_retained_vector_rows_unchanged': True}

    import torch
    assert not torch.cuda.is_initialized()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    from activitynet_repaired_eval import load_official
    module = load_official()
    from vllm.model_executor.models.qwen3_vl import Qwen3VLProcessingInfo
    processor = module.AutoProcessor.from_pretrained(
        str(ROOT/'checkpoints/VideoSearchR1__activitynet-stage2'), padding_side='left', local_files_only=True)
    tokens = processor.tokenizer([a['fig_desc'].strip() for a in annotations], add_special_tokens=False)['input_ids']
    longest_query = annotations[max(range(len(tokens)), key=lambda j: len(tokens[j]))]['fig_desc'].strip()
    del tokens
    repaired = list(manifest['accepted_source_records'])
    metadata = all_metadata['source_time_repaired_subset']
    largest = max(metadata, key=lambda x: metadata[x]['output_num_frames']*
                  round(metadata[x]['output_height']/32)*round(metadata[x]['output_width']/32))
    longest = max(metadata, key=lambda x: metadata[x]['source_duration_sec'])
    shortest = min(repaired, key=lambda x: metadata[x]['output_num_frames'])
    padded = next((x for x in repaired if metadata[x].get('even_pad_applied')), shortest)
    sample = 'v_KyrDumISv4A' if 'v_KyrDumISv4A' in metadata else repaired[0]
    selected = [('paired_example', sample), ('largest_visual', largest), ('longest_video', longest),
                ('fewest_repaired_frames', shortest), ('even_padding', padded)]
    checks = []
    for name, variant in manifest['variants'].items():
        data = Path(variant['data_root'])
        meta_index = module.load_video_meta_index(str(data/'test/video_npy_with_meta/meta.jsonl'))
        cases = selected if variant['source_timestamps'] else selected[:1]
        for role, video in cases:
            row = all_metadata[name][video]
            path = data/'test/video_npy_with_meta'/(video+'.npy')
            payload = module.resolve_video_meta_for_video_path(str(path), meta_index)
            inputs = module._build_generation_inputs(processor, module._make_system_prompt('<REFINE>'),
                longest_query, str(path), payload, 20384, 16, 100352, 602112, 90316800)
            if inputs['video_kwargs'].get('do_sample_frames') is not False:
                raise ValueError('Unexpected processor resampling')
            batch = processor(text=[inputs['text']], images=inputs['image_inputs'], videos=inputs['video_inputs'],
                              video_metadata=inputs['video_metadatas'], return_tensors='pt',
                              padding=True, truncation=False, **inputs['video_kwargs'])
            reported = inputs['video_metadatas'][0]
            hf = processor._calculate_timestamps(reported['frames_indices'], reported['fps'], 2)
            vl = Qwen3VLProcessingInfo._calculate_timestamps(None, reported['frames_indices'], reported['fps'], 2)
            expected_frames = row[FIELD] if variant['source_timestamps'] else list(range(row['output_num_frames']))
            expected = [(expected_frames[i]+expected_frames[i+1])/2 for i in range(0, len(expected_frames), 2)]
            if hf != vl or any(abs(a-b)>1e-6 for a, b in zip(hf, expected)) or len(hf) != len(expected):
                raise ValueError('HF/vLLM/source timestamp disagreement')
            rendered = processor.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
            prompt_times = re.findall(r'<([0-9.]+) seconds>', rendered)
            if prompt_times != [f'{x:.1f}' for x in hf]:
                raise ValueError('Actual rendered prompt time labels differ')
            visual = math.prod(batch['video_grid_thw'][0].tolist())//4
            token_count = int(batch['input_ids'].shape[1])
            if token_count > 20384 or visual != int(batch['input_ids'].eq(151656).sum()):
                raise ValueError('Prompt budget or visual-token count failure')
            checks.append({'variant': name, 'role': role, 'video': video,
                           'input_frames': row['output_num_frames'], 'visual_tokens': visual,
                           'input_tokens': token_count, 'last_model_timestamp': prompt_times[-1],
                           'source_last_timestamp': row['sampled_timestamps_sec'][-1],
                           'hf_vllm_timestamp_agreement': True, 'model_execution': False})
            del batch, inputs
            gc.collect()
    assert not torch.cuda.is_initialized()
    result.update(status='REPAIRED_DATA_CPU_PREFLIGHT_PASSED', passed=True, cuda_initialized=False,
                  processor_checks=checks, model_weights_loaded=False, gpu_smoke_required=True,
                  full_evaluation_started=False,
                  scope='Frozen source/data identity, paired corpora, actual HF processor and vLLM time helper. No model forward or engine.')
    if sha256(manifest_path) != result['manifest_sha256']:
        raise ValueError('Cohort changed during CPU verification')
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Fresh destination; historical results are never overwritten')
    main(parser.parse_args().output)
