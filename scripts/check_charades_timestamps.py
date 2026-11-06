#!/usr/bin/env python3
"""CPU qualification and predeclared cohorts for a small clock-only A/B."""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
from zoneinfo import ZoneInfo

from charades_timestamp_eval import load_official, source_payload, tensor_sha
from inspect_charades_localization_cases import DATA, LOCAL, ROOT, jsonl, sha, save
from run_charades_full import dry_run, verify_preflight

REPORT = ROOT/'reports/40_charades_timestamp_cpu.json'
DIAGNOSTIC = [26, 870, 1252, 1516, 2233, 2344]
SELECTION_SEED = 20261105


def select_indices(count=3720):
    pool = [i for i in range(count) if i not in DIAGNOSTIC]
    random_indices = sorted(random.Random(SELECTION_SEED).sample(pool, 64))
    return {'diagnostic_cases': DIAGNOSTIC, 'random64_excluding_diagnostic': random_indices}


def run(report_path):
    if report_path.exists():
        raise ValueError('Preserve previous CPU qualification')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Source scripts/env.sh and keep CUDA hidden')
    preflight = verify_preflight()
    entry = dry_run()
    stamp = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    output = ROOT/f'outputs/charades_timestamp_cpu_{stamp}'
    output.mkdir()
    cohorts = select_indices()
    selected = sorted(sum(cohorts.values(), []))
    selection = {'cohorts': cohorts, 'original_indices': selected, 'random_seed': SELECTION_SEED,
                 'selection_precedes_new_model_inference': True,
                 'limits': 'Six purposive diagnostic examples and a separate random64 cohort from the other 3714 questions. Neither is a full benchmark.'}
    save(output/'selection.json', selection)
    skip = output/'selection_exclusions.jsonl'
    skip.write_text(''.join(json.dumps({'index': i})+'\n' for i in range(3720) if i not in selected))
    metadata_path = DATA/'test/video_npy_with_meta/meta.jsonl'
    rows = jsonl(metadata_path)
    per_video = []
    for row in rows:
        payload = source_payload(row)
        actual = [i/payload['raw_fps'] for i in payload['frames_indices']]
        assert len(actual) == row['output_num_frames']
        assert max(abs(a-b) for a,b in zip(actual,row['sampled_timestamps_sec'])) <= .50001e-6
        per_video.append({'video_id': row['video_id'], 'frames': len(actual),
                          'last_source_sec': actual[-1], 'last_nominal_sec': len(actual)-1,
                          'last_difference_sec': actual[-1]-(len(actual)-1)})
    assert len(per_video) == 1334
    save(output/'all_video_clock_audit.json', per_video)
    print('[CPU] All 1334 source-time lists are valid; original source index/FPS correspondence verified.', flush=True)

    import torch
    from vllm.model_executor.models.qwen3_vl import Qwen3VLProcessingInfo
    module = load_official('nominal')
    processor = module.AutoProcessor.from_pretrained(
        str(ROOT/'checkpoints/VideoSearchR1__charades-stage2'), use_fast=True,
        padding_side='left', local_files_only=True)
    predictions = jsonl(LOCAL)
    comparisons = []
    for index in DIAGNOSTIC:
        example = predictions[index]
        video = example['gt_pos_doc_id']
        path = DATA/f'test/video_npy_with_meta/{video}.npy'
        before = sha(path)
        pair = {}
        for clock in ['nominal','source']:
            from charades_timestamp_eval import install_clock
            install_clock(clock)
            meta_index = module.load_video_meta_index(str(metadata_path))
            inputs = module._build_generation_inputs(processor, module._make_system_prompt('<REFINE>'),
                example['query'], str(path), module.resolve_video_meta_for_video_path(str(path),meta_index),
                20384,16,100352,602112,90316800)
            assert inputs['video_kwargs']['do_sample_frames'] is False
            batch = module._build_generation_batch_from_inputs(processor, inputs, 20384)
            input_meta = inputs['video_metadatas'][0]
            hf = processor._calculate_timestamps(list(input_meta['frames_indices']),input_meta['fps'],2)
            vllm = Qwen3VLProcessingInfo._calculate_timestamps(None,list(input_meta['frames_indices']),input_meta['fps'],2)
            assert hf == vllm
            text = processor.tokenizer.decode(batch['input_ids'][0],skip_special_tokens=False)
            labels = re.findall(r'<([0-9.]+) seconds>',text)
            assert labels == [f'{t:.1f}' for t in hf]
            assert batch['input_ids'].shape[1] < 20384
            pair[clock] = {'pixels': batch['pixel_values_videos'].clone(),
                'grid': batch['video_grid_thw'].clone(), 'kwargs': copy.deepcopy(inputs['video_kwargs']),
                'unexpanded_text': inputs['text'], 'expanded_text': text,
                'input_tokens': int(batch['input_ids'].shape[1]), 'labels': labels,
                'pixel_sha256': tensor_sha(batch['pixel_values_videos'])}
            (output/f'{index}_{clock}_prompt.txt').write_text(text)
            del batch, inputs
        a,b = pair['nominal'],pair['source']
        assert torch.equal(a['pixels'],b['pixels']) and torch.equal(a['grid'],b['grid'])
        assert a['kwargs'] == b['kwargs'] and a['unexpanded_text'] == b['unexpanded_text']
        strip = lambda s: re.sub(r'<[0-9.]+ seconds>','<TIME seconds>',s)
        assert strip(a['expanded_text']) == strip(b['expanded_text'])
        assert sha(path) == before
        comparisons.append({'index': index, 'video_id': video, 'npy_sha256': before,
            'processed_pixel_sha256': a['pixel_sha256'], 'pixels_bit_exact': True,
            'video_grid_thw': a['grid'].tolist(), 'sampling_kwargs_equal': True,
            'expanded_prompt_only_time_labels_differ': True,
            'nominal_labels': a['labels'], 'source_labels': b['labels'],
            'nominal_input_tokens': a['input_tokens'], 'source_input_tokens': b['input_tokens']})
        print(f'[CPU] {index}/{video}: exact equal pixels/grid; prompt differs only in timestamps.',flush=True)
        del pair,a,b
    assert not torch.cuda.is_initialized()
    arguments = dict(entry['worker_arguments'])
    arguments['--resume_skip_jsonl'] = str(skip)
    files = [Path(__file__),ROOT/'scripts/charades_timestamp_eval.py',
             ROOT/'scripts/run_charades_timestamp_ab.py', metadata_path, output/'selection.json',skip,LOCAL,
             ROOT/'scripts/score_charades_protocol.py']
    report = {'status':'TIMESTAMP_CPU_CHECK_PASSED','created_at':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'model_execution':False,'gpu_access':False,'cuda_initialized':False,
        'official_source_modified':False,'output_dir':str(output),
        'preflight_input_signature':preflight['input_signature'],'selection':selection,
        'arguments':arguments,'expected_queries':len(selected),'expected_corpus':1334,
        'all_video_clock_audit':str(output/'all_video_clock_audit.json'),
        'pixel_comparisons':comparisons,
        'timestamp_semantics':'Released sampled_indices_src/raw_fps, validated against sampled_timestamps_sec; no new container-PTS decoding.',
        'source_clock_transport':'Released original source frame indices and raw FPS passed through directly; no resampling.',
        'held_fixed':['weights','pixel arrays','frame count/order','visual budgets','backend','generation parameters','T=2','N=8','full retrieval corpus','GT','scoring'],
        'changed':'Frame time labels from nominal sampled-array index/1FPS to released source times. Different tokenized time digits are intrinsic to this intervention.',
        'file_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}}
    save(report_path,report)
    print(json.dumps({'status':report['status'],'report':str(report_path),'expected_queries':len(selected),'output':str(output)},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=REPORT)
    run(parser.parse_args().report)
