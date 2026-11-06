#!/usr/bin/env python3
"""CPU reconstruction of two localization failures remaining after clock repair.

Diagnostic visual review only: no new model predictions, no weights/GPU,
no GT changes and no Markdown output. Selection is fixed before viewing frames.
"""
from __future__ import annotations
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import textwrap
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image,ImageDraw

from charades_timestamp_eval import install_clock,load_official,tensor_sha
from inspect_charades_localization_cases import ROOT,DATA,BASE,jsonl,sha,save,font,recover_normalized_frames,repack_frames
from run_charades_timestamp_full import verify


def draw_sheet(case,arrays,directory):
    width,columns,cell_h,head=1500,6,240,325
    cell_w=width//columns
    image=Image.new('RGB',(width,head+math.ceil(len(arrays)/columns)*cell_h+20),'white')
    d=ImageDraw.Draw(image)
    d.text((16,10),f'Charades {case["index"]} / {case["video_id"]}: remaining localization failure',fill='black',font=font(22))
    for j,line in enumerate(textwrap.wrap(case['query'],130)):
        d.text((16,46+23*j),line,fill='black',font=font(16))
    d.text((16,108),f'CPU-reconstructed processor RGB; {len(arrays)} frames; source-time input; no new inference.',fill='#444444',font=font(15))
    colors={'GT':'#258445','A_nominal':'#2563bb','B_source':'#a137aa'}
    maximum=max(case['source_duration'],case['annotation_duration'],*[s[1] for s in case['spans'].values()])
    for j,(name,span) in enumerate(case['spans'].items()):
        y=145+j*42;left,right=280,width-30
        d.text((16,y),f'{name}: {span[0]:g}-{span[1]:g}s',fill=colors[name],font=font(17))
        d.line((left,y+14,right,y+14),fill='#dddddd',width=3)
        d.rectangle((left+(right-left)*span[0]/maximum,y+3,left+(right-left)*span[1]/maximum,y+25),fill=colors[name])
        for t in case['source_times']:
            x=left+(right-left)*t/maximum;d.line((x,y+29,x,y+33),fill='#888888')
    d.text((16,280),f'IoU: A={case["A_iou"]:.4f}; B={case["B_iou"]:.4f}. Captions: source time / B temporal-pair label.',fill='black',font=font(15))
    for i,array in enumerate(arrays):
        x=i%columns*cell_w;y=head+i//columns*cell_h
        frame=Image.fromarray(array);frame.thumbnail((cell_w-10,180),Image.Resampling.LANCZOS)
        image.paste(frame,(x+(cell_w-frame.width)//2,y))
        t=case['source_times'][i];label=case['B_labels'][i//2]
        d.text((x+5,y+185),f'i={i:02d} src={t:.2f}s / label={label}',fill='black',font=font(12))
        for j,(name,span) in enumerate(case['spans'].items()):
            on=span[0]<=t<=span[1]
            d.rectangle((x+5+j*81,y+208,x+82+j*81,y+227),fill=colors[name] if on else '#eeeeee')
            d.text((x+8+j*81,y+208),name.split('_')[0],fill='white' if on else '#999999',font=font(12))
    image.save(directory/'all_frames.png')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Keep CUDA hidden via scripts/env.sh')
    verify(ROOT/'reports/43_charades_timestamp_full_cpu.json')
    run=json.loads((ROOT/'reports/44_charades_timestamp_full.json').read_text())
    if run['status']!='complete':raise ValueError('Require both completed full conditions')
    groups={c:jsonl(Path(s['predictions'])) for c,s in run['conditions'].items()}
    for c,s in run['conditions'].items():
        if sha(s['predictions'])!=s['prediction_sha256']:raise ValueError('Prediction binding changed')
    annotations=jsonl(DATA/'raw_annotation/test.jsonl')
    metadata={r['video_id']:r for r in jsonl(DATA/'test/video_npy_with_meta/meta.jsonl')}
    def accepted(r):
        return len(r['turns'])==1 and r['turns'][-1]['top1_is_gt'] and r['final_answer']=='matched' and r['final_time_parse_ok']
    broad=[]
    for a,b in zip(groups['nominal'],groups['source']):
        gt_len=b['gt_end']-b['gt_start'];row=metadata[b['gt_pos_doc_id']]
        if (accepted(a) and accepted(b) and a['final_iou_r1']<.7 and .3<=b['final_iou_r1']<.5
                and b['final_pred_end']-b['final_pred_start']>2*gt_len
                and b['gt_end']<=min(annotations[b['index']]['duration'],row['source_duration_sec'])):
            broad.append(b['index'])
    selected=[2233,min(broad)]
    assert selected==[2233,81]
    stamp=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    output=ROOT/f'outputs/charades_remaining_cases_{stamp}';output.mkdir()
    selection={'indices':selected,'selected_before_visual_review':True,
        'rules':{'2233':'Previously inspected drinking failure; persists/regresses after source-time correction.',
                 '81':'Minimum original index among one-turn GT-accepted A/B failures, B IoU in [0.3,0.5), B duration >2x GT, GT within recorded duration.'},
        'limits':'Purposive diagnostic cases, not representative error frequencies. Released source-time transport corrected, but raw-video PTS and annotation provenance not newly verified.',
        'source_sha256':{c:s['prediction_sha256'] for c,s in run['conditions'].items()}}
    save(output/'selection.json',selection)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    from vllm.model_executor.models.qwen3_vl import Qwen3VLProcessingInfo
    module=load_official('source')
    processor=module.AutoProcessor.from_pretrained(str(ROOT/'checkpoints/VideoSearchR1__charades-stage2'),use_fast=True,padding_side='left',local_files_only=True)
    audit=jsonl(Path(run['conditions']['source']['input_audit']))
    cases=[]
    for i in selected:
        a,b=groups['nominal'][i],groups['source'][i]
        assert accepted(a) and accepted(b)
        vid=b['gt_pos_doc_id'];row=metadata[vid];path=DATA/row['output_npy_path'];before=sha(path)
        pair={}
        for clock in ['nominal','source']:
            install_clock(clock)
            index=module.load_video_meta_index(str(DATA/'test/video_npy_with_meta/meta.jsonl'))
            inputs=module._build_generation_inputs(processor,module._make_system_prompt('<REFINE>'),b['query'],str(path),module.resolve_video_meta_for_video_path(str(path),index),20384,16,100352,602112,90316800)
            assert inputs['video_kwargs']['do_sample_frames'] is False
            batch=module._build_generation_batch_from_inputs(processor,inputs,20384)
            meta=inputs['video_metadatas'][0]
            times=processor._calculate_timestamps(list(meta['frames_indices']),meta['fps'],2)
            assert times==Qwen3VLProcessingInfo._calculate_timestamps(None,list(meta['frames_indices']),meta['fps'],2)
            text=processor.tokenizer.decode(batch['input_ids'][0],skip_special_tokens=False)
            labels=re.findall(r'<([0-9.]+) seconds>',text)
            assert labels==[f'{t:.1f}' for t in times]
            assert batch['input_ids'].shape[1]<20384
            pair[clock]={'pixels':batch['pixel_values_videos'].clone(),'grid':batch['video_grid_thw'].clone(),
                         'labels':labels,'text':text,'tokens':int(batch['input_ids'].shape[1])}
            if clock=='source':
                events=[e for e in audit if e['video_id']==vid and e['query']==b['query']]
                assert events and all(e['time_labels']==labels and e['video_tensor_sha256']==[tensor_sha(v) for v in inputs['video_inputs']] for e in events)
        assert torch.equal(pair['nominal']['pixels'],pair['source']['pixels'])
        assert torch.equal(pair['nominal']['grid'],pair['source']['grid'])
        vp=processor.video_processor;grid=pair['source']['grid'][0].tolist();patches=pair['source']['pixels']
        normalized=recover_normalized_frames(patches,grid,vp.patch_size,vp.temporal_patch_size,vp.merge_size)
        assert torch.equal(repack_frames(normalized,vp.patch_size,vp.temporal_patch_size,vp.merge_size),patches)
        mean=torch.tensor(vp.image_mean).reshape(1,3,1,1);std=torch.tensor(vp.image_std).reshape(1,3,1,1)
        rgb=((normalized*std+mean)/vp.rescale_factor).round().clamp(0,255).to(torch.uint8).permute(0,2,3,1).numpy()
        assert len(rgb)==row['output_num_frames']
        d=output/f'case_{i:06d}_{vid}';d.mkdir();(d/'frames').mkdir()
        for j,array in enumerate(rgb):Image.fromarray(array).save(d/'frames'/f'{j:03d}.png')
        case={'index':i,'video_id':vid,'query':b['query'],
            'spans':{'GT':[b['gt_start'],b['gt_end']],'A_nominal':[a['final_pred_start'],a['final_pred_end']],
                     'B_source':[b['final_pred_start'],b['final_pred_end']]},
            'A_iou':a['final_iou_r1'],'B_iou':b['final_iou_r1'],
            'source_times':row['sampled_timestamps_sec'],'B_labels':pair['source']['labels'],
            'annotation_duration':annotations[i]['duration'],'source_duration':row['source_duration_sec'],
            'input_tokens':pair['source']['tokens'],'visual_tokens':math.prod(grid)//vp.merge_size**2,
            'processed_shape':list(rgb.shape),'video_grid_thw':grid,
            'npy_sha256':before,'processed_patch_sha256':tensor_sha(patches),
            'runtime_source_audit_matches_reconstruction':True,'A_B_pixel_tensor_exact':True,
            'patch_inverse_roundtrip_exact':True,'do_sample_frames':False,
            'both_one_turn_GT_accepted':True,'source_prediction':b,'nominal_prediction':a,
            'directory':str(d),'display_policy':'Processor RGB, inverse normalization and uint8 rounding only; no enhancement, crop or generated content.'}
        save(d/'case.json',case);(d/'source_prompt.txt').write_text(pair['source']['text'])
        draw_sheet(case,rgb,d)
        assert sha(path)==before
        cases.append(case)
        print(f'[case] {i}/{vid}: {len(rgb)} frames, {case["visual_tokens"]} visual tokens; runtime pixels/timestamps verified.',flush=True)
        del pair,normalized,patches,rgb
    assert not torch.cuda.is_initialized()
    report={'status':'REMAINING_CASE_INPUT_RECONSTRUCTION_PASSED','output_dir':str(output),
        'model_execution':False,'gpu_access':False,'cuda_initialized':False,
        'selection':selection,'cases':cases,'manual_visual_review_completed':False,
        'script_sha256':sha(Path(__file__))}
    save(output/'analysis.json',report)
    print(json.dumps({'status':report['status'],'output':str(output)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
