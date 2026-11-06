#!/usr/bin/env python3
"""Reconstruct and visualize selected Charades visual inputs on CPU.

Uses frozen official code/assets, with no model weights, forward pass or GPU.
Selection is frozen before rendering/review. Frames are recovered from the
HF video processor's patch tensor, rather than assumed to equal the raw NPY.
"""
from __future__ import annotations
import argparse
import base64
import csv
import datetime
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import textwrap
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE = 'external_verified_test_temporal_grounding_VideoSearchR1__charades-stage2'
LOCAL = ROOT/'outputs/charades_full_20261027_161353_178368'/(BASE+'.jsonl')
REFERENCE = ROOT/'cache/official_eval/charades-stage2/c78dc49c6b2cffdf9ae36bd01b738f9b9cdce283/external_verified_test_temporal_grounding.check.jsonl'
DATA = ROOT/'data/charades-sta'
SOURCE = ROOT/'upstream/scripts/internal/stage1/eval_verified_test_rerank_temporal_grounding.py'
COLORS = {'GT':'#258445', 'Local':'#2563bb', 'Author':'#cf7508'}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def recover_normalized_frames(patches, grid, patch_size, temporal_patch_size, merge_size):
    """Invert Qwen3-VL's actual temporal/spatial patch packing."""
    import torch
    gt, gh, gw = map(int,grid)
    p,t,m = patch_size,temporal_patch_size,merge_size
    shaped = patches.reshape(1,gt,gh//m,gw//m,m,m,3,t,p,p)
    return shaped.permute(0,1,7,6,2,4,8,3,5,9).contiguous().reshape(gt*t,3,gh*p,gw*p)


def repack_frames(frames, patch_size, temporal_patch_size, merge_size):
    f,c,h,w = frames.shape
    p,t,m = patch_size,temporal_patch_size,merge_size
    return (frames.reshape(1,f//t,t,c,h//p//m,m,p,w//p//m,m,p)
            .permute(0,1,4,7,5,8,3,2,6,9).reshape(-1,c*t*p*p))


def choose(local, reference, annotations):
    def accepted(r):
        last=r['turns'][-1]
        return last['top1_is_gt'] and last['answer']=='matched' and last['time_parse_ok']
    lost=[(a,b) for a,b in zip(local,reference) if accepted(a) and accepted(b)
          and b['final_iou_r1']>=.7 and a['final_iou_r1']<.7]
    selected=[(1252,'previously_fixed_extreme_duration'),(870,'previously_fixed_disjoint_shift'),
              (2233,'previously_fixed_partial_overlap')]
    fixed={i for i,_ in selected}
    edge=sorted([(a,b) for a,b in lost if a['index'] not in fixed and
                 annotations[a['index']]['time'][1]>annotations[a['index']]['duration']+.1],
                key=lambda p:(p[0]['final_iou_r1']-p[1]['final_iou_r1'],p[0]['index']))
    selected.append((edge[0][0]['index'],'annotation_end_exceeds_duration_largest_paired_drop'))
    fixed.add(selected[-1][0])
    near=sorted([(a,b) for a,b in lost if a['index'] not in fixed and .5<=a['final_iou_r1']<.69],
                key=lambda p:(-p[0]['final_iou_r1'],p[0]['index']))
    selected.append((near[0][0]['index'],'nearest_below_threshold_excluding_float_boundary'))
    controls=[a['index'] for a,b in zip(local,reference) if accepted(a) and accepted(b)
              and a['final_iou_r1']>=.7 and b['final_iou_r1']>=.7
              and 3.5<=a['gt_end']-a['gt_start']<=10
              and annotations[a['index']]['time'][1]<=annotations[a['index']]['duration']]
    selected.append((min(controls),'first_index_joint_success_short_event_control'))
    return selected


def font(size):
    for path in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf']:
        if Path(path).exists():return ImageFont.truetype(path,size)
    return ImageFont.load_default()


def contact_sheet(case, images, output):
    width,cell_w,cell_h,header = 1680,280,275,350
    n=len(images);height=header+math.ceil(n/6)*cell_h+20
    sheet=Image.new('RGB',(width,height),'white');d=ImageDraw.Draw(sheet)
    d.text((18,12),f'Charades index {case["index"]} | {case["video_id"]} | {case["role"]}',fill='black',font=font(21))
    for j,line in enumerate(textwrap.wrap(case['query'],130)):
        d.text((18,45+j*22),line,fill='black',font=font(17))
    d.text((18,103),f'CPU reconstruction of processor RGB; {n} frames; visual tokens={case["visual_tokens"]}; input tokens={case["input_tokens"]}',fill='#444444',font=font(15))
    d.text((18,128),'Frame titles: NPY index | source time | actual model temporal-pair label. Bottom bars indicate interval membership on source time.',fill='#444444',font=font(13))
    maximum=max(case['source_duration_sec'],case['annotation_duration'],*[span[1] for span in case['spans'].values()])
    left,right=170,width-40
    for j,(name,span) in enumerate(case['spans'].items()):
        y=172+j*37
        d.text((18,y),f'{name}: {span[0]:g}-{span[1]:g}s',fill=COLORS[name],font=font(17))
        d.line((left,y+15,right,y+15),fill='#dddddd',width=2)
        x0=left+(right-left)*span[0]/maximum;x1=left+(right-left)*span[1]/maximum
        d.rectangle((x0,y+4,x1,y+25),fill=COLORS[name])
        for t in case['source_timestamps_sec']:
            x=left+(right-left)*t/maximum;d.line((x,y+29,x,y+33),fill='#777777')
    axis_y=294
    for t in range(0,math.ceil(maximum)+1,5):
        x=left+(right-left)*t/maximum
        if x<=right:d.text((x-10,axis_y),str(t),fill='#444444',font=font(13))
    if case['annotation_duration']<case['spans']['GT'][1]:
        d.text((18,320),f'GT extends beyond annotation duration {case["annotation_duration"]:g}s; original labels retained.',fill='#b32323',font=font(15))
    for i,array in enumerate(images):
        x=(i%6)*cell_w;y=header+(i//6)*cell_h
        image=Image.fromarray(array);image.thumbnail((cell_w-12,210),Image.Resampling.LANCZOS)
        sheet.paste(image,(x+(cell_w-image.width)//2,y))
        src=case['source_timestamps_sec'][i];label=case['model_pair_labels_sec'][i//2]
        d.text((x+7,y+214),f'i={i:02d}  src={src:.2f}s  label={label:.1f}s',fill='black',font=font(13))
        for j,(name,span) in enumerate(case['spans'].items()):
            on=span[0]<=src<=span[1]
            d.rectangle((x+7+j*89,y+240,x+88+j*89,y+258),fill=COLORS[name] if on else '#eeeeee')
            d.text((x+12+j*89,y+240),name,fill='white' if on else '#999999',font=font(12))
    sheet.save(output/'contact_sheet.png')
    sheet.save(output/'contact_sheet.pdf',resolution=150)


def browser(cases, output):
    payload=[]
    for case in cases:
        c=dict(case);directory=output/c['case_directory'];c['images']=[]
        for image_path in sorted((directory/'frames').glob('*.png')):
            c['images'].append('data:image/png;base64,'+base64.b64encode(image_path.read_bytes()).decode())
        payload.append(c)
    template='''<!doctype html><meta charset="utf-8"><title>Charades 定位案例复核</title>
<style>body{font:16px system-ui;margin:24px;background:#f4f6f8;color:#17212b}select,button{font:inherit;padding:8px;margin:4px}main{display:grid;grid-template-columns:minmax(400px,1fr) 1fr;gap:24px}img{max-width:100%;max-height:65vh;display:block;margin:auto;background:#111}section{background:white;padding:18px;border-radius:10px}input{width:100%}.track{position:relative;background:#eee;height:26px;margin:8px 0}.bar{position:absolute;height:100%;opacity:.8}.cursor{position:absolute;height:100%;width:3px;background:#111}small{color:#53616d}#grid{display:none;grid-template-columns:repeat(6,1fr);gap:10px}#grid img{max-height:180px}.thumb{cursor:pointer;background:white;padding:4px;font-size:12px}</style>
<h1>Charades：模型输入帧与时间定位对照</h1><p>固定选样；使用冻结代码与资产在 CPU 重建 processor 输入，不加载模型权重。图像为 patch 张量还原的 RGB 显示；作者输入像素未公开核实。</p>
<select id="cases"></select><button id="play">播放/暂停</button><button id="toggle">展开/收起全部帧</button>
<h2 id="query"></h2><main><section><img id="frame"><input id="slider" type="range" min="0" value="0"><p id="time"></p><div id="jump"></div></section><section><div id="info"></div><div id="tracks"></div><p id="note"></p><p><b>GT / Local / Author</b> 以原始评测秒数绘制。光标使用发布元数据中的源采样时刻；label 是模型收到的两帧时间组标签。不同时间轴没有悄悄互相换算。</p><small>1 FPS 的稀疏采样不能证明相邻帧之间动作的精确开始或结束；肉眼观察只用于提出待验证假设。</small></section></main><div id="grid"></div>
<script id="data" type="application/json">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);const $=id=>document.getElementById(id);let c=data[0],timer=null;
data.forEach((x,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${x.index} / ${x.video_id} / ${x.role}`;$('cases').appendChild(o)});
function render(){let i=+$('slider').value,t=c.source_timestamps_sec[i];$('frame').src=c.images[i];$('time').textContent=`NPY帧 ${i}/${c.images.length-1} | 源采样时间 ${t.toFixed(3)}s | reader名义帧时刻 ${i}s | 模型两帧标签 ${c.model_pair_labels_sec[Math.floor(i/2)].toFixed(1)}s`;
document.querySelectorAll('.cursor').forEach(x=>x.style.left=(100*t/c.maximum_time)+'%')}
function load(){c=data[+$('cases').value];c.maximum_time=Math.max(c.source_duration_sec,c.annotation_duration,...Object.values(c.spans).flat());$('query').textContent=c.query;$('slider').max=c.images.length-1;$('slider').value=0;$('info').textContent=`GT ${c.spans.GT.join('–')}s；本地 ${c.spans.Local.join('–')}s / IoU ${c.local_iou.toFixed(3)}；作者 ${c.spans.Author.join('–')}s / IoU ${c.author_iou.toFixed(3)}。帧数 ${c.images.length}；视觉 tokens ${c.visual_tokens}。`;$('tracks').replaceChildren();$('jump').replaceChildren();$('grid').replaceChildren();
const colors={GT:'#258445',Local:'#2563bb',Author:'#cf7508'};
Object.entries(c.spans).forEach(([name,span])=>{let text=document.createElement('div');text.textContent=name+' '+span.join('–')+'s';$('tracks').appendChild(text);let track=document.createElement('div');track.className='track';let bar=document.createElement('div');bar.className='bar';bar.style.cssText=`left:${100*span[0]/c.maximum_time}%;width:${100*(span[1]-span[0])/c.maximum_time}%;background:${colors[name]}`;track.appendChild(bar);let cur=document.createElement('div');cur.className='cursor';track.appendChild(cur);$('tracks').appendChild(track);span.forEach((t,j)=>{let b=document.createElement('button');b.textContent=name+(j?'终点':'起点');b.onclick=()=>{let best=0;c.source_timestamps_sec.forEach((x,k)=>{if(Math.abs(x-t)<Math.abs(c.source_timestamps_sec[best]-t))best=k});$('slider').value=best;render()};$('jump').appendChild(b)})});
c.images.forEach((src,i)=>{let cell=document.createElement('div');cell.className='thumb';let img=document.createElement('img');img.src=src;cell.appendChild(img);let cap=document.createElement('div');cap.textContent=`i=${i} src=${c.source_timestamps_sec[i].toFixed(2)} label=${c.model_pair_labels_sec[Math.floor(i/2)].toFixed(1)}`;cell.appendChild(cap);cell.onclick=()=>{$('slider').value=i;render()};$('grid').appendChild(cell)});$('note').textContent=`annotation duration=${c.annotation_duration}s；source duration=${c.source_duration_sec.toFixed(3)}s；GT越界=${c.spans.GT[1]>c.annotation_duration+.1}`;render()}
$('cases').onchange=load;$('slider').oninput=render;$('toggle').onclick=()=>{$('grid').style.display=$('grid').style.display==='grid'?'none':'grid'};$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null}else timer=setInterval(()=>{$('slider').value=(+$('slider').value+1)%c.images.length;render()},700)};load();</script>'''
    (output/'case_browser.html').write_text(template.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False).replace('</','<\\/')))


def main(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Source scripts/env.sh; keep CUDA hidden')
    if output.exists():raise ValueError('Choose a fresh analysis directory')
    from run_charades_full import verify_preflight
    preflight=verify_preflight()
    local,reference,annotations=jsonl(LOCAL),jsonl(REFERENCE),jsonl(DATA/'raw_annotation/test.jsonl')
    assert sha(LOCAL)==json.loads((ROOT/'reports/10_charades_full_summary.json').read_text())['prediction_sha256']
    assert sha(REFERENCE)=='ac2256b6a5d882df706cc8b0730b22d29594b5bcb2d38894ac782b3def01ac2f'
    selected=choose(local,reference,annotations)
    output.mkdir(parents=True)
    selection={'created_at':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               'selected_before_visual_review':True,'model_execution':False,'gpu_access':False,
               'selection': [{'index':i,'role':r} for i,r in selected],
               'selection_limits':'Five purposive failures and one success control, not a representative or unbiased test subset.',
               'source_sha256':{'local_predictions':sha(LOCAL),'author_predictions':sha(REFERENCE),'official_eval':sha(SOURCE)},
               'preflight_input_signature':preflight['input_signature'],
               'visual_budget':{'image_patch_size':16,'video_min_pixels':100352,'video_max_pixels':602112,'video_total_pixels':90316800},
               'provenance':'Reconstruct frozen official preprocessing on CPU. Historical GPU activations/pixel inputs were not saved; no claim of author-pixel identity.'}
    save(output/'selection.json',selection)
    import torch
    assert not torch.cuda.is_initialized()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    spec=importlib.util.spec_from_file_location('charades_case_official_eval',SOURCE);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    processor=module.AutoProcessor.from_pretrained(str(ROOT/'checkpoints/VideoSearchR1__charades-stage2'),use_fast=True,padding_side='left',local_files_only=True)
    meta_index=module.load_video_meta_index(str(DATA/'test/video_npy_with_meta/meta.jsonl'))
    metadata={r['video_id']:r for r in jsonl(DATA/'test/video_npy_with_meta/meta.jsonl')}
    from vllm.model_executor.models.qwen3_vl import Qwen3VLProcessingInfo
    cases=[]
    for index,role in selected:
        left,right=local[index],reference[index];video=left['gt_pos_doc_id'];row=metadata[video]
        assert left['turns'][-1]['top1_doc_id']==video and left['turns'][-1]['answer']=='matched'
        path=DATA/row['output_npy_path'];before=sha(path)
        inputs=module._build_generation_inputs(processor,module._make_system_prompt('<REFINE>'),left['query'],str(path),module.resolve_video_meta_for_video_path(str(path),meta_index),20384,16,100352,602112,90316800)
        batch=module._build_generation_batch_from_inputs(processor,inputs,20384)
        reported=inputs['video_metadatas'][0]
        times=processor._calculate_timestamps(reported['frames_indices'],reported['fps'],2)
        vtimes=Qwen3VLProcessingInfo._calculate_timestamps(None,reported['frames_indices'],reported['fps'],2)
        assert times==vtimes
        text=processor.tokenizer.decode(batch['input_ids'][0],skip_special_tokens=False)
        labels=re.findall(r'<([0-9.]+) seconds>',text)
        assert labels==[f'{t:.1f}' for t in times]
        assert inputs['video_kwargs']['do_sample_frames'] is False
        vp=processor.video_processor;grid=batch['video_grid_thw'][0].tolist();patches=batch['pixel_values_videos']
        normalized=recover_normalized_frames(patches,grid,vp.patch_size,vp.temporal_patch_size,vp.merge_size)
        assert torch.equal(repack_frames(normalized,vp.patch_size,vp.temporal_patch_size,vp.merge_size),patches)
        # vLLM processes the video placeholder independently of the surrounding text.
        only=processor(text=['<|vision_start|><|video_pad|><|vision_end|>'],videos=inputs['video_inputs'],video_metadata=inputs['video_metadatas'],return_tensors='pt',**inputs['video_kwargs'])
        assert torch.equal(only['pixel_values_videos'],patches) and torch.equal(only['video_grid_thw'],batch['video_grid_thw'])
        mean=torch.tensor(vp.image_mean).reshape(1,3,1,1);std=torch.tensor(vp.image_std).reshape(1,3,1,1)
        rgb=((normalized*std+mean)/vp.rescale_factor).round().clamp(0,255).to(torch.uint8).permute(0,2,3,1).numpy()
        assert len(rgb)==row['output_num_frames']
        visual=math.prod(grid)//(vp.merge_size**2)
        assert visual==int(batch['input_ids'].eq(151656).sum()) and batch['input_ids'].shape[1]<20384
        name=f'case_{index:06d}_{video}';directory=output/name;directory.mkdir();(directory/'frames').mkdir()
        for i,array in enumerate(rgb):Image.fromarray(array).save(directory/'frames'/f'{i:03d}.png')
        case={'index':index,'qid':left['qid'],'role':role,'video_id':video,'query':left['query'],
              'spans':{'GT':[left['gt_start'],left['gt_end']],'Local':[left['final_pred_start'],left['final_pred_end']],'Author':[right['final_pred_start'],right['final_pred_end']]},
              'local_iou':left['final_iou_r1'],'author_iou':right['final_iou_r1'],
              'local_turns':len(left['turns']),'author_turns':len(right['turns']),
              'annotation_duration':annotations[index]['duration'],'source_duration_sec':row['source_duration_sec'],
              'source_timestamps_sec':row['sampled_timestamps_sec'],'source_frame_indices':row['sampled_indices_src'],
              'reader_metadata':reported,'model_pair_labels_sec':times,'rendered_model_time_labels':labels,
              'processed_shape':list(rgb.shape),'npy_shape':list(np.load(path,mmap_mode='r').shape),
              'video_grid_thw':grid,'visual_tokens':visual,'input_tokens':int(batch['input_ids'].shape[1]),
              'patch_round_trip_exact':True,'video_only_hf_route_pixel_tensor_equal':True,
              'hf_vllm_time_helper_agreement':True,'do_sample_frames':False,
              'npy_sha256':before,'processed_patch_sha256':hashlib.sha256(patches.numpy().tobytes()).hexdigest(),
              'case_directory':name,'source_metadata':row,
              'display_policy':'RGB reconstructed from normalized temporal/spatial patches, rounded to uint8 for display. No crop, sharpening or generated content.'}
        save(directory/'case.json',case);(directory/'expanded_prompt.txt').write_text(text)
        with (directory/'frames.csv').open('w',newline='') as handle:
            writer=csv.writer(handle,lineterminator='\n');writer.writerow(['npy_index','source_frame_index','source_time_seconds','reader_frame_time_seconds','model_temporal_pair_label_seconds','in_GT','in_local','in_author'])
            for i,t in enumerate(case['source_timestamps_sec']):writer.writerow([i,row['sampled_indices_src'][i],t,i,times[i//2],*[span[0]<=t<=span[1] for span in case['spans'].values()]])
        contact_sheet(case,rgb,directory)
        assert sha(path)==before
        cases.append(case)
        print(f'[case] {index} {video}: {len(rgb)} frames, {visual} visual tokens, {case["input_tokens"]} input tokens; original NPY unchanged',flush=True)
        del batch,only,inputs,normalized,patches,rgb
        gc.collect()
    assert not torch.cuda.is_initialized()
    browser(cases,output)
    summary={'status':'CPU_VISUAL_INPUT_RECONSTRUCTION_PASSED','created_at':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
             'model_execution':False,'gpu_access':False,'cuda_initialized':False,'selection':selection,
             'cases':cases,'output_dir':str(output),'browser':str(output/'case_browser.html'),
             'manual_visual_review_completed':False,'script_sha256':sha(Path(__file__))}
    save(output/'analysis.json',summary)
    print(json.dumps({'status':summary['status'],'indices':[c['index'] for c in cases],'browser':summary['browser']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.output.resolve())
