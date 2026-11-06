#!/usr/bin/env python3
"""Compare actual A/B predictions on the same reconstructed diagnostic frames."""
from __future__ import annotations
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def main():
    summary=json.loads((ROOT/'reports/42_charades_timestamp_summary.json').read_text())
    cpu=json.loads((ROOT/'reports/40_charades_timestamp_cpu.json').read_text())
    images=ROOT/'outputs/charades_localization_cases_20261105/case_browser.html'
    original=json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>',images.read_text(),re.S).group(1))
    pairs={r['index']:r for r in summary['paired_cases']}
    times={r['index']:r for r in cpu['pixel_comparisons']}
    data=[]
    for c in original:
        p=pairs[c['index']];t=times[c['index']]
        data.append({'index':c['index'],'video_id':c['video_id'],'query':c['query'],'images':c['images'],
            'source_times':c['source_timestamps_sec'],'nominal_labels':t['nominal_labels'],'source_labels':t['source_labels'],
            'spans':{'GT':p['gt'],'A':p['nominal_span'],'B':p['source_span'],'Author':p['author_span']},
            'iou':{'A':p['nominal_iou'],'B':p['source_iou'],'Author':p['author_iou']},
            'repeat_span':p['repeat_nominal_span'],'repeat_iou':p['repeat_nominal_iou'],
            'repeat_core_equal':p['AA_core_outcomes_equal']})
    template='''<!doctype html><meta charset="utf-8"><title>Charades 时间戳 A/B 案例</title>
<style>body{font:16px system-ui;margin:24px;background:#f4f6f8;color:#17212b}select,button{font:inherit;padding:8px;margin:4px}main{display:grid;grid-template-columns:minmax(380px,1fr) 1fr;gap:24px}section{background:white;padding:18px;border-radius:10px}img{display:block;max-width:100%;max-height:62vh;margin:auto}input{width:100%}.track{position:relative;background:#eee;height:26px;margin:8px 0}.bar{position:absolute;height:100%}.cursor{position:absolute;height:100%;width:3px;background:#111}small{color:#53616d}#grid{display:none;grid-template-columns:repeat(6,1fr);gap:10px}.thumb{background:white;padding:5px;cursor:pointer;font-size:12px}.thumb img{max-height:180px}@media(max-width:850px){main{grid-template-columns:1fr}#grid{grid-template-columns:repeat(3,1fr)}}</style>
<h1>Charades：同一批画面的时间戳 A/B 对照</h1>
<p>A：名义时间；B：发布元数据中的源时间；A′：重复 A。权重、像素、帧数、视觉预算和计分规则固定。这里展示六个诊断案例，随机 64 题结果另见报告。</p>
<select id="cases"></select><button id="play">播放/暂停</button><button id="toggle">展开/收起全部帧</button>
<h2 id="query"></h2><main><section><img id="frame"><input id="slider" type="range" min="0" value="0"><p id="time"></p><div id="jump"></div></section><section><div id="tracks"></div><p id="repeat"></p><small>同一帧只有一份图片；两组模型时间标签并列显示。时间轨道使用正式计分的原始秒数，光标使用源采样时刻。稀疏帧不能精确说明相邻帧之间的动作。作者输入配置未核实，预测仅供参考。</small></section></main><div id="grid"></div>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
let c=data[0],timer=null;const colors={GT:'#258445',A:'#2563bb',B:'#a137aa',Author:'#cf7508'};
data.forEach((x,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${x.index} / ${x.video_id}`;$('cases').appendChild(o)});
function render(){let i=+$('slider').value,t=c.source_times[i],j=Math.floor(i/2);$('frame').src=c.images[i];$('time').textContent=`帧 ${i}/${c.images.length-1}；源采样时刻 ${t.toFixed(3)}s；A 两帧标签 ${c.nominal_labels[j]}s；B 两帧标签 ${c.source_labels[j]}s`;document.querySelectorAll('.cursor').forEach(x=>x.style.left=100*t/c.max+'%')}
function load(){c=data[+$('cases').value];c.max=Math.max(...Object.values(c.spans).flat(),...c.source_times);$('query').textContent=c.query;$('slider').max=c.images.length-1;$('slider').value=0;$('tracks').replaceChildren();$('jump').replaceChildren();$('grid').replaceChildren();
Object.entries(c.spans).forEach(([name,span])=>{let p=document.createElement('p');p.textContent=`${name}: ${span.join('–')}s`+(name==='GT'?'':`；IoU ${c.iou[name].toFixed(4)}`);p.style.color=colors[name];$('tracks').appendChild(p);let track=document.createElement('div');track.className='track';let bar=document.createElement('div');bar.className='bar';bar.style.cssText=`left:${100*span[0]/c.max}%;width:${100*(span[1]-span[0])/c.max}%;background:${colors[name]}`;track.appendChild(bar);let cur=document.createElement('div');cur.className='cursor';track.appendChild(cur);$('tracks').appendChild(track);span.forEach((t,j)=>{let b=document.createElement('button');b.textContent=name+(j?'终点':'起点');b.onclick=()=>{let k=0;c.source_times.forEach((x,i)=>{if(Math.abs(x-t)<Math.abs(c.source_times[k]-t))k=i});$('slider').value=k;render()};$('jump').appendChild(b)})});
$('repeat').textContent=`A′：${c.repeat_span.join('–')}s；IoU ${c.repeat_iou.toFixed(4)}；A/A′核心结果一致：${c.repeat_core_equal}`;
c.images.forEach((src,i)=>{let cell=document.createElement('div');cell.className='thumb';let im=document.createElement('img');im.src=src;cell.appendChild(im);let cap=document.createElement('div');cap.textContent=`帧${i} 源${c.source_times[i].toFixed(2)}s / A${c.nominal_labels[Math.floor(i/2)]} / B${c.source_labels[Math.floor(i/2)]}`;cell.appendChild(cap);cell.onclick=()=>{$('slider').value=i;render()};$('grid').appendChild(cell)});render()}
$('cases').onchange=load;$('slider').oninput=render;$('toggle').onclick=()=>{$('grid').style.display=$('grid').style.display==='grid'?'none':'grid'};$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null}else timer=setInterval(()=>{$('slider').value=(+$('slider').value+1)%c.images.length;render()},700)};load();
</script>'''
    output=Path(summary['output_dir'])/'timestamp_pair_browser.html'
    if output.exists():raise ValueError('Preserve existing paired browser')
    output.write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')))
    print(output)


if __name__=='__main__':main()
