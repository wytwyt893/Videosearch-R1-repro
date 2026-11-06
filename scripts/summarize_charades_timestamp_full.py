#!/usr/bin/env python3
"""CPU-only full paired comparison; recoverable without another model run."""
from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

from inspect_charades_localization_cases import ROOT, LOCAL, REFERENCE, jsonl, sha
from run_charades_full import BASE
from score_charades_protocol import score_rows

REPORT = ROOT/'reports/45_charades_timestamp_full_summary.json'


def write(path,value):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def accepted_gt(row):
    t = row['turns'][-1]
    return t['top1_is_gt'] and t['answer']=='matched' and t['time_parse_ok']


def paired_statistics(a,b):
    if not a or len(a)!=len(b) or [r['index'] for r in a]!=[r['index'] for r in b]:
        raise ValueError('Require matched original indices without dropped examples')
    paired=[]
    alignment_keys=['query','qid','gt_pos_doc_id','gt_start','gt_end','orig_rank','turn1_top1_doc_id']
    for old,new in zip(a,b):
        if any(old[k]!=new[k] for k in alignment_keys):
            raise ValueError('Question/GT/initial retrieval changed between conditions')
        paired.append({'index':old['index'],'qid':old['qid'],'query':old['query'],
            'gt_video':old['gt_pos_doc_id'],'gt_start':old['gt_start'],'gt_end':old['gt_end'],
            'A_start':old['final_pred_start'],'A_end':old['final_pred_end'],
            'B_start':new['final_pred_start'],'B_end':new['final_pred_end'],
            'A_iou':old['final_iou_r1'],'B_iou':new['final_iou_r1'],
            'delta_iou':new['final_iou_r1']-old['final_iou_r1'],
            'A_last_video':old['turns'][-1]['top1_doc_id'],'B_last_video':new['turns'][-1]['top1_doc_id'],
            'A_answer':old['final_answer'],'B_answer':new['final_answer'],
            'A_final_rank':old['final_rank'],'B_final_rank':new['final_rank'],
            'A_refines':old['refine_applied_turns'],'B_refines':new['refine_applied_turns'],
            'both_accept_GT':accepted_gt(old) and accepted_gt(new)})
    thresholds={}
    for t in [.3,.5,.7]:
        gained=[p['index'] for p in paired if p['A_iou']<t<=p['B_iou']]
        lost=[p['index'] for p in paired if p['B_iou']<t<=p['A_iou']]
        thresholds[str(t)]={'gained':len(gained),'lost':len(lost),'net':len(gained)-len(lost),
            'delta_percentage_points':100*(len(gained)-len(lost))/len(paired),
            'gained_indices':gained,'lost_indices':lost}
    joint=[p for p in paired if p['both_accept_GT']]
    errors={}
    for c in ['A','B']:
        errors[c]={k:statistics.mean(abs(p[f'{c}_{part}']-p[f'gt_{part}']) for p in joint) if joint else None
            for part,k in [('start','start_MAE_seconds'),('end','end_MAE_seconds')]}
    summary={'N':len(paired),'thresholds':thresholds,
        'iou_improved':sum(p['delta_iou']>1e-9 for p in paired),
        'iou_worsened':sum(p['delta_iou']< -1e-9 for p in paired),
        'iou_unchanged':sum(abs(p['delta_iou'])<=1e-9 for p in paired),
        'joint_accepted_GT_N':len(joint),'joint_accepted_GT_boundary_errors':errors,
        'last_verified_video_changed':sum(p['A_last_video']!=p['B_last_video'] for p in paired),
        'final_answer_changed':sum(p['A_answer']!=p['B_answer'] for p in paired),
        'final_rank_changed':sum(p['A_final_rank']!=p['B_final_rank'] for p in paired),
        'mean_iou_delta':statistics.mean(p['delta_iou'] for p in paired)}
    return summary,paired


def audit_inputs(run,rows):
    metadata={r['video_id']:r for r in jsonl(ROOT/'data/charades-sta/test/video_npy_with_meta/meta.jsonl')}
    visual={};counts={}
    for clock,state in run['conditions'].items():
        audit=Path(state['input_audit'])
        if sha(audit)!=state['audit_sha256']:raise ValueError('Runtime input audit changed')
        events=jsonl(audit);counts[clock]=len(events)
        if len(events)!=sum(len(r['turns']) for r in rows[clock]):
            raise ValueError('Audit must cover every actual verification')
        for event in events:
            vid=event['video_id']
            current={k:event[k] for k in ['video_kwargs','video_shapes','video_tensor_sha256']}
            if current['video_kwargs'].get('do_sample_frames') is not False:
                raise ValueError('Unexpected additional sampling')
            if vid in visual and visual[vid]!=current:
                raise ValueError('Visual input changed between conditions or turns')
            visual[vid]=current
            n=metadata[vid]['output_num_frames']
            times=metadata[vid]['sampled_timestamps_sec'] if clock=='source' else list(range(n))
            if len(times)%2:times=times+[times[-1]]
            labels=[f'{(times[i]+times[i+1])/2:.1f}' for i in range(0,len(times),2)]
            if event['time_labels']!=labels:
                raise ValueError(f'Unexpected clock labels: {clock}/{vid}')
    return {'passed':True,'generation_calls':counts,'seen_videos':len(visual),
            'all_seen_video_tensors_consistent':True,'every_generation_time_label_verified':True}


def metric_row(score):
    p=score['primary'];v=p['VCMR'];vr=p['VR'];ver=p['VER']
    return {'VCMR_03':v['IoU@0.3@R1'],'VCMR_05':v['IoU@0.5@R1'],'VCMR_07':v['IoU@0.7@R1'],
        'VER_last':ver['accuracy'],'VER_valid_N':ver['N'],'VER_invalid':ver['invalid_answers'],
        'VER_all_examples':ver['accuracy_all_examples'],
        'VER_first_supplementary':score['supplementary']['first_turn_VER']['accuracy'],
        'VR_R1':vr['R@1'],'VR_R5':vr['R@5'],'VR_R10':vr['R@10'],'VR_R100':vr['R@100']}


def main(run_path,destination):
    if destination.exists():
        raise ValueError('Preserve existing full comparison; choose --report for recovery')
    run=json.loads(run_path.read_text())
    if run['status'] not in {'inference_complete','complete','failed'} or set(run['conditions'])!={'nominal','source'}:
        raise ValueError('Both complete inference conditions are required')
    output=Path(run['output_dir'])
    rows={};scores={}
    for clock,state in run['conditions'].items():
        if state['status']!='complete' or not state['resource_release']['passed']:
            raise ValueError('Each condition must be complete and release its GPU')
        predictions=Path(state['predictions'])
        if sha(predictions)!=state['prediction_sha256']:raise ValueError('Predictions changed')
        if sha(state['primary_result'])!=state['primary_result_sha256']:raise ValueError('Primary score changed')
        rows[clock]=jsonl(predictions)
        if [r['index'] for r in rows[clock]]!=list(range(3720)):
            raise ValueError('Full original index coverage is required')
        scores[clock]=score_rows(rows[clock],3720,include_first_turn=True)
    invariant_exclusions={'--output_json','--output_jsonl'}
    a_args={k:v for k,v in run['conditions']['nominal']['arguments'].items() if k not in invariant_exclusions}
    b_args={k:v for k,v in run['conditions']['source']['arguments'].items() if k not in invariant_exclusions}
    if a_args!=b_args:raise ValueError('Non-clock inference arguments differ')
    audit=audit_inputs(run,rows)
    stats,paired=paired_statistics(rows['nominal'],rows['source'])
    historical=jsonl(LOCAL);author=jsonl(REFERENCE)
    for label,group in [('historical',historical),('author_released',author)]:
        scores[label]=score_rows(group,3720,include_first_turn=True)
    metrics={name:metric_row(s) for name,s in scores.items()}
    comparison={'status':'FULL_TIMESTAMP_PAIRED_COMPARISON_COMPLETE',
        'created_at':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'run_report':str(run_path),'output_dir':str(output),
        'run_report_sha256_at_inference_complete':sha(run_path),
        'N':3720,'corpus_videos':1334,'gpu_index':run['gpu_index'],'gpu_uuid':run['gpu_uuid'],
        'metrics':metrics,'paired_statistics':stats,'runtime_input_audit':audit,
        'protocol':scores['nominal']['protocol'],
        'scores':scores,'time_semantics':json.loads(Path(run['plan']).read_text())['time_semantics'],
        'full_inference_counts':{k:len(v) for k,v in rows.items()},
        'inputs_unchanged':True,'historical_prediction_sha256':sha(LOCAL),'author_prediction_sha256':sha(REFERENCE),
        'prediction_sha256':{k:s['prediction_sha256'] for k,s in run['conditions'].items()},
        'summary_script_sha256':sha(Path(__file__)),
        'interpretation_limits':'Single fixed-seed paired full test. Timestamp inputs may change verification, SQR and ranking downstream. Source time inherits released index/FPS metadata, not newly decoded container PTS. Author input settings remain unconfirmed.'}
    write(output/'comparison.json',comparison)
    with (output/'paired_examples.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    with (output/'metrics_comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['condition']+list(metrics['nominal']))
        writer.writeheader()
        for name,m in metrics.items():writer.writerow({'condition':name,**m})
    for direction,reverse in [('gains',True),('losses',False)]:
        ordered=sorted(paired,key=lambda p:p['delta_iou'],reverse=reverse)
        write(output/f'largest_{direction}.json',ordered[:30])
    comparison['artifact_sha256']={name:sha(output/name) for name in [
        'comparison.json','paired_examples.csv','metrics_comparison.csv','largest_gains.json','largest_losses.json']}
    stamp=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%m.%d')
    doc=ROOT/f'doc/videosearch_r1 × CHARADES全量时间戳对照-{stamp}.md'
    if doc.exists():raise ValueError('Preserve an existing experiment document')
    f=lambda x:'—' if x is None else f'{100*x:.2f}'
    lines=[f'# videosearch_r1 × CHARADES全量时间戳对照-{stamp}','',
        'A 为当前官方名义时间，B 为原始帧号/raw FPS 的源时间。两组在同一张 GPU 顺序运行，各覆盖全部 3720 题和完整 1334 视频检索库，原始 index 0–3719 不去重。输入像素、帧数/顺序、模型权重、视觉预算、vLLM、T=2/N=8、生成和计分设置保持固定。','',
        '本实验沿用已通过 70 题 A/B 和 A/A 重复检查的适配器，只移除样本排除列表。历史全量与作者发布预测仅作补充参考，主因果对照使用本轮同期 A/B。','',
        '## 指标对照','',
        '以下为百分比，VER 主结果取末次实际验证并按有效回答计分；无效数与全部样本分母准确率见机器报告，首轮 VER 仅补充。VR 使用末轮更新后排名，VCMR 沿用最后实际验证并定位的视频。','',
        '| 条件 | VCMR 0.3/R@1 | 0.5/R@1 | 0.7/R@1 | 末次 VER | 首轮 VER | VR R@1 | R@5 | R@10 | R@100 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    labels={'nominal':'A：名义时间','source':'B：源时间','historical':'历史全量','author_released':'作者发布预测（同口径计分）'}
    for name in ['nominal','source','historical','author_released']:
        m=metrics[name]
        keys=['VCMR_03','VCMR_05','VCMR_07','VER_last','VER_first_supplementary','VR_R1','VR_R5','VR_R10','VR_R100']
        lines.append('| '+labels[name]+' | '+' | '.join(f(m[k]) for k in keys)+' |')
    lines += ['', '## 配对收益与退化','',
        '| IoU 阈值 | 新增命中 | 丢失命中 | 净增 | 变化（百分点） |',
        '| --- | --- | --- | --- | --- |']
    for t,s in stats['thresholds'].items():
        lines.append(f'| {t} | {s["gained"]} | {s["lost"]} | {s["net"]:+d} | {s["delta_percentage_points"]:+.3f} |')
    lines += ['',f'逐题 IoU 改善/下降/不变：{stats["iou_improved"]}/{stats["iou_worsened"]}/{stats["iou_unchanged"]}。双方都接受并定位 GT 的子集为 {stats["joint_accepted_GT_N"]} 题，起止边界误差见比较 JSON。',
        f'末次验证视频改变 {stats["last_verified_video_changed"]} 题，最终回答改变 {stats["final_answer_changed"]} 题，最终排名改变 {stats["final_rank_changed"]} 题。这些后续变化保留为时间输入干预的结果。','',
        '## 完整性与资源','',
        f'GPU{run["gpu_index"]}，UUID `{run["gpu_uuid"]}`；每组退出后确认显存释放，随后才进入下一阶段。', '',
        f'运行时审计覆盖全部生成调用；涉及 {audit["seen_videos"]} 个视频，已核验视觉张量/采样设置一致及两种时钟标签。源时间来自发布元数据，尚未重新解码全部原视频确认 PTS。','']
    for clock,state in run['conditions'].items():
        lines.append(f'- {labels[clock]}：{state["elapsed_seconds"]/3600:.3f} 小时；显存采样峰值 {state["peak_gpu_used_mib"]} MiB（包括 vLLM 缓存预留）；预测 3720 条；无效 VER 回答 {metrics[clock]["VER_invalid"]}。')
    rel=output.relative_to(ROOT)
    lines += ['', f'[完整运行与命令](../reports/44_charades_timestamp_full.json) · [CPU 资格核验](../reports/43_charades_timestamp_full_cpu.json) · [机器摘要](../reports/45_charades_timestamp_full_summary.json)', '',
        f'[配对比较 JSON](../{rel}/comparison.json) · [逐题 CSV](../{rel}/paired_examples.csv) · [指标 CSV](../{rel}/metrics_comparison.csv) · [最大改善案例](../{rel}/largest_gains.json) · [最大退化案例](../{rel}/largest_losses.json)', '',
        '历史预测、官方源码及数据未覆盖。本轮为源时间戳变体的单次全量验证；是否构成稳定的方法收益、是否能在其他数据集泛化，需要进一步实验。','']
    doc.write_text('\n'.join(lines))
    comparison['document']=str(doc)
    comparison['document_sha256']=sha(doc)
    write(destination,comparison)
    print(json.dumps({'status':comparison['status'],'report':str(destination),'document':str(doc),
                      'metrics':metrics,'paired_statistics':stats},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'reports/44_charades_timestamp_full.json')
    p.add_argument('--report',type=Path,default=REPORT)
    args=p.parse_args();main(args.run,args.report)
