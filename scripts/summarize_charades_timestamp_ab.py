#!/usr/bin/env python3
"""Offline summary of a completed, paired clock-only exploratory experiment."""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

from inspect_charades_localization_cases import LOCAL, REFERENCE, ROOT, jsonl, save, sha

RUN = ROOT/'reports/41_charades_timestamp_ab.json'
REPORT = ROOT/'reports/42_charades_timestamp_summary.json'
DOCUMENT = ROOT/'doc/videosearch_r1 × CHARADES时间戳对照检查-11.05.md'
REPEAT = ROOT/'reports/41a_charades_timestamp_nominal_repeat.json'


def outcome(row):
    keys=['orig_rank','final_rank','final_top1_doc_id','final_answer','final_pred_start','final_pred_end','final_iou_r1','refine_applied_turns']
    turn_keys=['turn','top1_doc_id','answer','pred_start','pred_end','time_parse_ok','policy_top1_doc_id','policy_rank','iou_r1','refine_applied']
    return {**{k:row[k] for k in keys},'turns':[{k:t[k] for k in turn_keys} for t in row['turns']]}


def main(run_path,report_path,document_path,repeat_path):
    if report_path.exists() or document_path.exists():raise ValueError('Preserve existing analysis')
    run=json.loads(run_path.read_text())
    if run['status']!='complete':raise ValueError('Inference and runtime input audit must both complete first')
    output=Path(run['output_dir']);comparison=run['comparison']
    rows={clock:jsonl(Path(condition['output_dir'])/'external_verified_test_temporal_grounding_VideoSearchR1__charades-stage2.jsonl')
          for clock,condition in run['conditions'].items()}
    for clock,group in rows.items():
        condition=run['conditions'][clock]
        if sha(Path(condition['output_dir'])/'external_verified_test_temporal_grounding_VideoSearchR1__charades-stage2.jsonl')!=condition['prediction_sha256']:
            raise ValueError('Saved prediction changed')
        if not condition['resource_release']['passed']:raise ValueError('GPU release check failed')
    historical=jsonl(LOCAL)
    if sha(REFERENCE)!='ac2256b6a5d882df706cc8b0730b22d29594b5bcb2d38894ac782b3def01ac2f':
        raise ValueError('Pinned author prediction file changed')
    author=jsonl(REFERENCE)
    repeat=json.loads(repeat_path.read_text())
    if repeat['status']!='complete' or repeat['ab_run_sha256']!=sha(run_path) or not repeat['resource_release']['passed']:
        raise ValueError('Require the completed A/A repeat bound to this A/B run')
    repeat_predictions=Path(repeat['output_dir'])/'external_verified_test_temporal_grounding_VideoSearchR1__charades-stage2.jsonl'
    if sha(repeat_predictions)!=repeat['prediction_sha256']:raise ValueError('A/A predictions changed')
    if sha(Path(repeat['output_dir'])/'input_audit.jsonl')!=repeat['input_audit_sha256']:
        raise ValueError('A/A input audit changed')
    rows['repeat_nominal']=jsonl(repeat_predictions)
    if [r['index'] for r in rows['repeat_nominal']]!=run['selection']['original_indices']:
        raise ValueError('A/A original indices changed')
    for group in rows.values():
        if any(r['max_turn']!=2 or any(t.get('rollout_depth_actual')!=8 or t.get('rollout_shape')!=[8,2048]
               for t in r['turns'] if t.get('refine_applied')) for r in group):
            raise ValueError('Applied refinement must retain T=2 and N=8')
    by={clock:{r['index']:r for r in group} for clock,group in rows.items()}
    outcome_replays=[r['index'] for r in rows['nominal'] if outcome(r)==outcome(historical[r['index']])]
    aa_replays=[r['index'] for r in rows['nominal'] if outcome(r)==outcome(by['repeat_nominal'][r['index']])]
    strata={}
    for name,cohort in comparison['cohorts'].items():
        ids=cohort['nominal']['original_indices']
        paired=[p for p in comparison['paired_cases'] if p['index'] in ids]
        joint=[p for p in paired if p['same_verified_GT_and_matched']]
        metrics={}
        scores={c:cohort[c] for c in ['historical','nominal','source']}
        scores['repeat_nominal']=repeat['cohorts'][name]
        for clock in ['historical','nominal','source','repeat_nominal']:
            score=scores[clock]['primary']
            metrics[clock]={'VER_last':score['VER']['accuracy'],
                'VER_valid':score['VER']['N'],'VER_invalid':score['VER']['invalid_answers'],
                'VER_first':scores[clock]['supplementary']['first_turn_VER']['accuracy'],
                'VR_R1':score['VR']['R@1'],'VR_R5':score['VR']['R@5'],
                'VCMR_03':score['VCMR']['IoU@0.3@R1'],
                'VCMR_05':score['VCMR']['IoU@0.5@R1'],
                'VCMR_07':score['VCMR']['IoU@0.7@R1'],'mean_iou_R1':score['VCMR']['mIoU@R1']}
        def boundary_errors(clock):
            return {label:statistics.mean(abs(by[clock][p['index']][field]-by[clock][p['index']][gt]) for p in joint)
                    if joint else None for label,field,gt in [('start_MAE_seconds','final_pred_start','gt_start'),('end_MAE_seconds','final_pred_end','gt_end')]}
        strata[name]={'N':len(ids),'metrics':metrics,
            'paired_gains_07':cohort['strict_0point7_gained'],'paired_losses_07':cohort['strict_0point7_lost'],
            'iou_improved':cohort['iou_improved'],'iou_worsened':cohort['iou_worsened'],'iou_unchanged':cohort['iou_unchanged'],
            'nominal_replays_historical_core_outcomes':sum(i in outcome_replays for i in ids),
            'nominal_repeat_core_outcomes_equal':sum(i in aa_replays for i in ids),
            'AA_iou_changed':sum(by['nominal'][i]['final_iou_r1']!=by['repeat_nominal'][i]['final_iou_r1'] for i in ids),
            'AA_strict_gained':sum(by['nominal'][i]['final_iou_r1']<.7<=by['repeat_nominal'][i]['final_iou_r1'] for i in ids),
            'AA_strict_lost':sum(by['repeat_nominal'][i]['final_iou_r1']<.7<=by['nominal'][i]['final_iou_r1'] for i in ids),
            'joint_accepted_GT_count':len(joint),'joint_accepted_GT_boundary_errors':{c:boundary_errors(c) for c in ['nominal','source']},
            'last_checked_video_changed':sum(p['nominal_last_video']!=p['source_last_video'] for p in paired),
            'final_answer_changed':sum(p['nominal_answer']!=p['source_answer'] for p in paired)}
    plan=json.loads(Path(run['plan']).read_text())
    clock_audit=json.loads(Path(plan['all_video_clock_audit']).read_text())
    report={'status':'TIMESTAMP_PAIRED_EXPLORATORY_CHECK_COMPLETE',
        'created_at':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'run_report':str(run_path),'run_report_sha256':sha(run_path),
        'repeat_report':str(repeat_path),'repeat_report_sha256':sha(repeat_path),
        'cpu_report':run['plan'],'cpu_report_sha256':run['plan_sha256'],
        'output_dir':str(output),'gpu_index':run['gpu_index'],'gpu_uuid':run['gpu_uuid'],
        'N_per_condition':70,'cohorts':strata,
        'paired_cases':comparison['paired_cases'],'runtime_audit':comparison['runtime_audit'],
        'all_video_clock_stats':{'N':len(clock_audit),'median_last_frame_difference_seconds':statistics.median(r['last_difference_sec'] for r in clock_audit)},
        'nominal_historical_core_outcome_mismatch_indices':[i for i in run['selection']['original_indices'] if i not in outcome_replays],
        'AA_core_outcome_mismatch_indices':[i for i in run['selection']['original_indices'] if i not in aa_replays],
        'repeat_condition_resources':{k:repeat[k] for k in ['elapsed_seconds','peak_gpu_used_mib','resource_release']},
        'condition_resources':{c:{k:s[k] for k in ['elapsed_seconds','peak_gpu_used_mib','resource_release','prediction_sha256','audit_sha256']} for c,s in run['conditions'].items()},
        'original_prediction_sha256':sha(LOCAL),'analysis_script_sha256':sha(Path(__file__)),
        'limits':comparison['limits'],
        'counterfactual':'Fresh nominal and source runs on the same GPU; actual input timestamps changed, not an output-coordinate remapping.',
        'actual_T2_N8_verified_all_conditions':True,
        'source_time_semantics':plan['timestamp_semantics'],
        'baseline_overwritten':False,'full_benchmark_result':False}
    for p in report['paired_cases']:
        r=by['repeat_nominal'][p['index']]
        p['repeat_nominal_iou']=r['final_iou_r1']
        p['repeat_nominal_span']=[r['final_pred_start'],r['final_pred_end']]
        p['AA_core_outcomes_equal']=p['index'] in aa_replays
        reference=author[p['index']]
        p['author_span']=[reference['final_pred_start'],reference['final_pred_end']]
        p['author_iou']=reference['final_iou_r1']
    save(report_path,report)
    fmt=lambda x:'—' if x is None else f'{x*100:.2f}'
    span=lambda x:'—' if any(v is None for v in x) else f'{x[0]:g}–{x[1]:g}'
    lines=['# videosearch_r1 × CHARADES时间戳对照检查-11.05','',
        '完成一次独立的时间戳单变量探索实验：A 使用官方名义时间，B 透传发布元数据中的原始帧号与 raw FPS。两组均重新推理，使用同一张 GPU、同一批题和完整 1334 视频检索库。原始模型、像素、标签、配置和历史全量预测保持不变。','',
        '本轮只改变时间信息传递；权重、帧数/顺序、视觉预算、生成参数、vLLM、T=2、N=8、末次 VER、末轮 SQR 和计分规则固定。时间数字的 token 化及其后续推理、验证、检索变化属于该干预的结果。','',
        '## 预先固定的样本与输入验收','',
        '6 个已查看案例单独作为诊断组；从剩余 3714 题中以种子 20261105 随机抽取 64 题，单独报告。两组并集为 70 题，原始 index 保留，query/qid 不去重。官方 resume-skip 入口只用于筛选题目，没有缩小检索库，也没有伪造其他题的预测。','',
        '观察到 A 与历史全量的部分 SQR 轨迹不同后，额外增加 A′：保持 A 的同样 70 题、顺序和参数重新运行。该控制是运行期间新增并明确记录，旨在检查重复性；没有根据 B 的效果重新选题。','',
        'CPU 检查覆盖全部 1334 条源时间列表；6 个诊断案例实际 HF 处理后的像素和视觉网格逐位一致，提示文本只有时间标签变化。运行时审计覆盖所有生成调用，逐视频检查视觉张量/采样设置一致，并核对实际标签。源时间继承官方 sampled_indices_src/raw_fps，未重新解码原视频验证容器 PTS。','',
        '## 配对结果','',
        '以下均为百分比；64 题的一个命中对应 1.5625 个百分点，6 题的一个命中对应 16.67 个百分点。不能直接外推为全量论文成绩。','',
        '| 组别 | 条件 | VCMR 0.3/R@1 | 0.5/R@1 | 0.7/R@1 | 末次 VER | 首轮 VER（补充） | VR R@1 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    labels={'diagnostic_cases':'诊断 6 题','random64_excluding_diagnostic':'随机 64 题'}
    for name,s in strata.items():
        for clock,label in [('historical','历史全量中对应题'),('nominal','A：重新运行名义时间'),('source','B：源时间'),('repeat_nominal','A′：名义时间重复控制')]:
            m=s['metrics'][clock]
            lines.append(f'| {labels[name]} | {label} | {fmt(m["VCMR_03"])} | {fmt(m["VCMR_05"])} | {fmt(m["VCMR_07"])} | {fmt(m["VER_last"])} | {fmt(m["VER_first"])} | {fmt(m["VR_R1"])} |')
    for name,s in strata.items():
        lines += ['',f'**{labels[name]}：**IoU 改善/下降/不变为 {s["iou_improved"]}/{s["iou_worsened"]}/{s["iou_unchanged"]}；0.7 阈值新增命中 {s["paired_gains_07"]}，丢失命中 {s["paired_losses_07"]}。A 与历史全量核心轨迹结果一致 {s["nominal_replays_historical_core_outcomes"]}/{s["N"]}。',
            f'A/A′ 核心结果一致 {s["nominal_repeat_core_outcomes_equal"]}/{s["N"]}，IoU 变化 {s["AA_iou_changed"]} 题，0.7 阈值新增/丢失命中 {s["AA_strict_gained"]}/{s["AA_strict_lost"]}。',
            f'双方均接受并定位 GT 的题数为 {s["joint_accepted_GT_count"]}；对应起止边界绝对误差见机器报告。末次检查的视频发生变化 {s["last_checked_video_changed"]} 题，回答发生变化 {s["final_answer_changed"]} 题。']
    lines += ['', '## 六个诊断案例','',
        '| index | GT（秒） | A 名义时间预测 | B 源时间预测 | A IoU | B IoU | A′ IoU | 作者 IoU |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for p in comparison['paired_cases']:
        if p['index'] in run['selection']['cohorts']['diagnostic_cases']:
            lines.append(f'| {p["index"]} | {span(p["gt"])} | {span(p["nominal_span"])} | {span(p["source_span"])} | {p["nominal_iou"]:.4f} | {p["source_iou"]:.4f} | {p["repeat_nominal_iou"]:.4f} | {p["author_iou"]:.4f} |')
    lines += ['', '作者列来自固定 revision 发布的预测，仅供对照；若 B 与其一致，也不足以反推出作者具体使用的视觉输入或时间戳配置。']
    lines += ['', '## 资源、证据与解释范围','',
        f'物理 GPU{run["gpu_index"]}，UUID `{run["gpu_uuid"]}`；A、B 顺序执行，分别检查进程退出后 GPU 回到空闲。','']
    for c,s in run['conditions'].items():
        lines.append(f'- {c}：{s["elapsed_seconds"]:.2f} 秒，设备显存采样峰值 {s["peak_gpu_used_mib"]} MiB；包含 vLLM 缓存预留。')
    lines.append(f'- A′：{repeat["elapsed_seconds"]:.2f} 秒，设备显存采样峰值 {repeat["peak_gpu_used_mib"]} MiB；退出后空闲验收通过。')
    lines += ['', '本轮检查区分了时间映射正确性与动作理解能力。小样本中的改善、退化都需保留；若 A 本身不能重放历史结果，应优先分析重复性，不能把历史全量与 B 的差异全部归因于时间戳。全量收益需另行正式验证。', '',
        f'[逐帧 A/B 对照浏览器](../{output.relative_to(ROOT)}/timestamp_pair_browser.html) · [CPU 验收](../reports/40_charades_timestamp_cpu.json) · [完整运行与实际命令](../reports/41_charades_timestamp_ab.json) · [A/A′ 重复控制](../reports/41a_charades_timestamp_nominal_repeat.json) · [离线结果摘要](../reports/42_charades_timestamp_summary.json) · [逐题配对结果](../{output.relative_to(ROOT)}/comparison.json)', '',
        '[时间戳适配器](../scripts/charades_timestamp_eval.py) · [CPU 检查](../scripts/check_charades_timestamps.py) · [A/B 启动器](../scripts/run_charades_timestamp_ab.py) · [离线汇总](../scripts/summarize_charades_timestamp_ab.py)', '']
    document_path.write_text('\n'.join(lines))
    print(json.dumps({'status':report['status'],'report':str(report_path),'document':str(document_path),'cohorts':strata},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--report',type=Path,default=REPORT)
    p.add_argument('--document',type=Path,default=DOCUMENT)
    p.add_argument('--repeat',type=Path,default=REPEAT)
    a=p.parse_args();main(a.run,a.report,a.document,a.repeat)
