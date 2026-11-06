#!/usr/bin/env python3
"""Record the inspected case evidence and create an offline review overview.

Manual notes are explicitly separated from dataset labels and model metrics.
The coordinate-only diagnostic is not a new inference or a corrected benchmark.
"""
from __future__ import annotations
import datetime
import json
from pathlib import Path
import statistics
import textwrap
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image,ImageDraw
from inspect_charades_localization_cases import ROOT,LOCAL,REFERENCE,DATA,font,jsonl,save,sha

OUTPUT=ROOT/'outputs/charades_localization_cases_20261105'
REPORT=ROOT/'reports/39_charades_localization_cases.json'
DOCUMENT=ROOT/'doc/videosearch_r1 × CHARADES定位失败案例分析-11.05.md'
NOTES={
1252:{'short':'Repeated eating episodes separated by jacket manipulation; whole-clip merge.',
      'frames':[0,8,23],
      'observations':['0–1 秒附近可见食物靠近嘴；约 3–20 秒间有脱、整理或穿外套的不同阶段；23–27 秒又出现进食。',
                      '本地 0–27 秒把前后相似动作及中间的外套操作合并；GT 只标 0–3.5 秒的早期实例。'],
      'hypothesis':'目标动作实例没有与持续坐姿、人物外观和其他动作阶段分开；重复实例也带来单一 GT 的选择歧义。',
      'limits':'稀疏帧不能确定每一口的精确起止；不据此修改 GT，也不声称后来进食必定满足全部细粒度描述。'},
870:{'short':'Approach/opening versus closing the wardrobe: wrong action stage.',
     'frames':[12,25,30],
     'observations':['约 3–12 秒主要是走近、站在柜前；约 12–14 秒柜门从关闭变成打开，之后整理物品。',
                     '约 26–32 秒可见柜门逐渐合上；GT 25–32.5 秒与这一阶段相符，本地 2.5–13 秒落在前面的接近/打开阶段。'],
     'hypothesis':'人物和物体匹配较好，但相反状态变化（开门/关门）及其先后关系被混淆。',
     'limits':'这是大幅错位案例，不能代表所有 79 条差异；约 1 秒时间轴修正无法把前段移动到后段。'},
2233:{'short':'Carrying/phone use versus drinking; repeated sips and an early lead-in.',
      'frames':[3,10,30],
      'observations':['约 3 秒人物走入；8–9 秒有持杯/看手中物体的阶段；10.5、12.6 秒可见杯子靠嘴，31.6 秒附近又有类似饮水。',
                      '本地起点 1.6 秒包含大量走近、持物的前导片段；GT 为 8.1–14.5 秒。'],
      'hypothesis':'把持杯、人物出现或准备阶段提前算入饮水；重复动作的实例选择需要更明确证据。',
      'limits':'GT 的确切边界仍比稀疏视觉动作更宽，不能把人工看到的举杯帧直接改成新 GT。'},
2344:{'short':'Late two-person interaction; subtle laughter and an overrun GT end.',
      'frames':[16,21,30],
      'observations':['早段以站立者的动作更明显；约 20–22 秒后，两人围绕图片、转头与互动的画面更明显。',
                      '本地 17–29 秒提前结束；GT 21.9–35 秒，annotation duration 为 33.93 秒，源视频最后采样帧约 33.64 秒。'],
      'hypothesis':'多主体关系与细微表情的时段判断较难，同时存在标注末端超出可观测视频的问题。',
      'limits':'低位/遮挡的人体可能在更早帧已出现；不能断言早段完全没有第二个人，也无法仅凭 1 FPS 帧精确定义开始发笑的时刻。'},
1516:{'short':'Same stand-up event; small endpoint and annotation-duration ambiguity.',
      'frames':[24,28,31],
      'observations':['人物前段持续倚躺；约 26–30 秒可见调整腿部、转身并起身，31 秒后已站立。',
                      '本地 25–33 秒与作者 26–34.37 秒都覆盖该事件；GT 23.4–35 秒包含更早准备阶段并延伸到声明时长之后。'],
      'hypothesis':'更像小幅边界、名义时钟和标注容差问题，而不是完全看错动作；适合单独测试时间轴。',
      'limits':'本地 IoU 0.6897 接近 0.7，但并非浮点舍入边界；不能仅凭跨阈值就声称语义明显变好。'},
26:{'short':'Success control: walking ends although the person and cup remain.',
    'frames':[1,4,12],
    'observations':['人物前几秒穿过门口，随后大部分时间站在门边持杯。',
                    '本地和作者均给出 0–5 秒，GT 为 0–6.3 秒；没有把后续持续站立、持杯全部算入穿门动作。'],
    'hypothesis':'同一处理流水线能够区分明显的动作阶段，失败不是所有视频都被统一输出为整段。',
    'limits':'这是按规则选出的一个成功对照，不是总体准确率的估计。'}
}


def iou(span,gt):
    overlap=max(0,min(span[1],gt[1])-max(span[0],gt[0]))
    return overlap/(max(span[1],gt[1])-min(span[0],gt[0]))


def overview(cases):
    w,h=1680,130+len(cases)*350
    image=Image.new('RGB',(w,h),'white');d=ImageDraw.Draw(image)
    d.text((18,12),'Charades localization case review: 5 failures + 1 success control',fill='black',font=font(24))
    d.text((18,50),'CPU-reconstructed processor RGB. Each thumbnail is an unaltered frame except display resizing.',fill='#444444',font=font(16))
    d.text((18,77),'Time captions show source sampling time / model temporal-pair label; full frame galleries and timelines are in case_browser.html.',fill='#444444',font=font(14))
    for r,case in enumerate(cases):
        note=NOTES[case['index']];y=130+r*350
        d.rectangle((0,y,w,y+346),fill='#f4f6f9' if r%2==0 else '#ffffff')
        d.text((18,y+8),f'{case["index"]} / {case["video_id"]}',fill='black',font=font(23))
        for j,line in enumerate(textwrap.wrap(note['short'],48)):
            d.text((18,y+45+j*21),line,fill='#333333',font=font(16))
        for j,(name,color) in enumerate([('GT','#258445'),('Local','#2563bb'),('Author','#cf7508')]):
            a,b=case['spans'][name]
            d.text((18,y+130+j*26),f'{name}: {a:g}-{b:g}s',fill=color,font=font(19))
        d.text((18,y+222),f'IoU: local={case["local_iou"]:.3f}; author={case["author_iou"]:.3f}',fill='black',font=font(16))
        d.text((18,y+253),f'{len(case["source_timestamps_sec"])} frames; {case["visual_tokens"]} visual tokens',fill='#555555',font=font(15))
        for col,i in enumerate(note['frames']):
            frame=Image.open(OUTPUT/case['case_directory']/'frames'/f'{i:03d}.png').convert('RGB')
            frame.thumbnail((360,275),Image.Resampling.LANCZOS)
            x=510+col*390
            image.paste(frame,(x+(360-frame.width)//2,y+5))
            t=case['source_timestamps_sec'][i];m=case['model_pair_labels_sec'][i//2]
            d.text((x,y+291),f'frame {i}: source {t:.2f}s / label {m:.1f}s',fill='#222222',font=font(14))
    image.save(OUTPUT/'case_overview.png');image.save(OUTPUT/'case_overview.pdf',resolution=150)


def main():
    if REPORT.exists() or DOCUMENT.exists():raise ValueError('Preserve an existing reviewed result')
    source=OUTPUT/'analysis.json';analysis=json.loads(source.read_text());cases=analysis['cases']
    assert {c['index'] for c in cases}==set(NOTES)
    for case in cases:
        span=case['spans'];source_times=np.asarray(case['source_timestamps_sec'])
        mapped=np.interp(span['Local'],np.arange(len(source_times)),source_times).tolist()
        mapped=[max(0,min(float(x),case['annotation_duration'])) for x in mapped]
        case['coordinate_only_diagnostic']={'mapping':'np.interp(saved_prediction_seconds, nominal_frame_indices, published_source_times), then official annotation-duration clipping; clamp outside frame grid',
            'mapped_span':mapped,'mapped_iou':iou(mapped,span['GT']),
            'inference_rerun':False,'primary_result_changed':False,
            'limitation':'Output-coordinate diagnostic only. Not the counterfactual of changing model input timestamps; not a new benchmark score.'}
        case['gt_source_frame_count']=int(sum(span['GT'][0]<=x<=span['GT'][1] for x in source_times))
        case['gt_model_pair_count']=len({i//2 for i,t in enumerate(source_times) if span['GT'][0]<=t<=span['GT'][1]})
        case['manual_review']={**NOTES[case['index']],
            'review_method':'Assistant inspected all six contact sheets and selected individual processor-RGB PNGs. No independent multi-rater labels or new raw-video decoding.',
            'human_review_is_ground_truth':False}
    local,reference=jsonl(LOCAL),jsonl(REFERENCE)
    accepted=lambda r:r['turns'][-1]['top1_is_gt'] and r['turns'][-1]['answer']=='matched' and r['turns'][-1]['time_parse_ok']
    lost=[a for a,b in zip(local,reference) if accepted(a) and accepted(b) and b['final_iou_r1']>=.7 and a['final_iou_r1']<.7]
    ratios=[(r['final_pred_end']-r['final_pred_start'])/(r['gt_end']-r['gt_start']) for r in lost]
    cohort={'definition':'Both accepted/parsed GT; author IoU>=0.7 and local IoU<0.7',
            'N':len(lost),'first_turn_only':sum(len(r['turns'])==1 for r in lost),
            'zero_overlap':sum(r['final_iou_r1']==0 for r in lost),
            'pred_duration_over_1point5x_GT':sum(x>1.5 for x in ratios),
            'pred_duration_under_2thirds_GT':sum(x<2/3 for x in ratios),
            'median_start_error_seconds':statistics.median(r['final_pred_start']-r['gt_start'] for r in lost),
            'median_end_error_seconds':statistics.median(r['final_pred_end']-r['gt_end'] for r in lost),
            'scope':'Descriptive outcome-selected cohort, not an unbiased test subset or a causal result.'}
    overview(cases)
    report={**analysis,'created_at':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'status':'CPU_CASE_VISUAL_REVIEW_COMPLETE','manual_visual_review_completed':True,
            'reconstruction_analysis':str(source),'reconstruction_analysis_sha256':sha(source),
            'cases':cases,'strict_localization_loss_cohort':cohort,
            'overview':str(OUTPUT/'case_overview.png'),'overview_pdf':str(OUTPUT/'case_overview.pdf'),
            'review_script_sha256':sha(Path(__file__)),
            'main_findings':['Query-specific event instances and opposite action stages need separation from whole-scene matching.',
                             'Small boundary shifts and annotation/source-time mismatches coexist with large semantic errors.',
                             'These six inputs are full-length, non-truncated reconstructions. Selected examples alone do not establish dataset-wide prevalence or a backend cause.'],
            'next_experiment':'Timestamp-only controlled evaluation, keeping pixels/model/backend/budgets/T/N/scoring fixed. Keep full-corpus membership. No such inference was run in this analysis.'}
    save(OUTPUT/'visual_review.json',report);save(REPORT,report)
    title='videosearch_r1 × CHARADES定位失败案例分析-11.05'
    lines=[f'# {title}', '',
           '本轮完成第一步：固定选取 5 个定位失败案例，加 1 个成功对照，重建当前流水线的视觉输入并逐帧检查。全程 CPU，无模型权重加载、前向推理或 GPU 任务；原始预测、标签、模型与配置保持不变。', '',
           '先打开[交互式逐帧浏览器](../outputs/charades_localization_cases_20261105/case_browser.html)，可切换案例、拖动帧、跳到 GT/本地/作者区间起止点、播放或展开全部帧。HTML 自包含，不需网络；浏览器直接打开本地文件即可。',
           '![六案例总览](../outputs/charades_localization_cases_20261105/case_overview.png)', '',
           '## 输入重建与证据边界', '',
           '使用固定官方提交及全量运行绑定的 NPY、metadata、processor 配置，在 CPU 复现 `_build_generation_inputs` 与实际 HF 视频处理。将 `pixel_values_videos` 的空间/时间 patch 排列还原，逆归一化成显示用 RGB；回打包张量完全一致，独立视频 placeholder 路径与完整 prompt 路径的视觉张量也完全一致。图像仅作展示缩放，没有增强、裁剪或生成内容。',
           '历史 GPU 运行没有保存视觉中间张量，因此这是冻结流水线的 CPU 重建，不冒称历史 GPU 截图或作者像素输入。HF 和 vLLM 的时间标签计算一致；没有再次抽帧、没有输入截断。两帧共用一个时间组标签不等于两帧被平均成一张图，两个原始帧仍在 temporal patch 中。', '',
           '## 选样与病例表', '',
           '前三例沿用此前已固定的极端差异案例；第四例按 GT 超出 annotation duration 且配对 IoU 下降最大选取；第五例按本地 IoU 最接近但低于 0.69 选取，排除纯浮点阈值；成功对照取共同达到 0.7、GT 持续 3.5–10 秒且不越界的最小 index。规则在看图前保存于 selection.json。五个失败例都在首轮接受 GT，没有进入 SQR；这是有目的的案例诊断，不是无偏抽样。', '',
           '| index / 视频 | GT（秒） | 本地（秒） | 作者（秒） | 本地 / 作者 IoU | 帧数 / 视觉 token |',
           '| --- | --- | --- | --- | --- | --- |']
    for c in cases:
        spans=c['spans'];fmt=lambda x:f'{x[0]:g}–{x[1]:g}'
        link='../outputs/charades_localization_cases_20261105/'+c['case_directory']+'/contact_sheet.png'
        lines.append(f'| [{c["index"]} / {c["video_id"]}]({link}) | {fmt(spans["GT"])} | {fmt(spans["Local"])} | {fmt(spans["Author"])} | {c["local_iou"]:.3f} / {c["author_iou"]:.3f} | {len(c["source_timestamps_sec"])} / {c["visual_tokens"]} |')
    for c in cases:
        note=c['manual_review'];diagnostic=c['coordinate_only_diagnostic']
        lines += ['',f'## 案例 {c["index"]}：{c["video_id"]}', '',f'原始查询：{c["query"]}', '']
        for s in note['observations']:lines.append('- '+s)
        lines += ['', '**待验证解释：**'+note['hypothesis'], '', '**观察限制：**'+note['limits'], '',
                  f'GT 区间内有 {c["gt_source_frame_count"]} 个源时刻对应的输入帧，涉及 {c["gt_model_pair_count"]} 个时间组；总输入 {c["input_tokens"]} tokens，未达到 20384 上限。',
                  f'仅将已保存输出秒数按 metadata 映射到源时钟，再按原 duration 裁剪，诊断 IoU 为 {diagnostic["mapped_iou"]:.4f}，原值 {c["local_iou"]:.4f}。这是输出坐标诊断，没有重跑模型，不能代替输入时间戳单变量实验或当作新成绩。', '',
                  f'[完整帧图](../outputs/charades_localization_cases_20261105/{c["case_directory"]}/contact_sheet.png) · [PDF](../outputs/charades_localization_cases_20261105/{c["case_directory"]}/contact_sheet.pdf) · [帧与时间标签 CSV](../outputs/charades_localization_cases_20261105/{c["case_directory"]}/frames.csv)']
    lines += ['', '## 共同问题与总体范围', '',
              '案例提示三个可分离的问题：场景/人物匹配后没有准确切分目标动作实例；相反动作或准备阶段与目标动作混淆；小幅边界差、稀疏采样及标注时长问题。没有发现这六例因视频后半段缺失、再次抽帧或上下文截断而丢失 GT。没有证据证明单独增加视觉 token 就能解决这些错误。',
              f'为避免被极端案例误导，另统计全部 79 条“双方接受 GT、作者命中 0.7、本地未中”的定位损失：{cohort["first_turn_only"]} 条首轮结束，只有 {cohort["zero_overlap"]} 条完全无重叠；预测长度 >1.5 倍 GT 的有 {cohort["pred_duration_over_1point5x_GT"]} 条，<2/3 的有 {cohort["pred_duration_under_2thirds_GT"]} 条；起点/终点误差中位数分别为 {cohort["median_start_error_seconds"]:.1f}/{cohort["median_end_error_seconds"]:.1f} 秒。总体上多数不是完全认错时段，小幅偏早和边界精度也值得优先验证。该子集按结果选出，不能作总体因果估计。', '',
              '## 后续验证顺序', '',
              '1. 时间戳唯一变量对照：同一像素 NPY、模型、视觉预算、后端、T/N、完整候选库与主计分，只切换名义时间/源时间标签。使用预先固定的评测范围，不按逐题 GT 调整补丁。',
              '2. 在上述结果基础上，再单独研究动作证据约束、实例分割或局部边界复核，尤其区分开/关、拿着/饮用、准备/动作本体；先验证单项，再组合。',
              '3. 局部更密集采样需要更密集的真实源视频帧；重复或插值现有 NPY 不会增加新动作证据。先确认资产可用性，再与增加 token/更换后端分开比较。', '',
              '[机器可读案例与观察](../reports/39_charades_localization_cases.json) · [六例总览 PDF](../outputs/charades_localization_cases_20261105/case_overview.pdf) · [重建脚本](../scripts/inspect_charades_localization_cases.py) · [观察记录与汇总脚本](../scripts/review_charades_localization_cases.py)', '',
              '本轮未修改主实验分数，没有启动时间戳对照或组合优化的 GPU 推理。可视化资产保留在本机 outputs/，不自动纳入 Git。', '']
    DOCUMENT.write_text('\n'.join(lines))
    print(json.dumps({'status':report['status'],'document':str(DOCUMENT),'report':str(REPORT),'browser':report['browser'],'overview':report['overview'],
                      'coordinate_only_diagnostics':[{ 'index':c['index'],'original_iou':c['local_iou'],'mapped_iou':c['coordinate_only_diagnostic']['mapped_iou']} for c in cases]},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
