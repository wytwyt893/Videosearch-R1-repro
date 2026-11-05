#!/usr/bin/env python3
"""Build the three-dataset Table 2 overview from versioned result summaries.

Standard library only; no model execution, downloads or raw-output dependencies.
"""
from __future__ import annotations
import csv
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATE = '2026-11-05'
TITLE = 'videosearch_r1 × 三数据集全量评估汇总-11.05'
DOC = ROOT/'doc'/(TITLE+'.md')
JSON = ROOT/'reports/38_three_dataset_summary.json'
CSV = ROOT/'reports/38_three_dataset_table2.csv'
PAPER = {
    'Charades-FIG':[16.5,13.4,8.2,75.7,24.6,42.9,52.2,84.7],
    'DiDeMo-FIG':[33.3,30.2,19.7,74.6,59.0,82.0,87.8,97.5],
    'ActivityNet-FIG':[33.8,22.3,12.3,83.3,61.1,81.7,88.5,98.4],
}
METRICS = ['VCMR_0.3_R@1','VCMR_0.5_R@1','VCMR_0.7_R@1','VER_Acc',
           'VR_R@1','VR_R@5','VR_R@10','VR_R@100']


def read(relative):
    return json.loads((ROOT/relative).read_text())


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(primary):
    return [100*primary['VCMR'][f'IoU@{t}@R1'] for t in (.3,.5,.7)]+[
        100*primary['VER']['accuracy']]+[100*primary['VR'][f'R@{k}'] for k in (1,5,10,100)]


def duration(seconds):
    seconds=round(seconds)
    return f'{seconds//3600} 小时 {(seconds%3600)//60} 分 {seconds%60} 秒'


def build():
    if any(p.exists() for p in (DOC,JSON,CSV)):
        raise ValueError('Preserve an existing overview; choose a new dated version')
    names = {r['old_path'].split('/')[-1]:Path(r['new_path']).name
             for r in read('reports/37_document_renames.json')['renames']}
    sources = [('Charades-FIG','reports/10_charades_full_summary.json',1334,
                '官方发布像素与名义时间轴','CHARADES_STAGE2_FULL_20261027.md'),
               ('DiDeMo-FIG','reports/18_didemo_full_summary.json',1002,
                '官方发布像素与名义时间轴','DIDEMO_STAGE2_FULL_20261028.md'),
               ('ActivityNet-FIG','reports/36b_activitynet_full_summary.json',4866,
                '364 个修复像素视频；全部视频透传源时间轴','ACTIVITYNET_STAGE2_REPAIRED_FULL_20261030_004132.md')]
    experiments=[]
    for dataset,source,videos,variant,old_doc in sources:
        data=read(source)
        released=data.get('resource_release_passed',data.get('gpu',{}).get('released',False))
        if data['status'] not in {'complete','DIDEMO_STAGE2_FULL_COMPLETE'} or not released:
            raise ValueError(f'Unfinished source experiment: {dataset}')
        primary=data['primary']
        first=(read('reports/11_charades_offline_summary.json')['local_first_turn_VER']
               if dataset=='Charades-FIG' else data['supplementary']['first_turn_VER'])
        values=metrics(primary)
        if primary['VER']['N']+primary['VER']['invalid_answers']!=primary['VR']['N']:
            raise ValueError('VER denominator does not account for every test example')
        if primary['VR']['N']!=primary['VCMR']['N']:
            raise ValueError('VR/VCMR corpora differ')
        elapsed=(data['timing_seconds']['process_seconds'] if dataset=='DiDeMo-FIG'
                 else data['elapsed_seconds'])
        experiments.append({'dataset':dataset,'questions':primary['VR']['N'],'videos':videos,
                            'data_variant':variant,'source_report':source,
                            'source_report_sha256':checksum(ROOT/source),
                            'document':'doc/'+names[old_doc],
                            'primary':primary,'first_turn_VER':first,
                            'paper_percent':dict(zip(METRICS,PAPER[dataset])),
                            'local_percent':dict(zip(METRICS,values)),
                            'local_minus_paper_percentage_points':dict(zip(METRICS,[a-b for a,b in zip(values,PAPER[dataset])])),
                            'elapsed_seconds':elapsed})
    summary={'compiled_on':DATE,'scope':'Released Stage2 checkpoint inference on three complete test sets; other baselines and training reproduction are not included.',
             'paper_source':'https://arxiv.org/html/2607.00446v1#S4.T2',
             'paper_table_values_source':'Original Table 2 screenshot supplied by the user; VideoSearch-R1 row only.',
             'metric_column_order':METRICS,'units':'percent; differences in percentage points',
             'protocol':{'T':2,'soft_tokens':8,'primary_VER':'last_verified','VER_denominator':'valid_answers; invalids and all-example accuracy retained',
                         'VR':'final ranking after official terminal update','VCMR':'last actually verified/localized video',
                         'paper_VER_round_confirmed':False},
             'total_evaluated_questions':sum(e['questions'] for e in experiments),
             'experiments':experiments,'document':str(DOC.relative_to(ROOT))}
    JSON.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with CSV.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['Dataset','Method',*METRICS],lineterminator='\n')
        writer.writeheader()
        for e in experiments:
            for method,key in [('VideoSearch-R1 (paper)','paper_percent'),('VideoSearch-R1 (local; last VER)','local_percent')]:
                writer.writerow({'Dataset':e['dataset'],'Method':method,**e[key]})
    lines=[f'# {TITLE}', '',
           '整理日期：2026-11-05；时间均为北京时间。本轮完成 Charades-FIG、DiDeMo-FIG、ActivityNet-FIG 三个完整测试集，共 **24,699 题**的发布权重推理评测。',
           '对应[论文 Table 2](https://arxiv.org/html/2607.00446v1#S4.T2) 的 **VideoSearch-R1 行**。下表保留原表的 Dataset、Method、VCMR、VER、VR 分组及列顺序；其他 baseline 尚未复现。', '',
           '百分比：论文保留原表一位小数，本地保留两位小数。本地主 VER 始终为预先确定的末次实际验证；论文多轮取轮未明确确认，VER 数值并列不代表其口径已被证明相同。', '',
           '<table>', '<thead>',
           '<tr><th rowspan="2">Dataset</th><th rowspan="2">Method</th><th colspan="3">VCMR</th><th>VER</th><th colspan="4">VR</th></tr>',
           '<tr><th>0.3/R@1</th><th>0.5/R@1</th><th>0.7/R@1</th><th>Acc</th><th>R@1</th><th>R@5</th><th>R@10</th><th>R@100</th></tr>',
           '</thead>', '<tbody>']
    for e in experiments:
        paper=''.join(f'<td>{v:.1f}</td>' for v in e['paper_percent'].values())
        local=''.join(f'<td><strong>{v:.2f}</strong></td>' for v in e['local_percent'].values())
        label='VideoSearch-R1（本地修复版†）' if e['dataset']=='ActivityNet-FIG' else 'VideoSearch-R1（本地）'
        lines += [f'<tr><td rowspan="2">{html.escape(e["dataset"])}</td><td>VideoSearch-R1（论文）</td>{paper}</tr>',
                  f'<tr><td>{label}</td>{local}</tr>']
    lines += ['</tbody>', '</table>', '',
              '† ActivityNet 本地结果使用修复版：364 个异常视频从公开镜像完整解码后重新采样，全部 4,866 个视频使用源时间轴；其中 364 个使用新解码 PTS，其他 4,502 个使用发布的源采样元数据。Charades、DiDeMo 本次保留官方发布像素与名义时间轴。三个测试集都保留原始题目和完整检索库，但输入修复方式不同，不能把本轮当作三数据集上的统一修复消融。', '',
              '## VER 取轮、分母与补充结果', '',
              '| 数据集 | 论文 VER | 本地末次（主指标） | 本地首轮（补充） | 主 VER 正确/有效 | 无效回答 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for e in experiments:
        ver=e['primary']['VER'];first=e['first_turn_VER']
        lines.append(f'| {e["dataset"]} | {e["paper_percent"]["VER_Acc"]:.1f}% | {100*ver["accuracy"]:.2f}% | {100*first["accuracy"]:.2f}% | {ver["correct"]}/{ver["N"]} | {ver["invalid_answers"]} |')
    lines += ['',
              '末次 VER 与该轮实际检查的视频配对；第二轮后若重排得到新视频，不把旧判断配给新视频。VR 使用最后更新后的排名，VCMR 使用最后实际检查、定位的视频。保持 T=2、N=8，触发时执行官方末轮查询更新与重排，不增加第三次验证。',
              'ActivityNet 唯一无效末次回答位于原始 index 10237：主 VER 分母为 16,965；若将它按错误计入全部 16,966 题，末次 VER 为 83.0249%。VR、VCMR 分母始终为全部题数。',
              'Charades、DiDeMo 的作者公开预测，其首轮 VER 更接近论文数值；DiDeMo 附件实际为 T=3。已保存的前两轮快照只能作离线诊断，不能代替新的 T=2 受控实验。我们没有按分数接近程度更换主 VER 口径。', '',
              '## 与论文的差值', '',
              '以下均为本地未四舍五入数值减论文公布值，单位为百分点；VER 取轮仍有上述限制。', '',
              '| 数据集 | VCMR 0.3 / 0.5 / 0.7 | 末次 VER | VR R@1 / R@5 / R@10 / R@100 |',
              '| --- | --- | ---: | --- |']
    for e in experiments:
        delta=list(e['local_minus_paper_percentage_points'].values())
        lines.append(f'| {e["dataset"]} | '+ ' / '.join(f'{v:+.2f}' for v in delta[:3])+f' | {delta[3]:+.2f} | '+' / '.join(f'{v:+.2f}' for v in delta[4:])+' |')
    lines += ['',
              '三数据集的 VR 各项与论文最大绝对差约 0.18 个百分点。Charades、DiDeMo 的 VCMR 仍低于论文，已有离线对照主要指向正确接受视频后的时间定位差异；ActivityNet 修复版 VCMR 高于论文约 0.68–1.96 个百分点。缺少同配置原版全量对照，不能将其更高分数直接归因于修复。', '',
              '## 数据、运行与验收', '',
              '| 数据集 | 问题数 | 完整视频库 | 输入版本 | 单卡推理实测耗时 |',
              '| --- | ---: | ---: | --- | --- |']
    for e in experiments:
        lines.append(f'| {e["dataset"]} | {e["questions"]:,} | {e["videos"]:,} | {e["data_variant"]} | {duration(e["elapsed_seconds"])} |')
    lines += ['',
              '公共设置：各数据集官方发布 Stage2 2B VLM，复用 2B 检索编码器；单卡 A100 80GB、BF16、vLLM、batch=1、TP=1、eager、T=2、8 soft tokens。外层 seed=42，官方 vLLM 构造器未传入 seed，engine 默认 seed=0。保留官方推理视觉处理，不另加 4096 视觉 token 硬上限；论文的 4096 描述属于训练设置。',
              '三个完整测试集均无跳题、无视频元数据缺失，结束后 GPU 释放检查通过。ActivityNet 于 10 月 30 日启动、10 月 31 日完成，文件名保留启动日期 10.30。其推理返回码为 0，原收尾因错误套用 smoke 零无效回答检查而退出；随后仅用 CPU 补齐报告，原预测、错误日志和失败清单均保留。IoU=0.5 的一个 float32 边界以原始 JSONL 与官方离线计分为准。',
              '原始标注边界问题保留，ActivityNet 的两个镜像来源保留显式视觉复核例外。公开镜像编码不声称与作者原文件逐字节一致。完整训练、其他 baseline 和论文其他表格尚未完成。', '',
              '## 实验报告与机器可读结果', '']
    for e in experiments:
        lines.append(f'- {e["dataset"]}：[完整报告](<{Path(e["document"]).name}>)；[结果摘要](../{e["source_report"]})。')
    lines += ['- [Charades 最小离线检查](<'+names['CHARADES_STAGE2_OFFLINE_CHECK_20261027.md']+'>)；[DiDeMo 最小离线检查](<'+names['DIDEMO_STAGE2_OFFLINE_CHECK_20261028.md']+'>)。',
              '- [汇总 JSON](../reports/38_three_dataset_summary.json)、[Table 2 CSV](../reports/38_three_dataset_table2.csv)、[文件重命名映射与原始哈希](../reports/37_document_renames.json)。', '',
              '## 下一阶段研究起点', '',
              '1. **先分离数据修复因素。** ActivityNet 已保留原版、仅修复画面、修复画面并透传源时间轴三个视图。固定题目、权重、预算和评估协议，做配对对照，确认帧覆盖与时间轴各自的影响。',
              '2. **聚焦时间定位。** Charades、DiDeMo 的检索接近论文，而正确接受 GT 后的区间预测仍有差异。先检查时间标签与源采样，再研究预算分配或局部二次定位，分别验证各因素。',
              '3. **研究迭代成本与收益。** 已保存首轮/末次轨迹，可先离线分析何时 SQR 改善排名、何时需要继续验证，再设计自适应停止或更新门控，并同时报告准确率与推理成本。', '',
              '以上是由现有证据引出的候选问题，尚未证明方法新颖性，也没有启动新的 GPU 实验；后续需查阅相关工作并设计受控实验。', '']
    DOC.write_text('\n'.join(lines))
    print(json.dumps({'document':str(DOC),'json':str(JSON),'csv':str(CSV),
                      'questions':summary['total_evaluated_questions'],'datasets':len(experiments)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    build()
