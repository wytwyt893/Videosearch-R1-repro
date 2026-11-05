#!/usr/bin/env python3
"""Compare a complete repaired ActivityNet run with the supplied paper Table 2.

Offline postprocessing only. Last-verification VER remains the primary metric;
first-verification VER is an explicitly separate diagnostic from the same run.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from run_activitynet_repair_smoke import ROOT, BASE, digest
from run_activitynet_full import write_json
from score_activitynet_protocol import score_rows

PAPER = {'VCMR IoU0.3 R@1': 33.8, 'VCMR IoU0.5 R@1': 22.3,
         'VCMR IoU0.7 R@1': 12.3, 'VER Acc': 83.3,
         'VR R@1': 61.1, 'VR R@5': 81.7, 'VR R@10': 88.5, 'VR R@100': 98.4}


def compare(report_path):
    manifest = json.loads(report_path.read_text())
    if manifest['status'] != 'complete' or manifest['run_kind'] != 'repaired_full' or not manifest['resource_release_passed']:
        raise ValueError('Require a complete full run with released GPU resources')
    output = Path(manifest['output_dir'])
    for key, hash_key in (('predictions','prediction_sha256'),
                          ('primary_result','primary_result_sha256'), ('input_audit','input_audit_sha256')):
        if digest(manifest[key]) != manifest[hash_key]:
            raise ValueError('Completed run evidence changed')
    destination = output/'paper_table2_comparison.json'
    if destination.exists():
        raise ValueError('Preserve previous analysis; choose a new analysis version')
    rows = [json.loads(x) for x in Path(manifest['predictions']).read_text().splitlines() if x.strip()]
    rescored = score_rows(rows, 16966, include_first_turn=True)
    if rescored['primary'] != manifest['primary']:
        raise ValueError('Offline re-scoring disagrees with the primary full-run report')
    primary = rescored['primary']
    local = {'VCMR IoU0.3 R@1': 100*primary['VCMR']['IoU@0.3@R1'],
             'VCMR IoU0.5 R@1': 100*primary['VCMR']['IoU@0.5@R1'],
             'VCMR IoU0.7 R@1': 100*primary['VCMR']['IoU@0.7@R1'],
             'VER Acc': 100*primary['VER']['accuracy']}
    local.update({f'VR R@{k}': 100*primary['VR'][f'R@{k}'] for k in (1,5,10,100)})
    table = [{'metric': key, 'paper_percent': value, 'local_percent': local[key],
              'local_minus_paper_percentage_points': local[key]-value,
              'primary_ver_round': 'last_verified' if key=='VER Acc' else None}
             for key, value in PAPER.items()]
    stamp = datetime.datetime.fromisoformat(manifest['created_at']).strftime('%Y%m%d_%H%M%S')
    document = ROOT/f'doc/ACTIVITYNET_STAGE2_REPAIRED_FULL_{stamp}.md'
    if document.exists():
        raise ValueError('Choose a fresh experiment document')
    analysis = {'created_at': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                'status': 'PAPER_COMPARISON_COMPLETE', 'source_report': str(report_path),
                'paper_source': 'https://arxiv.org/abs/2607.00446',
                'paper_table': 'Table 2, ActivityNet-FIG, VideoSearch-R1 row; values supplied in the user paper screenshot',
                'table': table, 'primary_protocol': rescored['protocol'],
                'supplementary_first_turn_VER': rescored['supplementary']['first_turn_VER'],
                'trajectory_diagnostics': rescored['diagnostics'],
                'num_examples': len(rows), 'num_corpus_videos': 4866,
                'prediction_sha256': manifest['prediction_sha256'],
                'comparability': {
                    'full_original_question_and_corpus_membership': True,
                    'original_labels_and_released_2B_checkpoint_retained': True,
                    'repaired_video_pixels': 364, 'source_time_transport_applied_to_all_videos': True,
                    'last_verification_is_primary': True,
                    'paper_verification_round_not_confirmed': True,
                    'first_verification_is_diagnostic_only': True,
                    'claim': 'Comparison of a declared repaired-input checkpoint evaluation. Matching scores alone do not establish an exact author-input or training reproduction.'},
                'document': str(document)}
    write_json(destination, analysis)
    with (output/'paper_table2_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    first = analysis['supplementary_first_turn_VER']
    hours = manifest['elapsed_seconds']/3600
    lines = [f'# ActivityNet-FIG Stage2 修复版全量结果（{stamp}）', '',
             f'本次完成原始测试集全部 **16,966 题 / 4,866 个候选视频**，使用 `{manifest["variant"]}`。',
             '这是官方发布 2B Stage2 权重的推理评测；364 个异常视频使用重新采样的画面，全部视频使用源时间轴。原始标签、检索向量和候选库成员保留。', '',
             '| 指标 | 论文 Table 2（%） | 本次（%） | 本次−论文（百分点） |',
             '| --- | ---: | ---: | ---: |']
    for row in table:
        label = row['metric']+('（末次验证，主口径）' if row['metric']=='VER Acc' else '')
        lines.append(f'| {label} | {row["paper_percent"]:.1f} | {row["local_percent"]:.4f} | {row["local_minus_paper_percentage_points"]:+.4f} |')
    lines += ['', f'末次 VER：{primary["VER"]["correct"]}/{primary["VER"]["N"]}；有效回答覆盖率 {100*primary["VER"]["coverage"]:.4f}%，无效回答 {primary["VER"]["invalid_answers"]}。',
              f'若将无效回答按错误计入全部测试题，末次 VER 为 {100*primary["VER"]["accuracy_all_examples"]:.4f}%（分母 {primary["VER"]["N_expected"]}）；VR 与 VCMR 始终以全部测试题为分母。',
              f'首轮 VER 仅作离线诊断：{100*first["accuracy"]:.4f}%（{first["correct"]}/{first["N"]}）。论文取轮未确认，不能凭接近论文分数改变主口径。', '',
              'VR 使用最后一次查询更新后的排名；VER 配对最后实际验证的视频；VCMR 使用最后实际验证、定位的视频及其 IoU。保持 T=2，末轮触发 SQR 时继续官方重排，不增加第三次验证。', '',
              f'运行设备：GPU{manifest["gpu_index"]}，{manifest["gpu_name"]}，单卡、BF16、vLLM、batch1、T=2、8 soft tokens。',
              f'推理进程耗时 {hours:.4f} 小时；设备显存采样峰值 {manifest["peak_gpu_used_mib"]} MiB，含 vLLM 缓存预留。结束后的 GPU 释放检查通过。',
              f'跳题计数：`{json.dumps(manifest["skip"],ensure_ascii=False)}`。',
              f'视频元数据：`{json.dumps(manifest["video_meta_stats"],ensure_ascii=False)}`。', '',
              '原始数据的 114 题标注时长边界异常及已知短视频异常保留。67 个修复来源是较低帧率的镜像版本，其中两例通过显式视觉复核；不声称镜像文件与作者原视频逐字节一致。源时间轴中，364 个来自新解码 PTS，其余 4,502 个沿用发布的源采样元数据。',
              '因此，本表用于比较修复版与论文的结果，不能把差值全部归因于模型误差，也不能据此声称完整训练流程已经复现。', '',
              f'- [论文](https://arxiv.org/abs/2607.00446)，Table 2 ActivityNet-FIG / VideoSearch-R1。',
              f'- [运行清单]({(output/"run_manifest.json").as_posix()})',
              f'- [主口径计分]({manifest["primary_result"]})',
              f'- [逐题预测]({manifest["predictions"]})',
              f'- [论文对照 JSON]({destination.as_posix()})',
              f'- [输入审计]({manifest["input_audit"]})',
              f'- [计时]({(output/"timing.json").as_posix()})',
              f'- [推理日志]({(output/"inference.log").as_posix()})', '',
              f'预测 SHA256：`{manifest["prediction_sha256"]}`。', '']
    if manifest.get('postprocessing_recovery'):
        recovery = manifest['postprocessing_recovery']
        lines += ['## 收尾恢复与时间记录', '',
                  f'推理于 {manifest.get("inference_completed_at", "见日志")} 完成，模型进程退出码为 0。',
                  f'原收尾脚本错误地要求全量 VER 零无效回答，遇到 {primary["VER"]["invalid_answers"]} 条无法解析的回答后退出。',
                  '本次只做 CPU 离线核验、重新计分与文档生成，未重跑模型，未删除或改写预测。原失败清单、退出码和日志保留。',
                  f'恢复记录：[{Path(recovery["report"]).name}]({recovery["report"]})。',
                  '计时保留原进程实际总耗时；原收尾中断前未保存逐题时间事件，因此不补造逐题耗时。', '']
    precision = manifest.get('metric_precision_observation')
    if precision:
        lines += ['内部运行摘要与离线报告在 IoU=0.5 的边界上相差 1 题：保存的双精度 IoU 略低于 0.5，转为 float32 后等于 0.5。本表统一以原始 JSONL 和官方离线计分结果为准，未修改阈值或预测。', '']
    document.write_text('\n'.join(lines))
    return {'status': 'complete', 'json': str(destination), 'csv': str(output/'paper_table2_comparison.csv'),
            'document': str(document), 'analysis_sha256': digest(destination), 'document_sha256': digest(document)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.report), ensure_ascii=False, indent=2))
