# videosearch_r1 × ACTIVITYNET修复版全量评估-10.30

整理说明（2026-11-05）：文件日期沿用原实验/报告日期，正文保留当时阶段的记录；当前三数据集结果见[全量评估汇总](<videosearch_r1 × 三数据集全量评估汇总-11.05.md>)。

本次完成原始测试集全部 **16,966 题 / 4,866 个候选视频**，使用 `source_time_repaired_subset`。
这是官方发布 2B Stage2 权重的推理评测；364 个异常视频使用重新采样的画面，全部视频使用源时间轴。原始标签、检索向量和候选库成员保留。

| 指标 | 论文 Table 2（%） | 本次（%） | 本次−论文（百分点） |
| --- | ---: | ---: | ---: |
| VCMR IoU0.3 R@1 | 33.8 | 35.7598 | +1.9598 |
| VCMR IoU0.5 R@1 | 22.3 | 23.5648 | +1.2648 |
| VCMR IoU0.7 R@1 | 12.3 | 12.9848 | +0.6848 |
| VER Acc（末次验证，主口径） | 83.3 | 83.0298 | -0.2702 |
| VR R@1 | 61.1 | 61.1989 | +0.0989 |
| VR R@5 | 81.7 | 81.6751 | -0.0249 |
| VR R@10 | 88.5 | 88.4180 | -0.0820 |
| VR R@100 | 98.4 | 98.3909 | -0.0091 |

末次 VER：14086/16965；有效回答覆盖率 99.9941%，无效回答 1。
若将无效回答按错误计入全部测试题，末次 VER 为 83.0249%（分母 16966）；VR 与 VCMR 始终以全部测试题为分母。
首轮 VER 仅作离线诊断：84.8285%（14392/16966）。论文取轮未确认，不能凭接近论文分数改变主口径。

VR 使用最后一次查询更新后的排名；VER 配对最后实际验证的视频；VCMR 使用最后实际验证、定位的视频及其 IoU。保持 T=2，末轮触发 SQR 时继续官方重排，不增加第三次验证。

运行设备：GPU1，NVIDIA A100-SXM4-80GB，单卡、BF16、vLLM、batch1、T=2、8 soft tokens。
推理进程耗时 27.8346 小时；设备显存采样峰值 73207 MiB，含 vLLM 缓存预留。结束后的 GPU 释放检查通过。
跳题计数：`{"query_not_found_in_meta": 0, "pos_doc_missing": 0, "video_file_missing": 0, "gt_time_invalid": 0}`。
视频元数据：`{"npy_or_npz_inputs": 24602, "meta_payload_hits": 24602, "meta_payload_misses": 0}`。

原始数据的 114 题标注时长边界异常及已知短视频异常保留。67 个修复来源是较低帧率的镜像版本，其中两例通过显式视觉复核；不声称镜像文件与作者原视频逐字节一致。源时间轴中，364 个来自新解码 PTS，其余 4,502 个沿用发布的源采样元数据。
因此，本表用于比较修复版与论文的结果，不能把差值全部归因于模型误差，也不能据此声称完整训练流程已经复现。

- [论文](https://arxiv.org/abs/2607.00446)，Table 2 ActivityNet-FIG / VideoSearch-R1。
- [运行清单](/data/wyt/videosearch_r1_repro/outputs/activitynet_repair_full_20261030_004132_840892/run_manifest.json)
- [主口径计分](/data/wyt/videosearch_r1_repro/outputs/activitynet_repair_full_20261030_004132_840892/external_verified_test_temporal_grounding_VideoSearchR1__activitynet-stage2.primary.result.json)
- [逐题预测](/data/wyt/videosearch_r1_repro/outputs/activitynet_repair_full_20261030_004132_840892/external_verified_test_temporal_grounding_VideoSearchR1__activitynet-stage2.jsonl)
- [论文对照 JSON](/data/wyt/videosearch_r1_repro/outputs/activitynet_repair_full_20261030_004132_840892/paper_table2_comparison.json)
- [输入审计](/data/wyt/videosearch_r1_repro/outputs/activitynet_repair_full_20261030_004132_840892/input_audit.jsonl)
- [计时](/data/wyt/videosearch_r1_repro/outputs/activitynet_repair_full_20261030_004132_840892/timing.json)
- [推理日志](/data/wyt/videosearch_r1_repro/outputs/activitynet_repair_full_20261030_004132_840892/inference.log)

预测 SHA256：`2130c85aaf3e4275d33869fc6cffabd068606c22d1892acb078666298cd381fd`。

## 收尾恢复与时间记录

推理于 2026-10-31T04:31:34.632000+08:00 完成，模型进程退出码为 0。
原收尾脚本错误地要求全量 VER 零无效回答，遇到 1 条无法解析的回答后退出。
本次只做 CPU 离线核验、重新计分与文档生成，未重跑模型，未删除或改写预测。原失败清单、退出码和日志保留。
恢复记录：[36a_activitynet_full_postprocess_recovery.json](<../reports/36a_activitynet_full_postprocess_recovery.json>)。
计时保留原进程实际总耗时；原收尾中断前未保存逐题时间事件，因此不补造逐题耗时。

内部运行摘要与离线报告在 IoU=0.5 的边界上相差 1 题：保存的双精度 IoU 略低于 0.5，转为 float32 后等于 0.5。本表统一以原始 JSONL 和官方离线计分结果为准，未修改阈值或预测。
