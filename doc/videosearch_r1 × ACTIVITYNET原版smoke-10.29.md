# videosearch_r1 × ACTIVITYNET原版smoke-10.29

整理说明（2026-11-05）：文件日期沿用原实验/报告日期，正文保留当时阶段的记录；当前三数据集结果见[全量评估汇总](<videosearch_r1 × 三数据集全量评估汇总-11.05.md>)。

**完整资产、CPU 预检与 10/10 题 smoke 均通过。官方资产基线的推理、SQR、计分和退出释放链条已跑通；全量 16966 题尚未启动。** 单卡 A100 的全量外推约 28.5–29.8 小时，建议先预留 40 小时；10 题小样本估计不能保证实际完成时间。

**严格论文时间轴/帧覆盖对齐仍未确认。** 本轮确认 364 个 MKV 的 NPY 元数据与检索编码元数据存在帧数/采样跨度差异，官方未知帧数分支可复现前 64 帧选择。保留官方发布数组和既定协议进行 smoke，没有把它解释为已恢复论文的完整视频输入，也没有猜测修正 FPS、时间戳或 GT。

## 资产准备已完成

下载于北京时间 2026-10-28 15:28 开始、2026-10-29 01:59 左右验收完成，退出码 0。首次批次留下 26 个网络失败文件，第二批保留 4840 个完整文件，仅补齐缺失项，再进行整库验收。没有下载训练视频。

| 资产 | 验收结果 |
|---|---|
| 测试问题 | 16966，原始题全部保留 |
| 视频数组 | 4866/4866，151311175824 字节，约 140.92 GiB |
| 向量 | query `(16966, 2048)`、视频 `(4866, 2048)` |
| 模型 | 官方 ActivityNet Stage2，实际 2B 架构，14 文件官方大小/哈希通过，16 个 refine tensors |
| 固定 Stage2 revision | `918ca416dc3678d28947b5794056607b7a4656b3` |
| 检索器 | 复用 `Qwen/Qwen3-VL-Embedding-2B`，revision `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` |
| 完整性范围 | 模型核验官方 LFS/Git 哈希；视频核验 manifest 指定 ranges/大小、下载 SHA256 收据、数组与元数据；官方未提供逐视频独立 SHA256 |

## CPU 预检

实际命令在 tmux 会话 `videosearch_activitynet_cpu` 执行：

```bash
source scripts/env.sh
python -B scripts/preflight_activitynet_full.py
```

CPU 全程隐藏 CUDA，没有模型 forward 或 vLLM engine。耗时 **228.75 秒**。官方模块仅用于 CPU 兼容权重导出和 processor；没有构造推理模型。

| 检查 | 结果 |
|---|---|
| 官方固定源码 | `69234fd9783b7235164d9f92b8903d9781dc284e`，工作树干净 |
| 真实 shell 入口传参 | 40 项关键参数通过，单 worker、单 shard、完整库、离线模型 |
| 依赖 | 官方 42 项 pins、pip check、必要 CPU 导入通过 |
| 模型 | 原始模型/检索器全部当前 SHA256 通过；兼容副本逐个保留 tensor 与原权重完全相等，仅去除 16 个 SQR tensors |
| 完整测试输入 | 4866 个 RGB NPY 逐个哈希、shape、uint8、连续内存、偶数帧及元数据通过；零 `.partial` |
| 向量与索引 | 完整向量有限非零，FAISS 4866×2048、id_map/docid2row 对应 |
| 查询映射 | 16966/16966 对应，无同文 query 覆盖；17 条首尾空格按官方 `.strip()` 处理 |
| 计分与计时夹具 | 9 项计分、4 项计时/资源释放检查通过 |
| smoke 与全量入口 | worker 参数仅 `--max_samples=10` 与 `0` 不同 |

CPU 初始检索诊断为 9350/16966=55.11% R@1，仅使用发布向量，不是 Stage2 或论文 Table 2 结果。

### 实际 CPU processor 边界样本

使用所有查询的 tokenizer 长度选出最长文本（index 580，103 text tokens），检查最大视觉输入、最少帧、偶数补帧、原始第 0 题、最长源视频、最大时长差及 GT 超出源时长样本。表中源末时间来自发布 metadata，不能无条件当作原始完整视频的真实时间。

| 检查角色 | 视频 | 视觉 tokens | prompt tokens | 发布源末时间（秒） | reader 末时间（秒） |
|---|---|---:|---:|---:|---:|
| largest_visual_input | v_HHG1kCydLYU | 6688 | 7321 | 189.723 | 63.000 |
| fewest_frames | v_j73Wh1olDsA | 108 | 466 | 2.800 | 1.000 |
| even_padding | v_altXks0a0qY | 3456 | 3963 | 35.500 | 35.000 |
| original_question_zero | v_u3uYs6SZFKo | 3840 | 4312 | 39.967 | 39.000 |
| longest_source_video | v_H-5nHSHwFOk | 6080 | 6713 | 745.400 | 63.000 |
| largest_source_annotation_duration_gap | v_BrgYIg6UXhU | 6144 | 6777 | 2.625 | 63.000 |
| GT_end_furthest_beyond_source_duration | v_BrgYIg6UXhU | 6144 | 6697 | 2.625 | 63.000 |

全部样本未截断，prompt 均低于输入预算 20384；生成预算 1024，vLLM 上下文 21408。保留官方推理缩放，没有新增 4096 个推理视觉 token 硬上限；论文的 4096 明确描述训练设置。

## GPU1 smoke

```bash
source scripts/env.sh
python -B scripts/run_activitynet_full.py --execute --smoke --gpu 1
```

会话 `videosearch_activitynet_smoke`，结果目录 `outputs/activitynet_final_smoke_20261029_144528_236908/`。启动前空闲检查和 CUDA/NVML UUID 交叉核验通过。

| 项目 | 实测 |
|---|---|
| GPU | 单张 NVIDIA A100-SXM4-80GB，物理 GPU1 |
| UUID | `GPU-677cef2a-aeb8-267d-4a59-19e72f682efc` |
| 样本 | 官方原始 index 0–9，未换题，完整 4866 视频检索库 |
| 设置 | T=2、N=8、BF16、vLLM、batch=1、TP=1、eager=True、memory utilization=0.45 |
| seed | 外层 42；官方 LLM 构造器未传入该 seed，engine 实际日志 seed=0 |
| 有效预测 | 10/10，所有 skip=0，VER 无无效回答 |
| 视频元数据 | 14/14 次命中，缺失 0 |
| SQR | 4 题实际触发，共应用 7 次；全部 shape=`[8, 2048]` |
| 原始输出 CPU 复核 | 14 轮回答、起止时间与 IoU 重新解析全部一致 |
| 推理返回码 / launcher | 均为 0 |
| 峰值设备显存采样 | 48305 MiB，约 47.17 GiB，包含 vLLM 缓存预留，不代表最低需求 |
| 资源释放 | 连续两次空闲验收通过；随后独立复查两张 GPU 均 0 MiB，无计算进程 |
| 历史实验保护 | Charades/DiDeMo 配置、runner、scorer、预检和历史报告前后哈希一致 |

小样本诊断：末次 VER=90%，VR R@1/5/10/100=80/90/90/100%，VCMR IoU≥0.3/0.5/0.7 R@1=70/30/30%。**只用于工程验收，不用于判断 Table 2 分数复现程度。**

计分仍采用事先确定的末次实际验证 VER（同轮答案/视频配对）、最终排名 VR、最后实际检查视频的官方 `final_iou_r1` VCMR。第二轮触发时保留最后一次 SQR 与重排，不增加第三次验证。

退出日志保留官方 NCCL 未显式 destroy 的提示。首次资源查询的利用率仍反映退出前的采样窗口，后续连续空闲检查及独立复查确认 GPU 已释放；没有本轮推理进程残留，`tail` 窗格仅用于查看日志。

## 全量估时

| 阶段 | 实测 |
|---|---:|
| 初始化与准备 | 37.89 秒 |
| 10 题正式循环 | 63.15 秒 |
| 退出 | 2.15 秒 |
| 推理子进程合计 | 103.20 秒 |
| 第 1 题 | 8.80 秒 |
| 后续 9 题平均 | 6.05 秒/题 |

按后续均值乘完整 16966 题，初始化、退出和首题超出均值部分各计一次，另预留 60 秒计分，约 **28.52 小时**。按全部 10 题循环均值交叉估计约 **29.79 小时**。先预留 **40 小时**，不是完成保证。

当前样本仅原始前 10 题，其中 40% 使用 SQR；全量的视频大小、长视频比例、生成长度和二轮比例可能不同。观测 p10/p90 对应极端外推约 15.1–52.9 小时，不是统计置信区间。估时适用于当前单卡和官方资产/时间轴设置；修正帧覆盖或时间戳后需重新估时。

## 时间轴及 MKV 帧覆盖：尚未解决的论文对齐限制

普通长视频预抽帧 NPY 由官方 reader 以 target fps=1、连续帧号提供给 processor，未转发真实采样时间。最长样本源记录跨度约 745 秒，模型名义末帧时间仍为 63 秒。

对 364 个 MKV 的进一步审计还发现：

- 全部 NPY metadata 的源帧数与输出帧数都为 64，选择源索引 `0..63`；涉及 1273 条测试题。
- 相同视频路径的检索编码 metadata 全部记录了大于 64 的总源帧数，并采样到索引 63 之后。
- 官方 `_get_sample_indices` 在 `video_stream.frames==0` 时直接返回前 `maxlen` 帧索引；CPU 纯函数复现：未知帧数时选 `0..63`，已知 6409 帧时选跨 `0..6408` 的 64 帧。
- 例如 `v_BrgYIg6UXhU` 的 annotation duration=267.13 秒，NPY metadata 记录 64 帧 / 24 FPS=2.67 秒，检索编码 metadata 则记录总帧数 6409、采样末索引 6408。

**已确认发布记录之间存在帧覆盖差异，并且源码分支可解释这种模式；这支持未知帧数回退可能导致前 64 帧抽取的判断。** 本轮未取得原始视频，也未重现原始数据导出版本，不能把推断写成已验证的原始视频内容事实，更不能直接将 2.67 秒认定为原视频真实长度。

原标注还有 114 条 GT 结束点超过 annotation duration 0.1 秒以上；这些记录全部保留。本轮不猜测修改 GT、FPS 或张量，不为了通过预检删题。**CPU/smoke 通过表示官方发布资产的工程链路可运行，不证明严格论文帧覆盖/时间轴对齐。**

若正式目标是严格还原论文输入，下一步需核实 MKV 原视频与这两套采样记录；若继续官方发布资产基线，全量结果必须注明此限制。本轮没有启动全量，也没有擅自再导出一套视频数据。

## 可复核文件

- [资产验收](../reports/22_activitynet_assets.json)、[实际参数验收](../reports/23_activitynet_alignment.json)
- [完整 CPU 预检](../reports/24_activitynet_cpu_preflight.json)、[完整 smoke 记录](../reports/25_activitynet_final_smoke.json)
- [精简结果与估时](../reports/26_activitynet_smoke_summary.json)、[MKV/时间元数据审计](../reports/27_activitynet_time_metadata_audit.json)
- [运行参数](../configs/activitynet_stage2_eval.bash)、[CPU 预检](../scripts/preflight_activitynet_full.py)、[同入口 runner](../scripts/run_activitynet_full.py)、[主 scorer](../scripts/score_activitynet_protocol.py)
- [预测 JSONL](../outputs/activitynet_final_smoke_20261029_144528_236908/external_verified_test_temporal_grounding_VideoSearchR1__activitynet-stage2.jsonl)
- [主协议计分](../outputs/activitynet_final_smoke_20261029_144528_236908/external_verified_test_temporal_grounding_VideoSearchR1__activitynet-stage2.primary.result.json)
- [官方 report](../outputs/activitynet_final_smoke_20261029_144528_236908/external_verified_test_temporal_grounding_VideoSearchR1__activitynet-stage2.result.json)
- [推理日志](../outputs/activitynet_final_smoke_20261029_144528_236908/inference.log)
- [运行记录](../outputs/activitynet_final_smoke_20261029_144528_236908/run_manifest.json)
- [逐题计时](../outputs/activitynet_final_smoke_20261029_144528_236908/timing.json)
- [显存曲线 CSV](../outputs/activitynet_final_smoke_20261029_144528_236908/gpu_memory.csv)
- [退出后的 GPU / tmux 记录](../outputs/activitynet_final_smoke_20261029_144528_236908/postrun_resources.json)
- [smoke 原始输出解析与哈希复核](../outputs/activitynet_final_smoke_20261029_144528_236908/postrun_integrity.json)

预测 SHA256：`ca6516d2633de0950a0ef7366da073a23f4df938deb072b5a644d220f1092c8c`；主计分 SHA256：`b6354a04f5e686bb8ad7f42490a968e726d8ac9e41b485cc1125b8a568267255`。退出后重新验证当前 CPU/smoke 绑定仍通过。所有实际预测、日志和显存曲线保存在本机 `outputs/`，轻量摘要与报告可纳入 Git。
