# DiDeMo-FIG Stage2 smoke 与全量估时（2026-10-28）

**本次 10/10 题 smoke 通过。按相同单卡配置，全量 4013 题的中心估计约 4.1–4.6 小时，工程安排建议先预留 6 小时。全量尚未启动。**

本轮先完成 DiDeMo 专用配置、计分与 CPU 预检适配，再通过 tmux 在 GPU1 上运行官方评测入口。以下指标只用于工程验收，不能作为论文 Table 2 的全量复现结果。

## 运行与固定设置

| 项目 | 本次值 |
|---|---|
| 运行目录 | `outputs/didemo_final_smoke_20261028_020857_537290/` |
| GPU | 物理 GPU1，NVIDIA A100-SXM4-80GB，单卡 |
| GPU UUID | `GPU-677cef2a-aeb8-267d-4a59-19e72f682efc` |
| VLM | 官方 `VideoSearchR1/didemo-stage2`，配置与权重为 2B 架构 |
| Stage2 revision | `a576c16040c35a5ec71468c66ec7628e7db71997` |
| 检索器 | 本地 `Qwen/Qwen3-VL-Embedding-2B`，复用已核验固定 revision |
| 官方源码 commit | `69234fd9783b7235164d9f92b8903d9781dc284e`，工作树干净 |
| 样本 | 官方测试集原始 index 0–9，未换题 |
| 检索输入 | 完整 1002 视频库和 4013 条 query 向量 |
| 主要参数 | T=2、N=8、BF16、vLLM、batch=1、TP=1、eager=True、memory utilization=0.45 |
| 输入 / 生成预算 | 20384 / 1024 tokens；vLLM 上下文 21408 |
| 视觉预算 | 保留官方处理；没有新增 4096 个推理视觉 token 的硬上限 |
| 计分 | 末次实际验证 VER、最终排名 VR、官方 JSONL 离线 helper 的 VCMR |
| 末轮行为 | 触发时执行末轮 SQR 与重排，不增加第三次验证 |
| 随机种子 | 外层 seed=42；官方 vLLM 构造器未传入该 seed，engine 使用默认 seed=0 |

smoke 与全量的实际 worker 参数只相差 `--max_samples=10` 与 `0`；检索库不缩小。本次命令：

```bash
cd /data/wyt/videosearch_r1_repro
source scripts/env.sh
python -B scripts/run_didemo_full.py --execute --smoke --gpu 1
```

## 验收结果

| 项目 | 结果 |
|---|---|
| 有效预测 | 10/10，原始 index 唯一且完整 |
| 跳题 / 元数据缺失 | 全部为 0；12/12 次视频元数据命中 |
| VER 可解析覆盖率 | 100%，无无效回答 |
| SQR 覆盖 | 2 题实际使用，共应用 3 次；每次 shape 为 `[8, 2048]` |
| 计分输出 | 官方 report 与本地同轮配对的主协议 JSON 均生成 |
| 退出码 | 推理及外层 runner 均为 0 |
| 设备显存采样峰值 | 46485 MiB，约 45.40 GiB，包含 vLLM 缓存预留，不代表最低显存需求 |
| 退出后的 GPU1 | 连续空闲采样通过，显存 0 MiB、利用率 0%，无本任务计算进程 |
| Charades 历史代码 | 适配前后 5 个原有配置/脚本 SHA256 一致，未修改原始流程 |

10 题的主 VER=80%，VR R@1=70%，VCMR IoU≥0.3/0.5/0.7 R@1=50%/40%/40%。这些是小样本链路诊断，不能据此判断论文分数是否复现。

官方退出日志保留了未显式 `destroy_process_group()` 的 NCCL 提示；实际连续 GPU 采样确认资源已释放。

## 耗时与全量估算

| 阶段 | 实测 |
|---|---:|
| 模型初始化与准备 | 31.56 秒 |
| 官方逐题循环，共 10 题 | 40.67 秒 |
| 进程退出阶段 | 2.20 秒 |
| 推理子进程合计 | 74.43 秒 |
| 第一题 | 7.45 秒 |
| 后续 9 题平均 | 3.69 秒/题 |

外层 GPU 映射检查、计分与资源释放验收另计；CPU 预检也不计入上述推理子进程时间。

全量估计使用两种方法交叉参考：

1. 初始化、退出和第一题超出稳态均值的部分各计一次，再用后续题均值乘以 4013，另预留 60 秒计分：约 **4.14 小时**。
2. 用全部 10 题的循环平均耗时外推 4013 题，再加一次初始化/退出和计分预留：约 **4.56 小时**。

据此建议先预留 **6 小时**。该值是调度预留，不是完成时间保证。当前只有前 10 题，其中仅 2 题需要继续验证；全量的长视频比例、生成长度和二轮比例可能不同。第一题较慢也不能全部归为纯 warmup。

单题观测 p10/p90 对应的极端外推场景约 2.11–8.88 小时，**不是统计置信区间**。正式运行后应以更大样本的实际吞吐更新 ETA。若改变 batch、视觉输入、时间轴、生成后端或轮数，应重新估时。

## CPU 预检发现的边界

本次预检覆盖 40 项实际传参、42 项依赖 pins、模型/检索器及完整测试输入的当前 SHA256、1002 个像素数组、query/video 映射、FAISS 与实际 CPU processor 边界样本。9 项计分夹具、4 项计时/释放检查通过。vLLM 兼容派生权重通过官方 CPU helper 生成，逐个保留 tensor 与原始权重完全一致，仅移除 16 个 SQR tensors。

- 14 条标注的 `fig_desc` 带首尾空格。按官方入口的 `.strip()` 方式核对映射与计分，没有改写原始标注文件。
- 1 组同文 query 导致原始 index 1067 使用官方文本查找覆盖后的 qid/向量映射。保留完整测试题，以原始 index 及各题 GT 视频、时间计分。
- 141 条标注结束时间超过 annotation duration 0.1 秒以上，最大约 2.95 秒。保留官方标注与预测裁剪，不修正或删除测试题。

### 时间轴差异：保留官方行为，尚不宣称完全对齐论文

**官方预抽帧 NPY 的 metadata 入口返回 target fps=1 和连续帧索引，并未转发真实源视频抽样时间戳。** 因此，长视频实际源时间跨度与模型收到的名义时间轴可能不同。

CPU 实测的最长视频：

| 项目 | 数值 |
|---|---:|
| 源视频时长 | 180.013 秒 |
| 最后一张抽样帧的源时间 | 179.980 秒 |
| 保存帧数 | 64 |
| 官方 NPY reader 传出的最后时间 | 63.000 秒，fps=1 |

本轮为官方发布 checkpoint 的基线 smoke，**保留该官方实现，没有插入未经声明的时间轴修正**。该行为及估时假设已记录于 CPU 预检和 smoke 摘要。速度估计可以用于当前配置；其存在不等于已经证明论文实验使用了另一种实现，也不能仅凭本次 10 题判断它对分数的影响。

后续追查时间定位指标时，应将“官方原样基线”与“真实时间戳透传”的受控对照分开记录。当前没有执行该对照，也没有启动全量推理。

## 文件与复核入口

- [配置检查](../reports/14_didemo_alignment.json)、[完整 CPU 预检](../reports/15_didemo_cpu_preflight.json)
- [完整 smoke 运行摘要](../reports/16_didemo_final_smoke.json)、[精简结果与估时](../reports/17_didemo_smoke_summary.json)
- [预测 JSONL](../outputs/didemo_final_smoke_20261028_020857_537290/external_verified_test_temporal_grounding_VideoSearchR1__didemo-stage2.jsonl)
- [主协议计分](../outputs/didemo_final_smoke_20261028_020857_537290/external_verified_test_temporal_grounding_VideoSearchR1__didemo-stage2.primary.result.json)
- [推理日志](../outputs/didemo_final_smoke_20261028_020857_537290/inference.log)、[逐题计时](../outputs/didemo_final_smoke_20261028_020857_537290/timing.json)、[显存采样](../outputs/didemo_final_smoke_20261028_020857_537290/gpu_memory.csv)
- [正式参数](../configs/didemo_stage2_eval.bash)、[同一 smoke/全量 runner](../scripts/run_didemo_full.py)、[DiDeMo 主协议 scorer](../scripts/score_didemo_protocol.py)

当前输入的 CPU 预检与最终 smoke 已通过 runner 的绑定检查。复查入口默认只做 CPU dry run：

```bash
source scripts/env.sh
python -B scripts/run_didemo_full.py --dry-run
```

未来全量的显式命令如下；本轮未执行：

```bash
python -B scripts/run_didemo_full.py --execute --gpu 1
```
