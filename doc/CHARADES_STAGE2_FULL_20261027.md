# Charades-FIG Stage2 全量评测结果（2026-10-27）

本报告记录官方发布的 Charades Stage2 checkpoint 在完整测试集上的一次推理评测。**3720/3720 题完成，计分与完整性验收通过，任务正常退出，GPU 已释放。** VR 指标接近论文 Table 2；VCMR 在较严格 IoU 阈值下仍有差距；主 VER 按事先约定采用末次实际验证，论文的多轮取轮口径尚未确认。

本次完成的是 Table 2 中 **Charades-FIG / VideoSearch-R1** 对应的发布权重推理评测。没有执行 SFT/GRPO 训练，没有评测其他方法或其他数据集，因此不等同于完成整张 Table 2 或完整训练复现。

## 1. 运行信息与来源

所有时间使用北京时间（Asia/Shanghai）。

| 项目 | 本次记录 |
|---|---|
| 工作目录 | `/data/wyt/videosearch_r1_repro` |
| 运行标识 | `charades_full_20261027_161353_178368` |
| 输出目录创建时间 | 2026-10-27 16:13:53，取自运行目录时间标识 |
| launcher 退出时间 | 2026-10-27 20:47:48，取自退出码文件修改时间 |
| 启动方式 | tmux 后台运行，会话 `videosearch_charades_full` |
| GPU | 物理 GPU1，NVIDIA A100-SXM4-80GB，单卡 |
| GPU UUID | `GPU-677cef2a-aeb8-267d-4a59-19e72f682efc` |
| 官方源码 | [mlvlab/VideoSearch-R1](https://github.com/mlvlab/VideoSearch-R1) |
| 固定源码 commit | `69234fd9783b7235164d9f92b8903d9781dc284e` |
| 源码状态 | `upstream/` 工作树干净；本地兼容与验收工具位于根目录 `scripts/` |
| 本次状态 | `complete`；推理返回码与 launcher 退出码均为 `0` |

实际启动命令如下。该命令会重新启动一次全量评测，记录于此供追溯，不表示本报告生成时又运行了模型。

```bash
cd /data/wyt/videosearch_r1_repro
source scripts/env.sh
python -B scripts/run_charades_full.py --execute --gpu 1
```

本地 runner 调用固定提交的官方入口：

```bash
bash upstream/scripts/inference/inference.bash charades \
  --checkpoint /data/wyt/videosearch_r1_repro/checkpoints/VideoSearchR1__charades-stage2
```

启动前已验证绑定当前输入的 CPU 预检和最终 10 题 smoke。此次全量使用完整本地测试资产，没有通过 smoke 的按需下载包装器读取视频。

## 2. 模型、数据与运行配置

### 2.1 实际加载的模型

| 组件 | 本地位置 | 固定 revision / 说明 |
|---|---|---|
| Charades Stage2 VLM | `checkpoints/VideoSearchR1__charades-stage2/` | `VideoSearchR1/charades-stage2`，revision `c78dc49c6b2cffdf9ae36bd01b738f9b9cdce283` |
| 检索编码器 | `checkpoints/Qwen__Qwen3-VL-Embedding-2B/` | `Qwen/Qwen3-VL-Embedding-2B`，revision `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` |
| vLLM 兼容派生权重 | Stage2 目录下 `_vllm_compat/` | 用于文本生成；原始 Stage2 中的 SQR 参数由 PyTorch 模型加载 |

实际 VLM 的配置和张量形状匹配 **Qwen3-VL-2B** 结构，检索编码器也是 2B 系列。运行日志确认加载了 **16 个 refine/SQR tensors**。发布模型卡的 `base_model` 字段曾标为 4B，与实际结构不一致；本次保留发布权重，不通过修改配置改变模型规模。结构核验不能单独证明完整训练来源，相关证据保存在[模型身份核验](../reports/04_MODEL_IDENTITY_CHECK.json)。

### 2.2 完整测试输入

| 输入 | 数量 / 形状 |
|---|---|
| 原始测试问题 | 3720 条，保留原始 index 0–3719 |
| query 向量 | `(3720, 2048)` |
| 完整视频检索库 | 1334 个视频，向量 `(1334, 2048)` |
| 测试视频像素张量 | 1334/1334 个 NPY，已全部取得并通过 CPU 验收 |
| 视频元数据 | 完整覆盖上述 1334 个视频 |
| 数据根目录 | `data/charades-sta/`；目录名为官方资产别名，查询使用 Charades-FIG 标注 |

视频 NPY 是抽帧、缩放后的 RGB 像素数组，用于 VLM 看视频；2048 维向量用于检索。两者承担不同功能。

### 2.3 本次生效参数

参数文件：[configs/charades_stage2_eval.bash](../configs/charades_stage2_eval.bash)。

| 参数 | 本次值 | 依据或边界 |
|---|---|---|
| 最大验证轮数 T | 2 | 论文明确的推理设置 |
| SQR soft tokens N | 8，应用时形状 `[8, 2048]` | 论文设置；运行后逐条验收 |
| 测试范围 | `MAX_SAMPLES=0`、ratio=1、单 shard | 全部 3720 题 |
| 检索与重排范围 | `RERANKS=all` | 使用完整视频向量库 |
| 初始查询 | 官方预计算 query 向量，`USE_UPDATED_QUERY=False` | 沿用官方入口 |
| 检索编码方式 | `latent_last`，归一化，文本最大长度 128 | 沿用已验收的官方调用链 |
| 精度 / 后端 | BF16 / vLLM | 本次实测配置 |
| batch / 并行 | batch=1、TP=1、worker=1、GPU=1 张 | 本次实测配置 |
| vLLM 运行参数 | eager=True，memory utilization=0.45 | 与最终 smoke 一致的本地执行选择 |
| 输入与输出预算 | 输入 20384，最大生成 1024，vLLM 上下文 21408 tokens | 总上下文预算，不能视为视觉 token 数 |
| 解码 | temperature=0，贪心生成 | 官方推理实现；不同于论文训练时的采样温度 |
| 随机种子 | 外层 seed=42；vLLM engine 日志 seed=0 | 官方构造器没有把外层 seed 传给 vLLM |
| 视频采样 | 官方预处理目标 1 FPS、最多 64 帧；运行时读取已保存帧，不再抽样 | 保留官方读取、偶数帧补齐及 processor 行为 |
| 推理视觉预算 | 没有新增 4096 个视觉 token 的硬上限 | 论文将 4096 明确写在训练设置中，未明确指定为推理上限 |
| 末轮查询更新 | 第二轮触发 refine 后仍执行 SQR 和重排，不增加第三次验证 | 事先约定保留官方实现 |

CPU processor 边界检查中，视频 HUWAZ 产生 6080 个视觉 tokens、完整 prompt 6659 tokens，未发生截断。这是预检中的边界样本观测，不表示全量运行每次输入都为该值。

主要环境版本：Python 3.11.17，Torch 2.8.0+cu128，Transformers 4.57.1，vLLM 0.11.0，FlashAttention 2.8.3.post1，faiss-cpu 1.13.2，decord 0.6.0。CPU 预检确认官方 42 项依赖 pins 匹配。

## 3. 固定的评测口径

**VR：**查看最终排名中标注视频的位置，统计 R@1/5/10/100。若末轮执行查询更新，使用更新后的最终排名。所有指标的分母均为完整 3720 题。

**VER 主指标：**每题取最后一次实际执行的验证，即 `turns[-1].answer` 与同一轮 `turns[-1].top1_is_gt` 配对。第一轮就结束的题取第一轮，执行两轮的题取第二轮。判断正确包括“正确接受相关视频”和“正确拒绝无关视频”。分母为可解析的 matched/not_matched 回答数，同时报告无效数与覆盖率；本次无效数为 0，分母实际为 3720。

**VCMR：**沿用官方 JSONL 中的 `final_iou_r1`，并用官方离线 report helper 按 IoU≥0.3/0.5/0.7 计分。它要求实际验证、定位的视频正确，并产生有效时间区间；错误视频、拒绝或无有效定位不能获得命中。分母为完整 3720 题。

例如末次检查视频 B，拒绝后又重排得到视频 C：VER 仍评价对 B 的判断，VR 使用重排后的排名，不把 B 的时间戳套到未经验证的 C 上。该协议在全量前已经确定，不根据本次结果切换主指标。

## 4. 结果与论文 Table 2 对照

论文参照：[VideoSearch-R1，Table 2 与第 4 节](https://arxiv.org/html/2607.00446v1#S4)。下表百分比保留两位小数；差值单位为**百分点**，使用未四舍五入的本次结果与论文公布值计算。

| 指标 | 本次命中 / 分母 | 本次结果 | 论文 Charades-FIG / VideoSearch-R1 | 差值 |
|---|---|---:|---:|---:|
| VCMR IoU≥0.3 R@1 | 610 / 3720 | 16.40% | 16.5% | -0.10 |
| VCMR IoU≥0.5 R@1 | 464 / 3720 | 12.47% | 13.4% | -0.93 |
| VCMR IoU≥0.7 R@1 | 257 / 3720 | 6.91% | 8.2% | -1.29 |
| VR R@1 | 922 / 3720 | 24.78% | 24.6% | +0.18 |
| VR R@5 | 1598 / 3720 | 42.96% | 42.9% | +0.06 |
| VR R@10 | 1944 / 3720 | 52.26% | 52.2% | +0.06 |
| VR R@100 | 3147 / 3720 | 84.60% | 84.7% | -0.10 |

补充统计：VR MRR 为 0.338290，平均 rank 为 58.519892；VCMR mIoU@R1 为 0.110797（11.08%）。

### 4.1 VER 主结果与补充口径

| VER 口径 | 正确 / 有效回答 | 准确率 | 用途 |
|---|---|---:|---|
| **末次实际验证，与实际检查视频配对** | **1944 / 3720** | **52.26%** | **本次预先确定的主指标** |
| 首轮验证 | 2784 / 3720 | 74.84% | 现有预测的 CPU 补充统计，与官方 report 首轮统计一致 |
| 官方 report 的 final 字段配对 | 1945 / 3720 | 52.28% | 兼容性诊断；不能替代同轮配对的主指标 |

三种口径本次均无无效回答。主 VER 的混淆矩阵为 TP=802、TN=1142、FP=1655、FN=121，其中正类表示视频与查询匹配。

论文报告 VER=75.7%，正文仅说明判断 match/not match 的准确率，没有明确多轮取轮方式。首轮统计更接近论文值，但仍相差约 0.86 个百分点，**不能据此证明论文使用首轮，也不能将主 VER 改为首轮后宣称对齐**。

### 4.2 官方两个计分入口的浮点边界差异

本次报告主表采用保存 JSONL 后、由官方 `write_temporal_result.py` helper 计算的结果，与本地主协议 JSON 一致。官方推理进程自己的 summary JSON 先将 IoU 转为 float32，在 IoU=0.7 附近产生不同判定：

| 入口 | IoU≥0.7 命中数 | 结果 |
|---|---:|---:|
| JSONL → 官方离线 report / 本地主协议 | 257 / 3720 | 6.908602% |
| 官方推理 summary，NumPy float32 | 259 / 3720 | 6.962366% |

差异仅涉及原始 index **166、1873**。两条 JSONL 的 `final_iou_r1` 都为 `0.6999999999999998`；Python float 与 0.7 比较不命中，转 float32 后的 NumPy 比较命中。IoU=0.3、0.5 的命中数一致。

本报告保留两份原始结果并注明差异，没有添加容差、改写预测或修改官方计分代码。该差异为 2/3720，即约 **0.054 个百分点**，不足以解释与论文 IoU=0.7 指标约 1.29 个百分点的全部差距。

## 5. 完整性、轨迹与资源验收

| 验收项 | 结果 |
|---|---|
| 原始 index | 0–3719，每个恰好一次，顺序完整 |
| 有效预测 | 3720 / 3720 |
| 跳题 | query 元数据缺失、正例视频缺失、视频文件缺失、GT 时间非法均为 0 |
| 视频元数据调用 | 6085 / 6085 命中，缺失 0 |
| 最终回答 | matched=2457、not_matched=1263、unknown=0 |
| 一轮结束 / 两轮执行 | 1355 / 2365 题 |
| 使用 SQR 的题数 | 2365 / 3720，63.58% |
| SQR 应用次数 | 3628；应用时 N=8、shape=`[8, 2048]` 验收通过 |
| 第二轮验证后仍执行 SQR | 1263 题；其中 22 题改变 top1 视频，3 题改变 top1 是否为 GT 的标签 |
| 预测及计分文件 SHA256 | 与运行 manifest 一致，报告生成时复核通过 |
| 官方源码 | 固定 commit，工作树干净 |
| 退出状态 | 推理及 launcher 均为 0；`resource_release_passed=true` |
| GPU 释放 | 退出后连续两次空闲采样，无计算进程、显存 0 MiB、利用率 0% |

| 耗时 / 显存项 | 实测 |
|---|---|
| 推理进程总耗时 | 16432.10 秒，约 4 小时 33 分 52 秒 |
| 初始化 | 103.56 秒 |
| 官方逐题循环 | 16325.45 秒，约 4 小时 32 分 5 秒 |
| 第一题 | 16.65 秒 |
| 后续题平均 | 4.385 秒 / 题 |
| 进程退出阶段 | 3.09 秒 |
| 设备显存采样峰值 | 49627 MiB，约 48.46 GiB，包含 vLLM 缓存预留，不代表最低显存需求 |

此前 10 题 smoke 的稳态均值投影约 4.35 小时，全题循环均值投影约 5.59 小时，工程预留为 6–7 小时。本次实际约 4.56 小时。上述估计来自小样本，不是统计置信区间。

## 6. 已知差异与“离线定位误差分析”的含义

**离线**表示复用已有预测、每轮轨迹、标注和日志，在 CPU 上统计；不重新加载 VLM 或生成新答案。**定位误差**特指预测开始、结束时间与标注区间的差异。

这项工作可以帮助解释本次结果与论文的差距，但需要区分两个层次：

1. **本次模型错在哪里：**逐题区分检索不到正确视频、对视频是否匹配判断错误、时间标签解析失败、视频正确但起止时间偏差过大。例如正确区间为 10–20 秒，预测为 5–25 秒时 IoU=0.5，能命中 0.5 阈值但不能命中 0.7 阈值。
2. **为什么和作者结果不同：**还需要将作者公开预测与本次预测按同一原始 index、相同指标协议逐题比较，再检查视频输入、视觉预算、时间映射、生成后端及数值精度。只有汇总分数，不能确认差距的具体原因。

本次全量报告首次编写时已完成基础完整性统计、首轮/末次 VER 对照和 IoU 浮点边界核对。随后完成的[最小离线检查](CHARADES_STAGE2_OFFLINE_CHECK_20261027.md)补充了作者公开预测逐题对照、起止时间偏差分布和错误阶段分解；后端或视觉输入导致差异的具体原因仍未通过受控实验确认。

CPU 预检还保留了两项数据观察，作为后续分组分析线索：

- 13 组重复 query 文本；官方文本查找存在覆盖行为，其中 1 组跨视频。本次保留全部原始题，以原始 index 验收，GT 视频与时间取各题原始标注。
- 539 条标注结束时间超出元数据视频时长 0.1 秒以上，最大约 1.47 秒。本次保留官方标注和原有预测时间裁剪，不删除或修正测试题。

这些观察目前没有经过对照实验，不能直接宣称它们造成了与论文的分数差异。模型卡来源字段冲突、推理视觉预算和 VER 多轮定义也继续作为已披露的复现边界保留。

## 7. 后续实验顺序

建议先固定本次 Charades 基线，完成上述最小离线核查，同时开始下一数据集的 CPU 准备；不必等到 Charades 每个小数位与论文完全一致后才推进。

后续按 **DiDeMo-FIG → ActivityNet-FIG** 补充 Table 2 的 VideoSearch-R1 主结果。固定提交的[官方资源说明](https://github.com/mlvlab/VideoSearch-R1/blob/69234fd9783b7235164d9f92b8903d9781dc284e/README.md)列出了三个数据集的预处理 bucket，以及各自 Stage1、Stage2 checkpoints，因此资源范围并非只有 Charades。

| 下一数据集 | 官方预处理数据入口 | 对应发布权重 |
|---|---|---|
| DiDeMo-FIG | `hf://buckets/VideoSearchR1/data/datasets/didemo` | `VideoSearchR1/didemo-stage2` |
| ActivityNet-FIG | `hf://buckets/VideoSearchR1/data/datasets/activitynet` | `VideoSearchR1/activitynet-stage2` |

每个数据集都需要重新完成：固定 checkpoint revision 与模型结构核验 → 取得完整评测 split 的标注、向量、元数据和视频张量 → 核对题数、检索库、时间单位与采样映射 → 将当前 Charades 专用预检及计分脚本中的路径、3720/1334 等断言参数化 → CPU 预检 → 同入口小样本 smoke → 独立目录全量与计分。

当前本地仅准备了 Charades 模型与测试数据，尚未为另外两个数据集启动下载或推理。后续可直接使用各自发布的 Stage2 权重做推理评测；训练复现另立阶段。若要复现整张 Table 2，还需补充 ZS、FT 等其他方法；Stage1 checkpoint 不能直接当作论文的普通 FT baseline。

## 8. 原始证据与文件索引

以下链接均相对本报告位置，可直接打开：

- [运行 manifest：配置签名、指标、耗时与释放记录](../outputs/charades_full_20261027_161353_178368/run_manifest.json)
- [完整 3720 条逐轮预测 JSONL](../outputs/charades_full_20261027_161353_178368/external_verified_test_temporal_grounding_VideoSearchR1__charades-stage2.jsonl)
- [本地主协议计分 JSON，主表取值来源](../outputs/charades_full_20261027_161353_178368/external_verified_test_temporal_grounding_VideoSearchR1__charades-stage2.primary.result.json)
- [官方推理 summary JSON，含 float32 汇总](../outputs/charades_full_20261027_161353_178368/external_verified_test_temporal_grounding_VideoSearchR1__charades-stage2.json)
- [官方离线 report JSON](../outputs/charades_full_20261027_161353_178368/external_verified_test_temporal_grounding_VideoSearchR1__charades-stage2.result.json)
- [推理日志](../outputs/charades_full_20261027_161353_178368/inference.log)、[官方计分日志](../outputs/charades_full_20261027_161353_178368/report.log)、[主协议计分日志](../outputs/charades_full_20261027_161353_178368/primary_score.log)
- [逐题耗时记录](../outputs/charades_full_20261027_161353_178368/timing.json)、[GPU 显存曲线 CSV](../outputs/charades_full_20261027_161353_178368/gpu_memory.csv)
- [tmux 启动回执](../logs/charades_full_tmux_20261027_161336.json)、[launcher 日志](../logs/charades_full_tmux_20261027_161336.log)、[退出码](../logs/charades_full_tmux_20261027_161336.exit_code)
- [CPU 预检](../reports/08_cpu_preflight.json)、[最终 smoke](../reports/09_final_smoke.json)、[全量视频准备验收](../reports/07_charades_test_videos.json)
- [本地全量 runner](../scripts/run_charades_full.py)、[主协议 scorer](../scripts/score_charades_protocol.py)、[官方离线指标 helper](../upstream/scripts/eval/write_temporal_result.py)

本次关键 SHA256：

```text
predictions.jsonl:
35cf85cc43a42d96c2bcc8fe47d25416014db7669b09c5b3bd4e06b4faa3f425

primary.result.json:
3cbf3e52be1d93ea14e461a0a9a26576337e33b7d338c946d8b33141ce9e8e30

experiment_signature:
3ebf3ab3b450ee1bb5b4b32819566cf8dc06eceacedced8124ecd0bee82c8102
```

报告编写仅执行 CPU 读取、统计和文档生成，没有修改本次预测、实验配置、官方源码或历史计分结果。
