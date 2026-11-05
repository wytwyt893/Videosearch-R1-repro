# videosearch_r1 × DIDEMO全量评估-10.28

整理说明（2026-11-05）：文件日期沿用原实验/报告日期，正文保留当时阶段的记录；当前三数据集结果见[全量评估汇总](<videosearch_r1 × 三数据集全量评估汇总-11.05.md>)。

**4013/4013 题完成，退出码为 0，完整性与计分验收通过，GPU1 已释放。VR 四项与论文最大相差约 0.06 个百分点；VCMR 三项低约 0.64–1.53 个百分点。** 主 VER 为末次实际验证的 71.19%；首轮 VER 74.93% 仅作为补充，两者不混用。

本次对应论文 Table 2 中 **DiDeMo-FIG / VideoSearch-R1** 的发布权重推理评测，未执行 SFT/GRPO 训练，未评测表中其他方法。论文 VER 的多轮取轮口径仍未确认，因此不能将数值差直接解释为同口径性能退化。

后续 CPU 核查已完成，见[最小离线对照报告](<videosearch_r1 × DIDEMO离线检查-10.28.md>)：作者公开预测实际为 T=3，首轮 VER=74.63%，末次=70.87%；其前两轮快照的 VR/VCMR 加上首轮 VER 按一位小数可匹配论文全部 8 项。该快照不是新的 T=2 推理实验；本报告的原始结果与末次主 VER 保持不变。

## 运行与固定设置

时间均为北京时间（Asia/Shanghai）。

| 项目 | 本次记录 |
|---|---|
| tmux 会话 | `videosearch_didemo_full` |
| launcher 启动 / 退出 | 2026-10-28 10:14:09 / 14:50:55 |
| 结果目录 | `outputs/didemo_full_20261028_101412_373223/` |
| GPU | 物理 GPU1，NVIDIA A100-SXM4-80GB，单卡 |
| GPU UUID | `GPU-677cef2a-aeb8-267d-4a59-19e72f682efc`；CUDA/NVML 一致 |
| 官方源码 commit | `69234fd9783b7235164d9f92b8903d9781dc284e`；结束后核对工作树干净 |
| Stage2 VLM | `checkpoints/VideoSearchR1__didemo-stage2/`；实际配置/张量为 2B 架构 |
| Stage2 revision | `a576c16040c35a5ec71468c66ec7628e7db71997` |
| 检索器 | `checkpoints/Qwen__Qwen3-VL-Embedding-2B/` |
| 检索器 revision | `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` |
| 测试输入 | 原始 index 0–4012；4013 个 query 向量、1002 个视频向量与全部视频像素 NPY |
| 固定参数 | T=2、N=8、BF16/vLLM、batch=1、TP=1、eager=True、memory utilization=0.45 |
| 输入 / 生成预算 | 20384 / 1024 tokens，vLLM 上下文 21408 |
| 视觉输入 | 保留官方预处理/reader；没有新增 4096 个推理视觉 token 的硬上限 |
| 解码与 seed | temperature=0；外层 seed=42，官方 vLLM 构造器未传入该 seed，engine 默认 seed=0 |

启动前 CPU 预检、最终 smoke 与全量输入指纹绑定检查通过。完整测试资产均已本地准备，本次没有在推理期间按需下载视频。发布模型卡的基座元数据与实际 2B 架构之间的历史矛盾仍保留，未修改权重或配置来更换规模。

实际命令如下，仅用于追溯；本报告生成时没有重新运行模型：

```bash
cd /data/wyt/videosearch_r1_repro
source scripts/env.sh
python -B scripts/run_didemo_full.py --execute --gpu 1
```

## Table 2 对照

论文来源：[VideoSearch-R1，Table 2](https://arxiv.org/html/2607.00446v1#S4.T2)。仅对照 DiDeMo-FIG 的 VideoSearch-R1 行。百分比保留两位小数；差值用未四舍五入的本次结果减论文公布值，单位为**百分点**。

| 指标 | 命中/正确数与分母 | 本次 | 论文 | 差值 |
|---|---|---:|---:|---:|
| VCMR IoU≥0.3 R@1 | 1275 / 4013 | 31.77% | 33.3% | -1.53 |
| VCMR IoU≥0.5 R@1 | 1156 / 4013 | 28.81% | 30.2% | -1.39 |
| VCMR IoU≥0.7 R@1 | 765 / 4013 | 19.06% | 19.7% | -0.64 |
| VER：末次实际验证（主指标） | 2857 / 4013 | 71.19% | 74.6% | -3.41 |
| VR R@1 | 2366 / 4013 | 58.96% | 59.0% | -0.04 |
| VR R@5 | 3293 / 4013 | 82.06% | 82.0% | +0.06 |
| VR R@10 | 3524 / 4013 | 87.81% | 87.8% | +0.01 |
| VR R@100 | 3913 / 4013 | 97.51% | 97.5% | +0.01 |

**VER 行的差值仅作数值参照，尚不能认定为严格同口径差距。** 论文定义了 match/not match 准确率，但未明确多轮结果的取轮规则。本次协议在运行前已确定，未根据结果切换主指标。

补充：VR MRR=0.689415，平均 rank=11.698978；VCMR mIoU@R1=0.255841。原始 query 排名的 R@1 为 55.15%，最终为 58.96%；这是同一次流程中原始/最终排名的诊断，不作为另一个模型的 FT/ZS 复现结果。

### VER 主指标与补充

| 口径 | 正确 / 有效回答 | 准确率 | 地位 |
|---|---|---:|---|
| 末次实际验证，与实际检查视频配对 | 2857 / 4013 | 71.19% | 预先确定的主指标 |
| 首轮验证 | 3007 / 4013 | 74.93% | CPU 离线补充，不替换主指标 |
| 官方 final 字段配对 | 2856 / 4013 | 71.17% | 兼容性诊断 |

三种统计本次均无无效回答，覆盖率 100%。主 VER 混淆矩阵 TP=2128、TN=729、FP=933、FN=223；正类表示实际检查的视频与查询匹配。首轮比论文 74.6% 高约 0.33 个百分点，接近论文数值，但不能据此证明论文采用首轮。

**固定配对原则：**若最后验证视频 B，之后触发 SQR 重排得到 C，VER 仍评价对 B 的判断；VR 使用更新后的最终排名；VCMR 使用最后实际验证/定位的视频与其时间区间，不将 B 的时间戳套到未经验证的 C 上。第二轮触发时保留官方查询更新和重排，不增加第三次验证。

## 完整性、轨迹与计分复核

| 验收项 | 结果 |
|---|---|
| 原始编号 / 标注 | index 0–4012 各一次，顺序完整；query、GT 视频与时间均匹配原始标注 |
| 跳题 | query 元数据缺失、正例视频缺失、视频文件缺失、GT 时间非法均为 0 |
| 视频元数据 | 5260/5260 次命中，缺失 0 |
| 一轮结束 / 执行两轮 | 2766 / 1247 题 |
| 最终回答 | matched=3061，not_matched=952，unknown=0 |
| SQR | 1247 题使用，应用 2199 次；逐次验证 N=8、shape=`[8, 2048]` |
| 末轮重排 | 第二轮后 952 次；2 题改变 top1 视频，1 题改变 top1 是否为 GT |
| 计分一致性 | 补充 scorer 的 primary 与原主协议 JSON、run manifest 完全一致 |
| IoU 阈值边界 | Python 离线与 NumPy float32 在 0.3/0.5/0.7 的命中数均为 1275/1156/765，无判定差异 |
| 输入与源码绑定 | 结束后 CPU 预检指纹、smoke 绑定及官方源码状态复核通过 |
| 退出 / 资源 | 推理、launcher 均为 0；GPU1 显存 0 MiB、利用率 0%，无计算进程 |

官方推理 summary 和离线 helper 的 mIoU 仅有约 2.8e-8 的浮点求和差异，不影响上述阈值命中数。原始输出均保留。日志仍含官方退出时 NCCL 未显式 destroy 的提示；连续 GPU 采样及本轮复查确认资源已释放。tmux 保留已退出的 runner 窗格与 `tail` 查看窗格，后者不是推理进程。

预先发现的标注边界仍按原协议保留：14 条 query 的首尾空格按官方 `.strip()` 处理；1 组同文 query 的官方覆盖映射涉及原始 index 1067；141 条 GT 结束时间超过 annotation duration 0.1 秒以上。没有删除、替换或改写测试题。

## 实测耗时与 smoke 估时核对

| 阶段 | 实测 |
|---|---:|
| 初始化与准备 | 35.73 秒 |
| 官方逐题循环 | 16561.89 秒，约 4 小时 36 分 2 秒 |
| 退出阶段 | 2.65 秒 |
| 推理子进程合计 | 16600.27 秒，约 4 小时 36 分 40 秒 |
| 后续 4012 题平均 | 4.13 秒/题 |
| 设备显存采样峰值 | 50147 MiB，约 48.97 GiB |

显存采样包含 vLLM 缓存预留，不代表最低显存要求。计分、外层 GPU 检查与资源释放验收另计；launcher 墙钟约 4 小时 36 分 46 秒。先前 10 题 smoke 外推约 4.14–4.56 小时，实测约 4.61 小时，与较保守外推接近，并低于 6 小时预留；全量二轮比例为 31.07%，高于 smoke 的 20%。

## 结果解释与仍待确认的事项

1. **VR 接近论文。** 四项与公布值的最大绝对差约 0.06 个百分点，当前检索结果支持这一结论，但不等于所有实现细节已证明一致。
2. **VCMR 仍有差距。** 三项均低于论文，IoU≥0.3/0.5/0.7 分别约低 1.53/1.39/0.64 个百分点。全量结束后的[逐题离线对照](<videosearch_r1 × DIDEMO离线检查-10.28.md>)显示，差异主要落在正确接受视频后的时间定位，长视频组较集中，但尚不能确认具体原因。
3. **VER 口径未完全对齐。** 保留末次 71.19% 为主指标、首轮 74.93% 为补充；不因首轮更接近论文就更改主协议。
4. **源时间轴差异继续保留。** CPU 检查确认官方 NPY metadata 入口未透传真实抽帧时间戳。最长样本源跨度约 180 秒，64 帧经 reader 的最后名义时间为 63 秒。本轮保留官方行为。它是可做单变量对照的因素，尚未证明是本次 VCMR 差距的原因，也未证明论文运行时使用了其他实现。

当前完成了 Charades-FIG 与 DiDeMo-FIG 的 Stage2 发布权重全量评测；ActivityNet-FIG、训练复现和表中其他 baseline 尚未完成。

## 文件与证据

- [可在 GitHub 发布的轻量完整结果摘要](../reports/18_didemo_full_summary.json)
- [CPU 预检](../reports/15_didemo_cpu_preflight.json)、[最终 smoke](../reports/16_didemo_final_smoke.json)、[smoke 估时摘要](../reports/17_didemo_smoke_summary.json)
- [预测 JSONL](../outputs/didemo_full_20261028_101412_373223/external_verified_test_temporal_grounding_VideoSearchR1__didemo-stage2.jsonl)
- [主协议计分](../outputs/didemo_full_20261028_101412_373223/external_verified_test_temporal_grounding_VideoSearchR1__didemo-stage2.primary.result.json)
- [首轮 VER 补充计分](../outputs/didemo_full_20261028_101412_373223/external_verified_test_temporal_grounding_VideoSearchR1__didemo-stage2.supplementary.result.json)
- [官方离线 report 结果](../outputs/didemo_full_20261028_101412_373223/external_verified_test_temporal_grounding_VideoSearchR1__didemo-stage2.result.json)
- [官方推理 summary](../outputs/didemo_full_20261028_101412_373223/external_verified_test_temporal_grounding_VideoSearchR1__didemo-stage2.json)
- [运行记录](../outputs/didemo_full_20261028_101412_373223/run_manifest.json)
- [推理日志](../outputs/didemo_full_20261028_101412_373223/inference.log)
- [逐题计时](../outputs/didemo_full_20261028_101412_373223/timing.json)
- [显存采样 CSV](../outputs/didemo_full_20261028_101412_373223/gpu_memory.csv)
- [完成后完整性复核](../outputs/didemo_full_20261028_101412_373223/postrun_integrity.json)
- [完成后 GPU/tmux 检查](../outputs/didemo_full_20261028_101412_373223/postrun_resources.json)

指向 `outputs/`、`logs/` 的文件仅在本机保留，不纳入 Git。预测 SHA256：`325224da4792728f095f9fdb5960a93971498ede827e994a9eab8dc3b64e065c`；主协议结果 SHA256：`e9b3fe51108f8768ccbb9247646064af9ac8cbdbaa21f30d3f64da38697d5ea0`。本轮补充统计仅使用 CPU 读取现有预测，没有改写原预测、主计分或重新运行模型。
