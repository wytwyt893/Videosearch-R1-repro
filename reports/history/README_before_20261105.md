# VideoSearch-R1 复现记录

本仓库用于复现 [VideoSearch-R1](https://github.com/mlvlab/VideoSearch-R1) 发布权重在 Charades-FIG、DiDeMo-FIG、ActivityNet-FIG 上的推理评测，主要对应[论文 Table 2](https://arxiv.org/html/2607.00446v1#S4) 的 VideoSearch-R1 行。

当前状态（2026-10-29）：**Charades Stage2 全量 3720 题、DiDeMo Stage2 全量 4013 题完成，最小离线对照通过；ActivityNet 完整测试资产、CPU 预检和前 10 题 smoke 已通过，全量 16966 题尚未启动。** ActivityNet smoke 在 GPU1 耗时 103.20 秒，当前配置全量外推约 28.5–29.8 小时。官方资产基线的工程链路已跑通，但长视频时间轴和 364 个 MKV 的帧覆盖仍存在论文对齐疑问，详见下方报告。当前结果来自作者发布 checkpoint；尚未执行完整 SFT/GRPO 训练复现，也未复现表中其他 baseline。

## Charades 当前结果

主设置为 2B Stage2 VLM、2B 检索编码器、T=2、N=8、单卡 A100 80GB、BF16/vLLM、batch=1；输入检索库包含全部 1334 个视频。主 VER 使用最后一次实际验证，并与同一轮实际检查的视频配对。

| 指标 | 本地全量 | 作者公开预测，同口径重算 |
|---|---:|---:|
| VR R@1 | 24.78% | 24.57% |
| VR R@5 / R@10 / R@100 | 42.96 / 52.26 / 84.60% | 42.93 / 52.20 / 84.76% |
| VCMR IoU≥0.3 / 0.5 / 0.7 R@1 | 16.40 / 12.47 / 6.91% | 16.48 / 13.44 / 8.17% |
| VER：末次实际验证（主指标） | 52.26% | 52.77% |
| VER：首轮（补充） | 74.84% | 75.65% |

论文 VER 的多轮取轮口径尚未确认，不能将末次 VER 与论文 75.7% 直接比较。较严格 VCMR 阈值仍有差距；逐题对照显示主要落在正确接受视频后的时间区间预测，具体后端或视觉输入原因尚未通过受控对照确认。

- [完整实验报告](doc/CHARADES_STAGE2_FULL_20261027.md)
- [最小离线检查报告](doc/CHARADES_STAGE2_OFFLINE_CHECK_20261027.md)
- [可直接在 GitHub 查看或下载的全量结果摘要](reports/10_charades_full_summary.json)
- [可直接在 GitHub 查看或下载的离线分析摘要](reports/11_charades_offline_summary.json)

## 代码来源与克隆

自己的复现代码和实验记录提交到 [wytwyt893/Videosearch-R1-repro](https://github.com/wytwyt893/Videosearch-R1-repro)。官方源码以 `upstream/` Git 子模块保留，固定提交为 `69234fd9783b7235164d9f92b8903d9781dc284e`，保持工作树干净。

```bash
git clone --recurse-submodules https://github.com/wytwyt893/Videosearch-R1-repro.git
cd Videosearch-R1-repro
git -C upstream rev-parse HEAD
```

已有普通 clone 可执行 `git submodule update --init upstream`。子模块只记录固定官方源码，不包含模型或数据。

工作区 `.vscode/settings.json` 仅在本项目关闭嵌套仓库/子模块自动检测，让 VS Code 源代码管理显示根复现仓库；不会删除官方 `.git`。如果之前已经打开过子模块仓库，可在命令面板执行 `Developer: Reload Window` 使已有视图重新加载。

## 工作目录

| 目录 | 内容 | 提交到 Git |
|---|---|---|
| `upstream/` | 固定提交的官方源码 | 子模块指针与来源 URL |
| `scripts/`、`tools/` | 本地兼容、下载、预检、运行和分析入口 | 是 |
| `configs/` | 冻结的实验设置 | 是 |
| `tests/` | 计分协议、运行计时与退出检查夹具 | 是 |
| `doc/` | 完整实验与离线检查报告 | 是 |
| `reports/` | 阶段审计、版本、哈希、资产验收及轻量结果摘要 | 是；其他项目的隔离审计快照除外 |
| `.vscode/settings.json` | 本项目 Git 视图设置 | 是 |
| `data/` | 标注、检索向量、FAISS 与视频像素张量 | 否 |
| `checkpoints/` | VLM、检索器与 vLLM 兼容派生权重 | 否 |
| `envs/`、`cache/` | 两套本地环境、下载缓存与资产 manifests | 否 |
| `outputs/`、`logs/`、`tmp/` | 原始预测、运行日志、显存采样和临时文件 | 否 |

文档内指向 `outputs/`、`cache/`、`logs/` 的链接需要本机保留的资产，在 GitHub 网页上不可用；轻量结果另存于 `reports/10_*`、`reports/11_*`、`reports/18_*`、`reports/19_*`。克隆本仓库不会自动恢复运行环境、模型或测试资产，历史 CPU 指纹报告也不能直接作为新机器的验收结果。

## 当前本机运行入口

以下命令要求先准备项目环境、固定 revision 模型、完整测试数据，并通过当前输入的 CPU 预检和最终 smoke。默认 dry run 不执行模型。

```bash
source scripts/env.sh

# 检查官方真实入口的传参，不运行模型。
python -B scripts/run_charades_full.py --dry-run

# 完整 CPU 环境、模型/数据哈希和 processor 验收。
python -B scripts/preflight_charades_full.py

# 同入口最终 smoke，以及独立目录全量评测。
python -B scripts/run_charades_full.py --execute --smoke --gpu 1
python -B scripts/run_charades_full.py --execute --gpu 1
```

`scripts/env.sh` 将路径与缓存限制在当前工作区，默认隐藏 CUDA；执行 runner 时再明确选择 GPU。下载通过本项目的 `scripts/network.sh` 设置进程局部网络，不修改系统代理、DNS 或其他项目环境。

新机器准备资产时参考固定提交的[官方资源与安装说明](upstream/README.md)，再对照本仓库的环境、模型和数据验收工具；当前 `prepare_models.py` 等工具依赖本地已获取的官方 manifests，不能直接跳过资产元数据准备。

## DiDeMo 当前结果与资产准备

完整 4013 题使用与最终 smoke 相同的 T=2、N=8、BF16/vLLM、batch=1 设置，保留完整 1002 视频库。与[论文 Table 2](https://arxiv.org/html/2607.00446v1#S4.T2) 对照如下，百分比保留两位小数。

| 指标 | 本地全量 | 论文 |
|---|---:|---:|
| VR R@1 / R@5 / R@10 / R@100 | 58.96 / 82.06 / 87.81 / 97.51% | 59.0 / 82.0 / 87.8 / 97.5% |
| VCMR IoU≥0.3 / 0.5 / 0.7 R@1 | 31.77 / 28.81 / 19.06% | 33.3 / 30.2 / 19.7% |
| VER：末次实际验证（主指标） | 71.19% | 74.6%，取轮待确认 |
| VER：首轮（补充） | 74.93% | 同上，不能据此确认论文采用首轮 |

- [DiDeMo 全量实验报告](doc/DIDEMO_STAGE2_FULL_20261028.md)
- [可在 GitHub 查看或下载的全量结果摘要](reports/18_didemo_full_summary.json)
- [DiDeMo 最小离线对照报告](doc/DIDEMO_STAGE2_OFFLINE_CHECK_20261028.md)
- [可在 GitHub 查看或下载的离线对照摘要](reports/19_didemo_offline_summary.json)

推理子进程耗时 16600.27 秒，峰值显存采样 50147 MiB（约 48.97 GiB，包含 vLLM 缓存预留）。完成后使用 CPU 对现有预测重算计分，原始编号、标注、哈希和 N=8 轨迹检查通过。VCMR 差距来源尚未确认；首轮 VER 仅为补充，不替换末次主指标。

后续离线对照确认：作者公开预测是 T=3，878 题进入第三轮；本地继续保留论文明确的 T=2。作者首轮 VER=74.63%，末次=70.87%。作者已保存的前两轮快照，其 VR/VCMR 加上首轮 VER 按一位小数可匹配论文全部 8 个指标；这支持取轮解释，但不能证明论文原始运行已被完整还原。第三轮仅使 VCMR 三个阈值各多命中 1 题。

两侧 4013 题的 qid、初始 GT 排名和首轮候选均一致；11332 轮原始回答/时间解析与 IoU 重算无不一致。剩余 VCMR 差异主要落在正确接受视频后的时间定位，且较集中于源视频 >64 秒的问题组；关联尚未证明时间轴或后端的因果作用。原始主计分、配置及预测均未改变。

固定 revision `a576c16040c35a5ec71468c66ec7628e7db71997` 的 2B Stage2 已通过官方大小/哈希核验，共 14 个运行文件、16 个 refine tensors。完整测试资产于北京时间 2026-10-28 01:54 验收完成：4013 个问题、1002 个视频像素张量（24976278588 字节，约 23.26 GiB），向量形状为 `(4013, 2048)` / `(1002, 2048)`。全部视频数组、元数据与本地 SHA256 收据核对通过，无缺失、无 `.partial`，任务退出码为 0。

资产准备阶段仅取得测试数据，复用已有 Qwen 检索器，未下载训练视频。218 个逻辑 tar 按官方 manifest 选择完整资产或拆分 parts，实际索引前 25 个逻辑分片即覆盖测试视频和尾部 metadata。首次任务因 TLS 超时剩余 15 个文件；后续保留已完成的 987 个文件，仅补齐缺失文件，再验证完整测试库。下载器已加入网络失败批次的有限重试；本地完整性错误仍直接失败。

```bash
# 当前本机查看保留的下载日志；Ctrl+B 后按 D 离开。
tmux attach -t videosearch_didemo_download
```

[下载启动快照](reports/12_didemo_download_plan.json)保留当时的来源和范围；资产结论见[模型与测试资产验收](reports/13_didemo_assets.json)。原始状态与收据位于本地 `cache/didemo_download/`，不纳入 Git。脚本为 `scripts/prepare_didemo_assets.py`，依赖事先获取到 `cache/manifests/didemo/` 的官方模型、config、bucket manifest 和物理分片元数据。

随后已完成 DiDeMo 专用配置与计分适配、完整 CPU 预检及同入口 smoke：10/10 有效预测、零跳题、2 题使用 SQR，推理子进程总耗时 74.43 秒，GPU 已释放。入口为 `scripts/run_didemo_full.py`，默认只做 CPU dry run，`--execute --smoke --gpu 1` 才执行前 10 题；全量需显式执行并通过当前输入的 smoke 绑定检查。

- [DiDeMo smoke 与估时摘要](reports/17_didemo_smoke_summary.json)
- [CPU 预检](reports/15_didemo_cpu_preflight.json)、[完整 smoke 记录](reports/16_didemo_final_smoke.json)、[精简结果与估时](reports/17_didemo_smoke_summary.json)

官方 NPY 入口把预抽帧数据按 target fps=1、连续帧号传给 processor；最长视频源时间约 180 秒，而 reader 的最后名义时间为 63 秒。smoke 与本次全量均保留官方实现；该行为尚未被证明是 VCMR 差距的原因，真实时间戳透传对照尚未执行。

## ActivityNet-FIG CPU / smoke 完成，全量未启动

资产下载于北京时间 2026-10-29 01:59 左右完成，4866 个视频数组全部验收，零缺失、零 `.partial`。Stage2 revision 固定为 `918ca416dc3678d28947b5794056607b7a4656b3`，2B 架构，14 个运行文件共 4924782807 字节，官方大小/哈希和 16 个 refine tensors 核验通过；检索器复用已有 2B 权重。这一 revision 的模型文件列表没有 `eval/` 预测附件。

官方 `activitynet_fig_val_2` 映射为项目的 `raw_annotation/test.jsonl`，含 16966 个问题与 4866 个 GT 视频。发布 NPY metadata 声明目标 1 FPS、最多 64 帧，实际视频 NPY 共 151311175824 字节，约 140.92 GiB；完整 train/test bundle 约 435.61 GiB，本轮没有下载训练视频。源 metadata 与 annotation duration 分别统计，不自动修改 GT。

CPU 预检耗时 228.75 秒，覆盖全部模型/视频当前 SHA256、16966 条映射、42 项依赖、40 项传参和实际 CPU processor 边界。smoke 使用官方原始 index 0–9、完整视频库，T=2、N=8、BF16/vLLM、batch=1：10/10 有效，零跳题、零元数据缺失，4 题使用 SQR、共 7 次更新。推理子进程 103.20 秒，峰值显存采样约 47.17 GiB，GPU1 已释放。该小样本不用于宣称 Table 2 分数复现。

```bash
tmux attach -t videosearch_activitynet_smoke
```

这是保留的 smoke 日志视图，runner 已退出，`tail` 仅用于查看。下载状态为 `cache/activitynet_download/status.json`，哈希收据为同目录的 `data_receipts.jsonl`；最终资产验收已写入 `reports/22_activitynet_assets.json`。

- [下载范围、固定来源与启动快照](reports/20_activitynet_download_plan.json)
- [完整资产验收](reports/22_activitynet_assets.json)、[实际参数验收](reports/23_activitynet_alignment.json)
- [CPU 预检](reports/24_activitynet_cpu_preflight.json)、[最终 smoke](reports/25_activitynet_final_smoke.json)、[结果与估时摘要](reports/26_activitynet_smoke_summary.json)
- [CPU / smoke 实验报告](doc/ACTIVITYNET_STAGE2_SMOKE_20261029.md)、[时间轴及 MKV 元数据审计](reports/27_activitynet_time_metadata_audit.json)
- [完整测试核心数据 EDA](reports/21_activitynet_eda.json)：16966 条 annotation/qid/video 映射通过，向量形状 `(16966, 2048)` / `(4866, 2048)`，无重复 query 文本。
- 本地 [EDA 分布图](outputs/activitynet_eda_20261028/activitynet_eda.png)与[可导出 PDF](outputs/activitynet_eda_20261028/activitynet_eda.pdf)；图仅描述统计分布，不是推理结果。
- EDA 入口：`scripts/activitynet_eda.py`；元数据入口：`scripts/fetch_activitynet_metadata.py`；下载入口：`scripts/prepare_activitynet_assets.py`。

EDA 中 NPY 源 metadata 时长中位数为 103.52 秒，标注 duration 中位数为 116.50 秒，GT 事件时长中位数为 26.39 秒。364 个超过 1 秒的时长差异全部来自 MKV，其中 360 个超过 10 秒；这些 NPY 记录均为 64 个源帧、源索引 0–63，但相同路径的检索编码 metadata 记录更长的源帧序列。官方未知帧数回退分支可重现前 64 帧选择，支持帧覆盖疑问；本轮未解码原始视频，不能将短 metadata 时长直接当作原始完整时长。17 条 query 的空格按官方 `.strip()` 处理，114 条 GT 时长边界全部保留。

工程入口和输入绑定已经通过，可追溯当前官方资产基线。严格论文时间轴/帧覆盖对齐仍需核实，不能用 smoke 成功代替这项确认。当前全量外推约 28.5–29.8 小时，建议先预留 40 小时，估时随全量样本和后续输入修正变化。本轮未启动全量。完成 ActivityNet 全量可补齐 Table 2 中 VideoSearch-R1 的三数据集行；其他 baseline、训练阶段/奖励消融及训练复现随后逐项安排。

## 历史记录

`reports/00_WORKSPACE_README_HISTORY.md` 保存首次发布前的工作区 README，其中包含当时尚未完成安装、下载或全量评测的历史状态。`01_*` 至 `09_*` 继续保留各阶段证据；当前实验结论优先查看 `doc/` 与最新摘要。

本仓库不修改其他研究项目。上游源码及其许可证以官方子模块为准。
