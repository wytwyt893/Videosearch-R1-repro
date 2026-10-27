# VideoSearch-R1 复现记录

本仓库用于复现 [VideoSearch-R1](https://github.com/mlvlab/VideoSearch-R1) 发布权重在 Charades-FIG、DiDeMo-FIG、ActivityNet-FIG 上的推理评测，主要对应[论文 Table 2](https://arxiv.org/html/2607.00446v1#S4) 的 VideoSearch-R1 行。

当前状态（2026-10-28）：**Charades Stage2 全量 3720 题完成，最小离线核查通过；DiDeMo-FIG CPU 预检和同入口 10 题 smoke 已通过，全量尚未启动**。DiDeMo 在单卡 A100 80GB、batch=1 下外推约 4.1–4.6 小时，建议先预留 6 小时；其官方 NPY 时间轴与源时间戳的差异已记录，暂保留官方行为。后续扩展 ActivityNet-FIG。当前结果来自作者发布 checkpoint；尚未执行完整 SFT/GRPO 训练复现，也未复现整张表中的所有 baseline。

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

文档内指向 `outputs/`、`cache/`、`logs/` 的链接需要本机保留的资产，在 GitHub 网页上不可用；轻量结果已另存于上述 `reports/10_*`、`reports/11_*`。克隆本仓库不会自动恢复运行环境、模型或测试资产，历史 CPU 指纹报告也不能直接作为新机器的验收结果。

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

## DiDeMo 资产准备

固定 revision `a576c16040c35a5ec71468c66ec7628e7db71997` 的 2B Stage2 已通过官方大小/哈希核验，共 14 个运行文件、16 个 refine tensors。完整测试资产于北京时间 2026-10-28 01:54 验收完成：4013 个问题、1002 个视频像素张量（24976278588 字节，约 23.26 GiB），向量形状为 `(4013, 2048)` / `(1002, 2048)`。全部视频数组、元数据与本地 SHA256 收据核对通过，无缺失、无 `.partial`，任务退出码为 0。

此次只准备测试资产，复用已有 Qwen 检索器，没有执行模型或下载训练视频。218 个逻辑 tar 按官方 manifest 选择完整资产或拆分 parts，实际索引前 25 个逻辑分片即覆盖测试视频和尾部 metadata。首次任务因 TLS 超时剩余 15 个文件；后续保留已完成的 987 个文件，仅补齐缺失文件，再验证完整测试库。下载器已加入网络失败批次的有限重试；本地完整性错误仍直接失败。

```bash
# 当前本机查看保留的下载日志；Ctrl+B 后按 D 离开。
tmux attach -t videosearch_didemo_download
```

[下载启动快照](reports/12_didemo_download_plan.json)保留当时的来源和范围；资产结论见[模型与测试资产验收](reports/13_didemo_assets.json)。原始状态与收据位于本地 `cache/didemo_download/`，不纳入 Git。脚本为 `scripts/prepare_didemo_assets.py`，依赖事先获取到 `cache/manifests/didemo/` 的官方模型、config、bucket manifest 和物理分片元数据。

随后已完成 DiDeMo 专用配置与计分适配、完整 CPU 预检及同入口 smoke：10/10 有效预测、零跳题、2 题使用 SQR，推理子进程总耗时 74.43 秒，GPU 已释放。入口为 `scripts/run_didemo_full.py`，默认只做 CPU dry run，`--execute --smoke --gpu 1` 才执行前 10 题；全量需显式执行并通过当前输入的 smoke 绑定检查。

- [DiDeMo smoke 与估时报告](doc/DIDEMO_STAGE2_SMOKE_20261028.md)
- [CPU 预检](reports/15_didemo_cpu_preflight.json)、[完整 smoke 记录](reports/16_didemo_final_smoke.json)、[精简结果与估时](reports/17_didemo_smoke_summary.json)

官方 NPY 入口把预抽帧数据按 target fps=1、连续帧号传给 processor；最长视频源时间约 180 秒，而 reader 的最后名义时间为 63 秒。本轮保留官方实现，明确区分基线估时、全量指标复现与未来真实时间戳透传对照，详见 smoke 报告。

## 历史记录

`reports/00_WORKSPACE_README_HISTORY.md` 保存首次发布前的工作区 README，其中包含当时尚未完成安装、下载或全量评测的历史状态。`01_*` 至 `09_*` 继续保留各阶段证据；当前实验结论优先查看 `doc/` 与最新摘要。

本仓库不修改其他研究项目。上游源码及其许可证以官方子模块为准。
