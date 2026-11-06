# VideoSearch-R1 Baseline 复现

当前工作（2026-10-26）：先完成 **Charades Stage2 实验配置对齐**，暂不下载、排队或运行模型。逐项结论见 [实验配置对照](reports/05_EXPERIMENT_ALIGNMENT.md)，正式参数见 [Charades 评测配置](configs/charades_stage2_eval.bash)。真实官方 Shell 调用链的 40 项关键传参检查已通过。视觉预算已按用户规则决定：论文只明确训练时的 4096 设置，未明确要求推理硬上限，因此保留官方推理脚本，不新增限制。VER 协议、末轮重排定义及发布模型卡的来源标注仍有记录的不确定性；当前 checkpoint 的 2B 结构已确认。只检查配置可执行 `python -B scripts/check_charades_alignment.py`。下文为此前准备和 smoke 的历史记录。

当前范围：用户已授权从问题修复、环境和资产准备推进到 **Charades 单卡 smoke**。本轮计划和实际进度见 `reports/03_FIX_TO_SMOKE.md`；下文阶段一状态保留为历史记录。现有 `/data/setr_repro` 及其他研究者目录只读、不移动、不修改。

最新状态（2026-10-23）：**smoke 已通过，5/5 条有效预测，零跳题，3 题实际使用 SQR**。运行环境 42 项 pins、CPU 导入与 pip check 通过；Stage2 和检索器均已固定 revision 并校验哈希。已准备完整测试向量与元数据，8 个视频张量按需取得；完整评测和训练尚未启动。复跑入口为 `source scripts/env.sh` 后执行 `python -B scripts/run_charades_smoke.py --gpu 1`。结果见 [修复与 smoke 记录](reports/03_FIX_TO_SMOKE.md)，实际环境见 `reports/environment.smoke.pip.freeze.txt`。

工作区已迁至 `/data/wyt/videosearch_r1_repro/`。自有脚本优先使用相对路径，并根据脚本自身位置计算根目录，不依赖调用时的工作目录。两套 Python 环境已按原版本在新前缀重建；旧根目录软链接已移除，在旧路径不可访问时通过 CPU 复核。迁移前脚本、环境和审计快照保留在 `cache/path_relocation_20261023/`，历史日志不批量改写。此项路径修复不代表正式模型运行依赖已装完。

| 目录 / 入口 | 用途 |
|---|---|
| `upstream/` | 官方固定提交的原始代码，保持干净；本次 Git 传输故障，采用官方归档＋原 Git 对象精确校验恢复，不冒充标准 clone |
| `envs/videosearchr1/` | 独立 Conda 环境；不激活或升级 SetR 环境 |
| `envs/hf_tools/`、`tools/bin/hf` | 独立下载CLI，固定hub1.20.1；已验证 `hf buckets --help`，不升级模型环境的0.36依赖 |
| `scripts/env.sh` | 进程局部路径、缓存和低 CPU 并发；默认隐藏 CUDA |
| `scripts/install_runtime.sh` | 先安装编译前置及 Torch，再安装官方 requirements 和 editable 包；不跑模型 |
| `scripts/audit_stage1.py` | CPU 语法、计分函数夹具、资源、依赖版本、SetR 隔离检查；`--imports` 才做无 CUDA 上下文的库导入检查 |
| `scripts/restore_upstream.py` | 本次网络故障的一次性恢复取证入口；已有源码时拒绝覆盖，不是日常运行步骤 |
| `reports/01_ENGINEERING_AUDIT.md` | 人工可读的阶段一报告，含问题、修复、未验证项及下一步 |
| `reports/01_engineering_audit.json` | 实际 CPU 检查结果；`runtime_ready_cpu` 不等于 GPU 推理/训练可用 |
| `reports/environment.*` | 实际依赖与 Conda 锁定记录，不是论文原始环境的证明 |
| `logs/` | 本次获取代码、网络、Conda、pip 和 CPU 审计原始日志 |
| `data/`、`checkpoints/`、`outputs/` | 后续数据、模型与运行结果；目前不下载正式 benchmark 或权重 |

## 只读 / CPU 复核

```bash
cd /data/wyt/videosearch_r1_repro
source scripts/env.sh
envs/videosearchr1/bin/python -B scripts/audit_stage1.py
# 运行依赖实际安装完成后：
envs/videosearchr1/bin/python -B scripts/audit_stage1.py --imports
git -C upstream rev-parse HEAD
git -C upstream status --short
```

固定来源：`https://github.com/mlvlab/VideoSearch-R1`，commit `69234fd9783b7235164d9f92b8903d9781dc284e`。工作树 Git tree 与官方一致；恢复方式与归档 SHA256 见 `logs/01_upstream_provenance.json` 和阶段一报告。

网络注意：本次继承的 `127.0.0.1:10809` 代理未监听。安装脚本只在自身进程内忽略它，不改系统代理。Hugging Face 的服务器直连尚未成功；不把网页文档中“已发布”当作本机已经取得资产。

运行环境状态：Conda/Python与构建前置已安装；Torch等正式运行依赖**尚未装完**。本次大包下载吞吐低，安装已暂停并保留缓存，未安排后台续装。应先解决资产网络通路、完整取得并校验官方FlashAttention wheel，再手动继续 `bash scripts/install_runtime.sh`；不能现在直接启动推理。脚本发现wheel缺失或SHA256不符会先停止，不自动退回长时间源码编译。未完成的下载已标为`.whl.partial`，不得安装。

GPU 注意：官方推理/训练 Shell 会自行设置 GPU 编号，不能仅凭 `env.sh` 隐藏 CUDA 就认为任意官方入口都安全。阶段一禁止执行这些入口；后续独立 runner 必须检查用户授权、空闲进程、物理 GPU 和模型/输入哈希。不得运行官方 README 中的系统 CUDA 安装或全局路径修改命令。

根仓库与官方源码仓库分开管理；未创建自己的研究提交，不自动 commit 或 push。
