# VideoSearch-R1 复现与优化

本项目复现 [VideoSearch-R1](https://github.com/mlvlab/VideoSearch-R1) 发布 Stage2 权重的三数据集推理，随后分析时间定位失败并探索优化。官方源码固定为 `69234fd9783b7235164d9f92b8903d9781dc284e`，保留在 `upstream/` 子模块。

**当前阶段：复现完成后的定位优化。** 日常入口：[实验目录](experiments/README.md)；汇报入口：[文档目录](docs/README.md)。

## 当前结果

截至 2026-11-06，Charades-FIG、DiDeMo-FIG、ActivityNet-FIG 完整测试共 **24699 题**完成。对应论文 Table 2 的 VideoSearch-R1 行；训练及其他 baseline 尚未复现。

| Benchmark | 论文 VCMR@0.3 / 0.5 / 0.7 R@1 | 原始复现基线 | 结果 |
| --- | --- | --- | --- |
| Charades-FIG | 16.5 / 13.4 / 8.2 | 16.40 / 12.47 / 6.91 | [Charades](experiments/01_reproduction/charades/RESULTS.md) |
| DiDeMo-FIG | 33.3 / 30.2 / 19.7 | 31.77 / 28.81 / 19.06 | [DiDeMo](experiments/01_reproduction/didemo/RESULTS.md) |
| ActivityNet-FIG | 33.8 / 22.3 / 12.3 | 35.76 / 23.56 / 12.98 | [ActivityNet 修复版](experiments/01_reproduction/activitynet/01_evaluation/RESULTS.md) |

数值为百分比。ActivityNet 使用 364 个修复像素视频和源时间轴，不能将全部差异归因于修复。完整 VR/VER/VCMR 分母和口径见 [复现汇总](docs/01_reproduction.md)。

后续 Charades [源时间戳全量对照](experiments/03_optimization/charades/01_timestamp_alignment/01_full/RESULTS.md)将 VCMR@0.7 从 **6.91% 提高至 7.96%**，净增 39 题，同时 VR@1 有轻微下降。其结果单列，不覆盖上表基线。[二次定位小实验](experiments/03_optimization/charades/02_evidence_verification/RESULTS.md)暂未显示稳定收益，尚未扩大。

## 工作目录

```text
common/                         # 环境、网络、下载、模型准备；早期验收在 00_setup/
experiments/
  01_reproduction/              # charades / didemo / activitynet / summary
  02_error_analysis/            # 错题与标注等离线分析
  03_optimization/              # 按 benchmark 和方法组织的后续优化
docs/                           # 三阶段汇报；旧报告在 archive/
tests/                          # common / reproduction / optimization
configs/                        # 三 benchmark 冻结推理配置
data/  checkpoints/             # 本机数据与模型，不提交 Git
envs/  cache/                   # 本机环境与缓存，不提交 Git
upstream/                       # 官方固定版本子模块
```

每个实验内含编号脚本、`README.md`、`RESULTS.md`、`reports/`，以及本机的 `outputs/`、`logs/`。模型、数据、环境、缓存和运行大文件不提交 Git；轻量报告和 Markdown 可提交。

`common/` 的 12 个工具已实际迁入。历史 benchmark/方法实现保留根 `scripts/`，各实验通过编号入口访问。根 `reports/`、`outputs/`、`logs/`、`doc/` 和旧 experiments 目录为兼容路径，VS Code 默认隐藏；规范目录是上面的主视图。兼容链接保持历史哈希、资格和旧命令有效。

## 本机检查

```bash
source common/env.sh
python -B tests/run_tests.py
python -B common/00_setup/90_validate_layout.py
python -B experiments/01_reproduction/charades/04_run_charades_full.py --dry-run
python -B experiments/01_reproduction/didemo/03_run_didemo_full.py --dry-run
python -B experiments/01_reproduction/activitynet/01_evaluation/02_run_activitynet_repair_full.py
```

以上参数检查不执行模型。实际运行以对应实验 README 和绑定的 CPU/smoke 资格为准，不覆盖历史输出。新机器需要重新准备数据、模型、环境并验收，历史本机资格不能直接继承。

## 共同协议与来源

- 实际为官方 2B Stage2 VLM 和 Qwen3-VL-Embedding-2B 检索器；发布模型卡元数据矛盾不改变下载权重。
- T=2、N=8、BF16/vLLM、batch=1、TP=1、eager，单卡 A100 80GB；外层 seed=42、engine seed=0。
- 主 VER 评价最后实际验证的视频，VR 使用末轮更新后排名，VCMR 使用最后实际验证/定位的视频。论文 VER 取轮尚未正式确认，首轮单列补充。
- 原始测试题和 GT 保留；推理保留官方视觉预算，不强加论文训练中的 4096 token 上限。ActivityNet 数据修复和时间轴变体独立声明。

复现仓库：[wytwyt893/Videosearch-R1-repro](https://github.com/wytwyt893/Videosearch-R1-repro)。

```bash
git clone --recurse-submodules https://github.com/wytwyt893/Videosearch-R1-repro.git
```

后续规范见 [AGENTS.md](AGENTS.md) 与 [实验结果模板](experiments/RESULTS_TEMPLATE.md)。当前候选是实际候选片段核验，先在开发数据验证，再考虑配对留出与完整评测；详见 [优化进展](docs/03_optimization.md)。
