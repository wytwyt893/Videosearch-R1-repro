# didemo · DiDeMo-FIG 下载、全量评测与离线检查

下载 DiDeMo-FIG 的完整测试资产，完成 CPU 预检、单卡 smoke、全量推理与最小离线对照。

**当前结论：4013 题完成；VR 接近原表，首轮 VER 接近论文，末次 VER 口径单列。**

先读 [RESULTS.md](<RESULTS.md>) 查看可汇报的设置、结果和限制。以下编号用于理解步骤；模块文件、历史补救步骤并不要求全部依次执行。

## 脚本入口

| 编号入口 | 职责 | 原实现 |
| --- | --- | --- |
| [00_prepare_didemo_assets.py](<../../../scripts/prepare_didemo_assets.py>) | 取得 DiDeMo 模型、标注、向量与视频张量 | [prepare_didemo_assets.py](<../../../scripts/prepare_didemo_assets.py>) |
| [01_check_didemo_alignment.py](<../../../scripts/check_didemo_alignment.py>) | 核对 DiDeMo 推理入口与既定协议 | [check_didemo_alignment.py](<../../../scripts/check_didemo_alignment.py>) |
| [02_preflight_didemo_full.py](<../../../scripts/preflight_didemo_full.py>) | DiDeMo 全量前 CPU 资产和环境验收 | [preflight_didemo_full.py](<../../../scripts/preflight_didemo_full.py>) |
| [03_run_didemo_full.py](<../../../scripts/run_didemo_full.py>) | 单卡 smoke/全量 runner；默认参数检查 | [run_didemo_full.py](<../../../scripts/run_didemo_full.py>) |
| [04_score_didemo_protocol.py](<../../../scripts/score_didemo_protocol.py>) | DiDeMo 保存预测的三类指标计分 | [score_didemo_protocol.py](<../../../scripts/score_didemo_protocol.py>) |
| [05_analyze_didemo_offline.py](<../../../scripts/analyze_didemo_offline.py>) | DiDeMo 逐题完整性、取轮及论文差值离线对照 | [analyze_didemo_offline.py](<../../../scripts/analyze_didemo_offline.py>) |

历史编号入口为相对符号链接，编辑它会编辑同一份 `scripts/` 实现。保留实现原路径与字节，是为了使历史哈希、模块导入及 CPU 资格检查继续有效。后续新实验在相应阶段、数据集下创建独立方法目录，直接编写编号脚本。

## 本实验产物

| 目录 / 文件 | 内容 |
| --- | --- |
| [RESULTS.md](<RESULTS.md>) | 汇报用结果与解释范围 |
| [reports/](<reports>) | 轻量 JSON、验收、哈希与来源；可提交 Git |
| [outputs/](<outputs>) | 原预测、逐题分析、帧图和运行内日志；仅本机 |
| [logs/](<logs>) | 准备、tmux 启动及补救日志；仅本机 |

### 轻量记录

- [12_didemo_download_plan.json](<reports/12_didemo_download_plan.json>)
- [13_didemo_assets.json](<reports/13_didemo_assets.json>)
- [14_didemo_alignment.json](<reports/14_didemo_alignment.json>)
- [15_didemo_cpu_preflight.json](<reports/15_didemo_cpu_preflight.json>)
- [16_didemo_final_smoke.json](<reports/16_didemo_final_smoke.json>)
- [17_didemo_smoke_summary.json](<reports/17_didemo_smoke_summary.json>)
- [18_didemo_full_summary.json](<reports/18_didemo_full_summary.json>)
- [19_didemo_offline_summary.json](<reports/19_didemo_offline_summary.json>)

### 运行输出

- [didemo_cpu_preflight_20261028_020342_190460/](<outputs/didemo_cpu_preflight_20261028_020342_190460>)
- [didemo_cpu_preflight_20261028_020641_154946/](<outputs/didemo_cpu_preflight_20261028_020641_154946>)
- [didemo_final_smoke_20261028_020857_537290/](<outputs/didemo_final_smoke_20261028_020857_537290>)
- [didemo_full_20261028_101412_373223/](<outputs/didemo_full_20261028_101412_373223>)
- [didemo_offline_minimum_20261028/](<outputs/didemo_offline_minimum_20261028>)

### 原始汇报文档

- [videosearch_r1 × DIDEMO全量评估-10.28.md](<../../../docs/archive/videosearch_r1 × DIDEMO全量评估-10.28.md>)
- [videosearch_r1 × DIDEMO离线检查-10.28.md](<../../../docs/archive/videosearch_r1 × DIDEMO离线检查-10.28.md>)

## 使用约定

从项目根目录先执行 `source common/env.sh`。编号 Python 入口可直接以 `python -B experiments/01_reproduction/didemo/文件名.py` 调用；具体参数先查脚本帮助和原文档。

历史 runner 通常拒绝覆盖已完成报告；原实验目录是证据，不是新一轮实验的空目录。新结果使用新的实验/运行目录，完成后同步记录 Markdown。

[返回实验总目录](<../../README.md>)
