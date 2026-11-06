# charades · Charades-FIG 原版全量评测与离线检查

复现论文 Table 2 中 Charades-FIG 的发布 Stage2 权重推理，并用离线检查核实题数、映射和计分口径。

**当前结论：3720 题完成；VR 接近原表，严格时间定位有差距。**

先读 [RESULTS.md](<RESULTS.md>) 查看可汇报的设置、结果和限制。以下编号用于理解步骤；模块文件、历史补救步骤并不要求全部依次执行。

## 脚本入口

| 编号入口 | 职责 | 原实现 |
| --- | --- | --- |
| [00_check_charades_alignment.py](<../../../scripts/check_charades_alignment.py>) | 核对 Charades 推理参数与计分协议 | [check_charades_alignment.py](<../../../scripts/check_charades_alignment.py>) |
| [01_prepare_charades_test_videos.py](<../../../scripts/prepare_charades_test_videos.py>) | 取得完整测试视频张量并保存下载凭证 | [prepare_charades_test_videos.py](<../../../scripts/prepare_charades_test_videos.py>) |
| [02_preflight_charades_full.py](<../../../scripts/preflight_charades_full.py>) | 全量前 CPU 环境、数据、模型与参数指纹检查 | [preflight_charades_full.py](<../../../scripts/preflight_charades_full.py>) |
| [03_run_charades_smoke.py](<../../../scripts/run_charades_smoke.py>) | 最初 5 题工程 smoke 的单卡 runner | [run_charades_smoke.py](<../../../scripts/run_charades_smoke.py>) |
| [04_run_charades_full.py](<../../../scripts/run_charades_full.py>) | 正式 smoke/全量 runner；默认只做参数 dry-run | [run_charades_full.py](<../../../scripts/run_charades_full.py>) |
| [05_score_charades_protocol.py](<../../../scripts/score_charades_protocol.py>) | 从保存预测计算 VR、VER 与 VCMR | [score_charades_protocol.py](<../../../scripts/score_charades_protocol.py>) |
| [06_analyze_charades_offline.py](<../../../scripts/analyze_charades_offline.py>) | 离线核对 Charades 覆盖、分母与取轮差异 | [analyze_charades_offline.py](<../../../scripts/analyze_charades_offline.py>) |

历史编号入口为相对符号链接，编辑它会编辑同一份 `scripts/` 实现。保留实现原路径与字节，是为了使历史哈希、模块导入及 CPU 资格检查继续有效。后续新实验在相应阶段、数据集下创建独立方法目录，直接编写编号脚本。

## 本实验产物

| 目录 / 文件 | 内容 |
| --- | --- |
| [RESULTS.md](<RESULTS.md>) | 汇报用结果与解释范围 |
| [reports/](<reports>) | 轻量 JSON、验收、哈希与来源；可提交 Git |
| [outputs/](<outputs>) | 原预测、逐题分析、帧图和运行内日志；仅本机 |
| [logs/](<logs>) | 准备、tmux 启动及补救日志；仅本机 |

### 轻量记录

- [06_primary_protocol_smoke.json](<reports/06_primary_protocol_smoke.json>)
- [07_charades_test_video_receipts.jsonl](<reports/07_charades_test_video_receipts.jsonl>)
- [07_charades_test_videos.json](<reports/07_charades_test_videos.json>)
- [08_cpu_preflight.json](<reports/08_cpu_preflight.json>)
- [09_final_smoke.json](<reports/09_final_smoke.json>)
- [10_charades_full_summary.json](<reports/10_charades_full_summary.json>)
- [11_charades_offline_summary.json](<reports/11_charades_offline_summary.json>)

### 运行输出

- [charades_cpu_preflight_20261026_234603_199979/](<outputs/charades_cpu_preflight_20261026_234603_199979>)
- [charades_cpu_preflight_20261027_000252_194164/](<outputs/charades_cpu_preflight_20261027_000252_194164>)
- [charades_cpu_preflight_20261027_000908_335609/](<outputs/charades_cpu_preflight_20261027_000908_335609>)
- [charades_final_smoke_20261027_000427_560958/](<outputs/charades_final_smoke_20261027_000427_560958>)
- [charades_full_20261027_161353_178368/](<outputs/charades_full_20261027_161353_178368>)
- [charades_offline_minimum_20261027/](<outputs/charades_offline_minimum_20261027>)
- [charades_smoke_20261023_212627/](<outputs/charades_smoke_20261023_212627>)
- [charades_smoke_20261023_213015/](<outputs/charades_smoke_20261023_213015>)

### 原始汇报文档

- [videosearch_r1 × CHARADES全量评估-10.27.md](<../../../docs/archive/videosearch_r1 × CHARADES全量评估-10.27.md>)
- [videosearch_r1 × CHARADES离线检查-10.27.md](<../../../docs/archive/videosearch_r1 × CHARADES离线检查-10.27.md>)

## 使用约定

从项目根目录先执行 `source common/env.sh`。编号 Python 入口可直接以 `python -B experiments/01_reproduction/charades/文件名.py` 调用；具体参数先查脚本帮助和原文档。

历史 runner 通常拒绝覆盖已完成报告；原实验目录是证据，不是新一轮实验的空目录。新结果使用新的实验/运行目录，完成后同步记录 Markdown。

[返回实验总目录](<../../README.md>)
