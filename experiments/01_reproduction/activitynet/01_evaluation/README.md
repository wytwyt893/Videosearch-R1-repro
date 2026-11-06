# 01_evaluation · ActivityNet-FIG 原版 smoke 与修复版全量评测

对 ActivityNet 原版做 smoke，对修复像素及源时间轴版本做全量评测，明确数据变体和收尾恢复。

**当前结论：16966 题修复版完成；1 条末次回答无法解析，完整分母和无效数均记录。**

先读 [RESULTS.md](<RESULTS.md>) 查看可汇报的设置、结果和限制。以下编号用于理解步骤；模块文件、历史补救步骤并不要求全部依次执行。

## 脚本入口

| 编号入口 | 职责 | 原实现 |
| --- | --- | --- |
| [00_run_activitynet_full.py](<../../../../scripts/run_activitynet_full.py>) | 官方原版输入的 smoke/全量入口 | [run_activitynet_full.py](<../../../../scripts/run_activitynet_full.py>) |
| [01_run_activitynet_repair_smoke.py](<../../../../scripts/run_activitynet_repair_smoke.py>) | 修复版 smoke 与输入审计 | [run_activitynet_repair_smoke.py](<../../../../scripts/run_activitynet_repair_smoke.py>) |
| [02_run_activitynet_repair_full.py](<../../../../scripts/run_activitynet_repair_full.py>) | 修复版正式全量 runner | [run_activitynet_repair_full.py](<../../../../scripts/run_activitynet_repair_full.py>) |
| [03_score_activitynet_protocol.py](<../../../../scripts/score_activitynet_protocol.py>) | ActivityNet VR/VER/VCMR 计分，单列无效回答 | [score_activitynet_protocol.py](<../../../../scripts/score_activitynet_protocol.py>) |
| [04_finalize_activitynet_repair_saved.py](<../../../../scripts/finalize_activitynet_repair_saved.py>) | 仅 CPU 验证已有完整预测并恢复收尾报告 | [finalize_activitynet_repair_saved.py](<../../../../scripts/finalize_activitynet_repair_saved.py>) |
| [05_compare_activitynet_full_to_paper.py](<../../../../scripts/compare_activitynet_full_to_paper.py>) | 将已完成 ActivityNet 分数与论文对照 | [compare_activitynet_full_to_paper.py](<../../../../scripts/compare_activitynet_full_to_paper.py>) |
| [06_activitynet_repaired_eval.py](<../../../../scripts/activitynet_repaired_eval.py>) | 模型评测适配模块：加载选定修复数据视图 | [activitynet_repaired_eval.py](<../../../../scripts/activitynet_repaired_eval.py>) |

历史编号入口为相对符号链接，编辑它会编辑同一份 `scripts/` 实现。保留实现原路径与字节，是为了使历史哈希、模块导入及 CPU 资格检查继续有效。后续新实验在相应阶段、数据集下创建独立方法目录，直接编写编号脚本。

## 本实验产物

| 目录 / 文件 | 内容 |
| --- | --- |
| [RESULTS.md](<RESULTS.md>) | 汇报用结果与解释范围 |
| [reports/](<reports>) | 轻量 JSON、验收、哈希与来源；可提交 Git |
| [outputs/](<outputs>) | 原预测、逐题分析、帧图和运行内日志；仅本机 |
| [logs/](<logs>) | 准备、tmux 启动及补救日志；仅本机 |

### 轻量记录

- [25_activitynet_final_smoke.json](<reports/25_activitynet_final_smoke.json>)
- [26_activitynet_smoke_summary.json](<reports/26_activitynet_smoke_summary.json>)
- [35_activitynet_repair_smoke.json](<reports/35_activitynet_repair_smoke.json>)
- [36_activitynet_repair_full.json](<reports/36_activitynet_repair_full.json>)
- [36a_activitynet_full_postprocess_recovery.json](<reports/36a_activitynet_full_postprocess_recovery.json>)
- [36b_activitynet_full_summary.json](<reports/36b_activitynet_full_summary.json>)

### 运行输出

- [activitynet_final_smoke_20261029_144528_236908/](<outputs/activitynet_final_smoke_20261029_144528_236908>)
- [activitynet_repair_full_20261030_004132_840892/](<outputs/activitynet_repair_full_20261030_004132_840892>)
- [activitynet_repair_smoke_20261030_003309_295850/](<outputs/activitynet_repair_smoke_20261030_003309_295850>)

### 原始汇报文档

- [videosearch_r1 × ACTIVITYNET原版smoke-10.29.md](<../../../../docs/archive/videosearch_r1 × ACTIVITYNET原版smoke-10.29.md>)
- [videosearch_r1 × ACTIVITYNET修复版全量评估-10.30.md](<../../../../docs/archive/videosearch_r1 × ACTIVITYNET修复版全量评估-10.30.md>)

## 使用约定

从项目根目录先执行 `source common/env.sh`。编号 Python 入口可直接以 `python -B experiments/01_reproduction/activitynet/01_evaluation/文件名.py` 调用；具体参数先查脚本帮助和原文档。

历史 runner 通常拒绝覆盖已完成报告；原实验目录是证据，不是新一轮实验的空目录。新结果使用新的实验/运行目录，完成后同步记录 Markdown。

[返回实验总目录](<../../../README.md>)
