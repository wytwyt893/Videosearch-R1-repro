# 01_full · Charades 时间戳全量配对对照

把小样本通过的时间标签改动扩展至全部 3720 题，以同一次运行的名义时间轴 A 和源采样时间轴 B 做逐题配对。

**当前结论：VCMR@0.7：6.91% → 7.96%，净增 39 题；仍有失败，不能归结为纯时间轴问题。**

先读 [RESULTS.md](<RESULTS.md>) 查看可汇报的设置、结果和限制。以下编号用于理解步骤；模块文件、历史补救步骤并不要求全部依次执行。

## 脚本入口

| 编号入口 | 职责 | 原实现 |
| --- | --- | --- |
| [00_run_charades_timestamp_full.py](<../../../../../scripts/run_charades_timestamp_full.py>) | 准备/验收 CPU 计划并执行单卡 3720×2 配对运行 | [run_charades_timestamp_full.py](<../../../../../scripts/run_charades_timestamp_full.py>) |
| [01_summarize_charades_timestamp_full.py](<../../../../../scripts/summarize_charades_timestamp_full.py>) | 逐题配对统计、阈值增减、结果 JSON/CSV/Markdown | [summarize_charades_timestamp_full.py](<../../../../../scripts/summarize_charades_timestamp_full.py>) |

历史编号入口为相对符号链接，编辑它会编辑同一份 `scripts/` 实现。保留实现原路径与字节，是为了使历史哈希、模块导入及 CPU 资格检查继续有效。后续新实验在相应阶段、数据集下创建独立方法目录，直接编写编号脚本。

## 本实验产物

| 目录 / 文件 | 内容 |
| --- | --- |
| [RESULTS.md](<RESULTS.md>) | 汇报用结果与解释范围 |
| [reports/](<reports>) | 轻量 JSON、验收、哈希与来源；可提交 Git |
| [outputs/](<outputs>) | 原预测、逐题分析、帧图和运行内日志；仅本机 |
| [logs/](<logs>) | 准备、tmux 启动及补救日志；仅本机 |

### 轻量记录

- [43_charades_timestamp_full_cpu.json](<reports/43_charades_timestamp_full_cpu.json>)
- [44_charades_timestamp_full.json](<reports/44_charades_timestamp_full.json>)
- [45_charades_timestamp_full_summary.json](<reports/45_charades_timestamp_full_summary.json>)

### 运行输出

- [charades_timestamp_full_20261106_011200_472756/](<outputs/charades_timestamp_full_20261106_011200_472756>)

### 原始汇报文档

- [videosearch_r1 × CHARADES全量时间戳对照-11.06.md](<../../../../../docs/archive/videosearch_r1 × CHARADES全量时间戳对照-11.06.md>)

## 使用约定

从项目根目录先执行 `source common/env.sh`。编号 Python 入口可直接以 `python -B experiments/03_optimization/charades/01_timestamp_alignment/01_full/文件名.py` 调用；具体参数先查脚本帮助和原文档。

历史 runner 通常拒绝覆盖已完成报告；原实验目录是证据，不是新一轮实验的空目录。新结果使用新的实验/运行目录，完成后同步记录 Markdown。

[返回实验总目录](<../../../../README.md>)
