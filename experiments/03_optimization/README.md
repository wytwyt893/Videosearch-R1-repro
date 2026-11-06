# 03 · 复现后的优化实验

当前主线为 Charades 时间定位，按“数据集 / 方法 / 对照规模”管理。未来可新增 DiDeMo、ActivityNet 或其他方法目录。

| 方法 | 状态 | 入口 |
| --- | --- | --- |
| 时间戳对齐 | 70 题配对与 A/A 已完成；3720×2 全量已完成 | [charades/01_timestamp_alignment/](charades/01_timestamp_alignment/README.md) |
| 二次定位与证据提示 | 35 题条件定位小实验完成，尚无稳定收益，不扩大 | [charades/02_evidence_verification/](charades/02_evidence_verification/README.md) |

新方法从 `数据集/03_方法名/` 继续编号，在目录内实际编写 `00_prepare.py`、`01_validate.py`、`02_run.py`、`03_analyze.py` 等。后续结果均写 Markdown。归因时固定基线、数据划分、计分与资源预算，保留新增命中和丢失。

汇报摘要：[docs/03_optimization.md](../../docs/03_optimization.md)。

[返回实验总目录](../README.md)
