# 01 · 发布权重复现

按 benchmark 管理数据、脚本、CPU 验收、smoke、全量和离线检查。这里保留复现基线，后续方法改进放阶段 03。

| Benchmark | 范围 | 入口 |
| --- | --- | --- |
| Charades-FIG | 3720 题、1334 视频，原版全量及离线检查 | [charades/](charades/README.md) |
| DiDeMo-FIG | 4013 题、1002 视频，资产准备至全量及离线检查 | [didemo/](didemo/README.md) |
| ActivityNet-FIG | 16966 题、4866 视频，数据修复与修复版全量分别记录 | [activitynet/](activitynet/README.md) |
| 三数据集总表 | 对应论文 Table 2 的 VideoSearch-R1 行 | [summary/](summary/README.md) |

环境、模型与下载通用工具见 [common/](../../common/README.md)。汇报摘要见 [docs/01_reproduction.md](../../docs/01_reproduction.md)。

[返回实验总目录](../README.md)
