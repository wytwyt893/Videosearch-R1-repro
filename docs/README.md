# 文档与汇报入口

当前处于**发布权重复现完成后的优化阶段**。文档按阶段阅读，实验目录保存具体脚本、原始产物和可追溯验收。

| 文档 | 用途 |
| --- | --- |
| [01_reproduction.md](01_reproduction.md) | 三个 benchmark 的复现范围、论文对照与尚未完成项 |
| [02_error_analysis.md](02_error_analysis.md) | 错题、时间轴和数据标注问题如何区分 |
| [03_optimization.md](03_optimization.md) | 时间戳对照、定位小实验、当前结论及下一步 |
| [archive/](archive/README.md) | 旧版按日期命名的详细报告，保留原文 |

实验文件地图见 [experiments/README.md](../experiments/README.md)；通用准备见 [common/README.md](../common/README.md)；测试说明见 [tests/README.md](../tests/README.md)。

## 后续记录方式

新实验在所属阶段、benchmark 和方法下建目录，包含编号脚本、`README.md`、`RESULTS.md`、`reports/`，以及本机运行所需的 `outputs/`、`logs/`。每轮有效结果写 Markdown，完成后只更新本阶段汇总，不在 `docs/` 根目录继续堆放同层的日期报告。

详细数字以实验 `RESULTS.md` 和轻量 JSON 为准，阶段汇总用于汇报导航。旧 `doc/` 路径是本机兼容入口；新链接优先使用 `docs/` 与规范实验路径。
