# VideoSearch-R1 实验目录

当前处于**复现完成后的优化阶段**。实验按阶段分组，再按 benchmark、分析问题或方法继续划分；目录内保留编号脚本、结果与原始产物。

```text
experiments/
  01_reproduction/              # 发布权重复现
    charades/
    didemo/
    activitynet/
      00_data_repair/
      01_evaluation/
    summary/                   # 三数据集论文原表对照
  02_error_analysis/            # 复现后的错题分析
    charades/
      01_failure_cases/
      02_annotation_audit/
  03_optimization/              # 复现后的方法优化
    charades/
      01_timestamp_alignment/
        00_smoke/
        01_full/
      02_evidence_verification/
```

| 阶段 | 当前内容 | 汇报摘要 |
| --- | --- | --- |
| [01_reproduction](01_reproduction/README.md) | 三 benchmark 共 24699 题发布权重全量；ActivityNet 单列修复 | [复现汇总](../docs/01_reproduction.md) |
| [02_error_analysis](02_error_analysis/README.md) | 可见帧、GT、定位失败、粗细描述与标注风险 | [问题分析](../docs/02_error_analysis.md) |
| [03_optimization](03_optimization/README.md) | 时间轴全量已完成；证据提示小实验尚无稳定收益 | [优化进展](../docs/03_optimization.md) |

## 通用目录

- [common/](../common/README.md)：12 个跨 benchmark 工具的实际实现，以及环境和早期审计 `00_setup/`。
- [docs/](../docs/README.md)：按三个阶段汇报；历史详细报告收进 `archive/`。
- [tests/](../tests/README.md)：按 common / reproduction / optimization 分类的 17 个回归测试模块。
- [index.json](index.json)：阶段、实验和规范路径的机器可读索引。

## 单个实验的结构

```text
03_optimization/charades/03_方法名/
  00_prepare.py
  01_validate.py
  02_run.py
  03_analyze.py
  README.md                  # 脚本职责与产物地图
  RESULTS.md                 # 可汇报的实验设置、分母、结果、成本与限制
  reports/                   # 轻量配置、来源、验收、哈希
  outputs/运行ID/            # 预测、图、运行日志、显存记录；不提交 Git
  logs/                      # 准备与启动日志；不提交 Git
```

这是后续命名规则，不是已经创建的空实验。按实际步骤增减脚本，使用 [RESULTS_TEMPLATE.md](RESULTS_TEMPLATE.md) 记录结果，完成后更新所属阶段的 docs 汇总。实验层级可以嵌套，不再依赖固定 `parents[2]` 推导根目录。

历史编号入口仍链接到同一份已验收实现。旧 experiments 平铺目录、根 reports/outputs/logs/doc 仅作兼容，在 VS Code 中隐藏；新工作使用上面的规范目录。原始模型、正式标注、预测和已验收代码未因整理改变。

目录整理证据：[53_workspace_phase_reorganization.json](../common/00_setup/reports/53_workspace_phase_reorganization.json)。旧版细分迁移记录和资格仍在 [common/00_setup](../common/00_setup/README.md)。
