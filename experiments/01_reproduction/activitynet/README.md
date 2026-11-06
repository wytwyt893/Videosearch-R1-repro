# ActivityNet-FIG 复现

ActivityNet 的数据对应和修复工作较多，在 benchmark 内分成两步，避免和后续算法优化混在一起。

- [00_data_repair/](00_data_repair/README.md)：EDA、采样异常审计、原视频来源对应、364 个修复、数据视图及 CPU 完整性验收。
- [01_evaluation/](01_evaluation/README.md)：原版 smoke、修复版 smoke/全量、计分及 CPU 收尾恢复。

数据修复版全量覆盖 16966 题。结果与限制见 [01_evaluation/RESULTS.md](01_evaluation/RESULTS.md)；缺少同配置原版全量消融，不能把相对论文的全部差值归因于修复。

[返回复现阶段](../README.md)
