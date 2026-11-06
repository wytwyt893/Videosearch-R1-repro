# 环境、模型与初始 smoke 准备 · 结果摘要

整理于 2026-11-06；历史数值来自本目录轻量报告，不重新执行模型。

## 设置与验收

- 官方源码固定在 `69234fd9783b7235164d9f92b8903d9781dc284e`，保留为根目录 `upstream/` 子模块。
- Torch 2.8.0+cu128、FlashAttention 2.8.3.post1；官方依赖 pins、导入和 pip check 的原始验收见本目录 `reports/`。
- 实际 Stage2 VLM 与检索器均为 2B 结构；发布模型卡元数据矛盾不能用于改变已下载权重。
- 初始 Charades 5 题 smoke：5/5 有效、无跳题，3 题触发 SQR。工程链条通过，不视作论文全量分数。

## 本次整理

报告、输出、日志的实体迁入各实验目录；历史相对路径保留兼容链接。70 个历史脚本入口以编号链接组织，`scripts/` 中实现字节与修改时间不变。整理没有重跑 GPU，也没有改变模型、正式数据或实验参数。

`reports/51_workspace_reorganization.json` 记录 837 个迁移文件的原路径、新路径、SHA256、修改时间及清理记录。`90_validate_layout.py` 是本次新增的只读 CPU 目录检查入口。

## 整理验收

清理节省约 **125.17 MiB**：两份临时候选 NPY 与正式修复文件 SHA256 一致，改为相对链接；名义时钟全量预测与历史基线逐字节相同，改为链接；两份复核 NPZ 做无损压缩并核对全部数组；删除空 `tmp/` 目录。模型、原始数据、正式 GT、原预测内容和已验收实现均未改变。

- 76 项 CPU 测试通过。
- 三数据集的编号运行入口均通过参数 dry-run，无模型执行、无 GPU 访问。
- 全量时间戳和最新定位小实验的两份原 CPU 资格继续通过。
- 原始预测、失败日志和关键帧证据均保留；历史 README 快照保留当时文字，不作为当前导航。

完整验收见 [52_workspace_layout_validation.json](<reports/52_workspace_layout_validation.json>)。

[脚本与文件清单](<README.md>) · [实验总目录](<../../experiments/README.md>)

## 按阶段再次归并

按用户后续要求，将工作目录进一步归为 `01_reproduction`、`02_error_analysis`、`03_optimization`。复现阶段按 Charades、DiDeMo、ActivityNet 分目录；ActivityNet 内分数据修复和评测。12 个通用工具实际迁入根 `common/`，旧详细报告收进 `docs/archive/`，当前汇报使用 docs 下三个阶段汇总。

17 个测试模块均保留为回归测试，按 common、reproduction、optimization 分类，通过 `tests/run_tests.py` 收集一次。目录整理保持 104 个已绑定文件的内容和修改时间，原两份 CPU 资格继续有效；没有执行模型或修改参数、正式数据和 GT。

迁移记录：[53_workspace_phase_reorganization.json](reports/53_workspace_phase_reorganization.json)；验收：[54_phase_layout_validation.json](reports/54_phase_layout_validation.json)。
