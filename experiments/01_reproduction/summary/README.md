# summary · 论文 Table 2 三数据集结果汇总

用已验收的轻量 JSON 重建三数据集全量结果，对照论文 Table 2 的 VideoSearch-R1 行。

**当前结论：三数据集共 24699 题；仅复现发布权重推理，未复现训练或其他 baseline。**

先读 [RESULTS.md](<RESULTS.md>) 查看可汇报的设置、结果和限制。以下编号用于理解步骤；模块文件、历史补救步骤并不要求全部依次执行。

## 脚本入口

| 编号入口 | 职责 | 原实现 |
| --- | --- | --- |
| [00_build_three_dataset_summary.py](<../../../scripts/build_three_dataset_summary.py>) | 用轻量结果生成论文原表格式总表与 CSV | [build_three_dataset_summary.py](<../../../scripts/build_three_dataset_summary.py>) |

历史编号入口为相对符号链接，编辑它会编辑同一份 `scripts/` 实现。保留实现原路径与字节，是为了使历史哈希、模块导入及 CPU 资格检查继续有效。后续新实验在相应阶段、数据集下创建独立方法目录，直接编写编号脚本。

## 本实验产物

| 目录 / 文件 | 内容 |
| --- | --- |
| [RESULTS.md](<RESULTS.md>) | 汇报用结果与解释范围 |
| [reports/](<reports>) | 轻量 JSON、验收、哈希与来源；可提交 Git |

### 轻量记录

- [37_document_renames.json](<reports/37_document_renames.json>)
- [38_three_dataset_summary.json](<reports/38_three_dataset_summary.json>)
- [38_three_dataset_table2.csv](<reports/38_three_dataset_table2.csv>)

### 原始汇报文档

- [videosearch_r1 × 三数据集全量评估汇总-11.05.md](<../../../docs/archive/videosearch_r1 × 三数据集全量评估汇总-11.05.md>)

## 使用约定

从项目根目录先执行 `source common/env.sh`。编号 Python 入口可直接以 `python -B experiments/01_reproduction/summary/文件名.py` 调用；具体参数先查脚本帮助和原文档。

历史 runner 通常拒绝覆盖已完成报告；原实验目录是证据，不是新一轮实验的空目录。新结果使用新的实验/运行目录，完成后同步记录 Markdown。

[返回实验总目录](<../../README.md>)
