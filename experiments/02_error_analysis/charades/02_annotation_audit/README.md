# 02_annotation_audit · Charades 粗细描述、时间标注与画面审计

核对 cog_desc、fig_desc、原 Charades-STA 时间区间与可见画面，区分模型定位失败和标注风险；正式 GT 不修改。

**当前结论：确认部分粗细动作/时间范围风险；543 条时间越界不等于 543 条语义错标。**

先读 [RESULTS.md](<RESULTS.md>) 查看可汇报的设置、结果和限制。以下编号用于理解步骤；模块文件、历史补救步骤并不要求全部依次执行。

## 脚本入口

| 编号入口 | 职责 | 原实现 |
| --- | --- | --- |
| [00_fetch_charades_annotation_sources.py](<../../../../scripts/fetch_charades_annotation_sources.py>) | 取得固定版本 FIG/STA 及数据集来源资料 | [fetch_charades_annotation_sources.py](<../../../../scripts/fetch_charades_annotation_sources.py>) |
| [01_audit_charades_annotations.py](<../../../../scripts/audit_charades_annotations.py>) | 冻结样本并核对粗细描述、GT 与时长结构 | [audit_charades_annotations.py](<../../../../scripts/audit_charades_annotations.py>) |
| [02_review_charades_annotation_audit.py](<../../../../scripts/review_charades_annotation_audit.py>) | 复核文本和案例画面，输出标注风险证据 | [review_charades_annotation_audit.py](<../../../../scripts/review_charades_annotation_audit.py>) |

历史编号入口为相对符号链接，编辑它会编辑同一份 `scripts/` 实现。保留实现原路径与字节，是为了使历史哈希、模块导入及 CPU 资格检查继续有效。后续新实验在相应阶段、数据集下创建独立方法目录，直接编写编号脚本。

## 本实验产物

| 目录 / 文件 | 内容 |
| --- | --- |
| [RESULTS.md](<RESULTS.md>) | 汇报用结果与解释范围 |
| [reports/](<reports>) | 轻量 JSON、验收、哈希与来源；可提交 Git |
| [outputs/](<outputs>) | 原预测、逐题分析、帧图和运行内日志；仅本机 |

### 轻量记录

- [46_charades_annotation_audit.json](<reports/46_charades_annotation_audit.json>)

### 运行输出

- [charades_annotation_audit_20261106_174412_692658/](<outputs/charades_annotation_audit_20261106_174412_692658>)

## 使用约定

从项目根目录先执行 `source common/env.sh`。编号 Python 入口可直接以 `python -B experiments/02_error_analysis/charades/02_annotation_audit/文件名.py` 调用；具体参数先查脚本帮助和原文档。

历史 runner 通常拒绝覆盖已完成报告；原实验目录是证据，不是新一轮实验的空目录。新结果使用新的实验/运行目录，完成后同步记录 Markdown。

[返回实验总目录](<../../../README.md>)
