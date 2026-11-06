# 01_failure_cases · Charades 定位失败案例与帧证据

从“视频找对但起止时间不准”切入，重建模型实际看到的像素、逐帧时间标签和 GT，识别后续可验证的失败假设。

**当前结论：时间标签偏差、动作与上下文混淆、标注语义范围风险需分别验证。**

先读 [RESULTS.md](<RESULTS.md>) 查看可汇报的设置、结果和限制。以下编号用于理解步骤；模块文件、历史补救步骤并不要求全部依次执行。

## 脚本入口

| 编号入口 | 职责 | 原实现 |
| --- | --- | --- |
| [00_inspect_charades_localization_cases.py](<../../../../scripts/inspect_charades_localization_cases.py>) | 重建首批 6 个案例实际 processor 帧、时间标签和 GT | [inspect_charades_localization_cases.py](<../../../../scripts/inspect_charades_localization_cases.py>) |
| [01_review_charades_localization_cases.py](<../../../../scripts/review_charades_localization_cases.py>) | 制作首批案例静态证据与定性分析 | [review_charades_localization_cases.py](<../../../../scripts/review_charades_localization_cases.py>) |
| [02_inspect_charades_remaining_cases.py](<../../../../scripts/inspect_charades_remaining_cases.py>) | 排除时间轴改动后重建 2233、81 的模型可见输入 | [inspect_charades_remaining_cases.py](<../../../../scripts/inspect_charades_remaining_cases.py>) |
| [03_review_charades_remaining_cases.py](<../../../../scripts/review_charades_remaining_cases.py>) | 制作两例静态帧对照并记录剩余失败假设 | [review_charades_remaining_cases.py](<../../../../scripts/review_charades_remaining_cases.py>) |

历史编号入口为相对符号链接，编辑它会编辑同一份 `scripts/` 实现。保留实现原路径与字节，是为了使历史哈希、模块导入及 CPU 资格检查继续有效。后续新实验在相应阶段、数据集下创建独立方法目录，直接编写编号脚本。

## 本实验产物

| 目录 / 文件 | 内容 |
| --- | --- |
| [RESULTS.md](<RESULTS.md>) | 汇报用结果与解释范围 |
| [reports/](<reports>) | 轻量 JSON、验收、哈希与来源；可提交 Git |
| [outputs/](<outputs>) | 原预测、逐题分析、帧图和运行内日志；仅本机 |
| [logs/](<logs>) | 准备、tmux 启动及补救日志；仅本机 |

### 轻量记录

- [39_charades_localization_cases.json](<reports/39_charades_localization_cases.json>)

### 运行输出

- [charades_localization_cases_20261105/](<outputs/charades_localization_cases_20261105>)
- [charades_remaining_cases_20261106_110858_071732/](<outputs/charades_remaining_cases_20261106_110858_071732>)

### 原始汇报文档

- [videosearch_r1 × CHARADES定位失败案例分析-11.05.md](<../../../../docs/archive/videosearch_r1 × CHARADES定位失败案例分析-11.05.md>)

## 使用约定

从项目根目录先执行 `source common/env.sh`。编号 Python 入口可直接以 `python -B experiments/02_error_analysis/charades/01_failure_cases/文件名.py` 调用；具体参数先查脚本帮助和原文档。

历史 runner 通常拒绝覆盖已完成报告；原实验目录是证据，不是新一轮实验的空目录。新结果使用新的实验/运行目录，完成后同步记录 Markdown。

[返回实验总目录](<../../../README.md>)
