# 00_smoke · Charades 时间戳小样本 A/B 与 A/A 复查

固定权重、像素、采样帧数、视觉预算和检索协议，只改变向 VLM 传递的时间标签，并加入相同名义时钟 A/A 重复控制。

**当前结论：70 题 A/B 与同顺序 A/A 完成；诊断与随机样本分别报告。**

先读 [RESULTS.md](<RESULTS.md>) 查看可汇报的设置、结果和限制。以下编号用于理解步骤；模块文件、历史补救步骤并不要求全部依次执行。

## 脚本入口

| 编号入口 | 职责 | 原实现 |
| --- | --- | --- |
| [00_check_charades_timestamps.py](<../../../../../scripts/check_charades_timestamps.py>) | 冻结 6+64 个 QA 并检查时间轴 A/B 像素等价 | [check_charades_timestamps.py](<../../../../../scripts/check_charades_timestamps.py>) |
| [01_run_charades_timestamp_ab.py](<../../../../../scripts/run_charades_timestamp_ab.py>) | 单卡顺序运行同一组 QA 的两种时间标签 | [run_charades_timestamp_ab.py](<../../../../../scripts/run_charades_timestamp_ab.py>) |
| [02_repeat_charades_timestamp_nominal.py](<../../../../../scripts/repeat_charades_timestamp_nominal.py>) | 相同顺序重复名义时间轴，检查 A/A 重放 | [repeat_charades_timestamp_nominal.py](<../../../../../scripts/repeat_charades_timestamp_nominal.py>) |
| [03_summarize_charades_timestamp_ab.py](<../../../../../scripts/summarize_charades_timestamp_ab.py>) | 分诊断/随机组汇总 A/B、A/A 和边界误差 | [summarize_charades_timestamp_ab.py](<../../../../../scripts/summarize_charades_timestamp_ab.py>) |
| [04_build_charades_timestamp_browser.py](<../../../../../scripts/build_charades_timestamp_browser.py>) | 离线生成逐帧时间标签与预测对照 HTML | [build_charades_timestamp_browser.py](<../../../../../scripts/build_charades_timestamp_browser.py>) |
| [05_charades_timestamp_eval.py](<../../../../../scripts/charades_timestamp_eval.py>) | 评测适配模块：仅替换送给 VLM 的时间标签 | [charades_timestamp_eval.py](<../../../../../scripts/charades_timestamp_eval.py>) |

历史编号入口为相对符号链接，编辑它会编辑同一份 `scripts/` 实现。保留实现原路径与字节，是为了使历史哈希、模块导入及 CPU 资格检查继续有效。后续新实验在相应阶段、数据集下创建独立方法目录，直接编写编号脚本。

## 本实验产物

| 目录 / 文件 | 内容 |
| --- | --- |
| [RESULTS.md](<RESULTS.md>) | 汇报用结果与解释范围 |
| [reports/](<reports>) | 轻量 JSON、验收、哈希与来源；可提交 Git |
| [outputs/](<outputs>) | 原预测、逐题分析、帧图和运行内日志；仅本机 |
| [logs/](<logs>) | 准备、tmux 启动及补救日志；仅本机 |

### 轻量记录

- [40_charades_timestamp_cpu.json](<reports/40_charades_timestamp_cpu.json>)
- [41_charades_timestamp_ab.json](<reports/41_charades_timestamp_ab.json>)
- [41a_charades_timestamp_nominal_repeat.json](<reports/41a_charades_timestamp_nominal_repeat.json>)
- [42_charades_timestamp_summary.json](<reports/42_charades_timestamp_summary.json>)
- [42a_charades_timestamp_validation.json](<reports/42a_charades_timestamp_validation.json>)

### 运行输出

- [charades_timestamp_ab_20261105_171942_386257/](<outputs/charades_timestamp_ab_20261105_171942_386257>)
- [charades_timestamp_cpu_20261105_171815_896639/](<outputs/charades_timestamp_cpu_20261105_171815_896639>)
- [charades_timestamp_nominal_repeat_20261105_173238_083843/](<outputs/charades_timestamp_nominal_repeat_20261105_173238_083843>)

### 原始汇报文档

- [videosearch_r1 × CHARADES时间戳对照检查-11.05.md](<../../../../../docs/archive/videosearch_r1 × CHARADES时间戳对照检查-11.05.md>)

## 使用约定

从项目根目录先执行 `source common/env.sh`。编号 Python 入口可直接以 `python -B experiments/03_optimization/charades/01_timestamp_alignment/00_smoke/文件名.py` 调用；具体参数先查脚本帮助和原文档。

历史 runner 通常拒绝覆盖已完成报告；原实验目录是证据，不是新一轮实验的空目录。新结果使用新的实验/运行目录，完成后同步记录 Markdown。

[返回实验总目录](<../../../../README.md>)
