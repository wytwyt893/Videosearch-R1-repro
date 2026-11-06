# 02_evidence_verification · Charades 二次定位与证据提示小实验

在训练来源、首轮检索正确的样本上，比较原始定位、普通二次判断和证据提示二次判断，判断值得继续哪条优化路径。

**当前结论：23 题留出组：证据提示比普通二次判断多命中 1 题，区间含 0，暂不扩大。**

先读 [RESULTS.md](<RESULTS.md>) 查看可汇报的设置、结果和限制。以下编号用于理解步骤；模块文件、历史补救步骤并不要求全部依次执行。

## 脚本入口

| 编号入口 | 职责 | 原实现 |
| --- | --- | --- |
| [00_prepare_charades_localization_probe.py](<../../../../scripts/prepare_charades_localization_probe.py>) | 冻结训练来源样本、开发/留出划分并取得像素 | [prepare_charades_localization_probe.py](<../../../../scripts/prepare_charades_localization_probe.py>) |
| [01_charades_localization_probe_worker.py](<../../../../scripts/charades_localization_probe_worker.py>) | --cpu-check 做三条件输入验收；--run 由 runner 调用 | [charades_localization_probe_worker.py](<../../../../scripts/charades_localization_probe_worker.py>) |
| [02_run_charades_localization_probe.py](<../../../../scripts/run_charades_localization_probe.py>) | 单卡执行三条件定位，记录资源并做配对统计 | [run_charades_localization_probe.py](<../../../../scripts/run_charades_localization_probe.py>) |
| [03_analyze_charades_localization_probe.py](<../../../../scripts/analyze_charades_localization_probe.py>) | 仅离线检查区间变化、提示遵循和静态案例 | [analyze_charades_localization_probe.py](<../../../../scripts/analyze_charades_localization_probe.py>) |
| [04_launch_charades_localization_probe.bash](<../../../../scripts/launch_charades_localization_probe.bash>) | 历史 GPU1 启动命令；不是新实验的默认启动器 | [launch_charades_localization_probe.bash](<../../../../scripts/launch_charades_localization_probe.bash>) |

历史编号入口为相对符号链接，编辑它会编辑同一份 `scripts/` 实现。保留实现原路径与字节，是为了使历史哈希、模块导入及 CPU 资格检查继续有效。后续新实验在相应阶段、数据集下创建独立方法目录，直接编写编号脚本。

## 本实验产物

| 目录 / 文件 | 内容 |
| --- | --- |
| [RESULTS.md](<RESULTS.md>) | 汇报用结果与解释范围 |
| [reports/](<reports>) | 轻量 JSON、验收、哈希与来源；可提交 Git |
| [outputs/](<outputs>) | 原预测、逐题分析、帧图和运行内日志；仅本机 |
| [logs/](<logs>) | 准备、tmux 启动及补救日志；仅本机 |

### 轻量记录

- [47_charades_localization_probe_prepare.json](<reports/47_charades_localization_probe_prepare.json>)
- [48_charades_localization_probe_cpu.json](<reports/48_charades_localization_probe_cpu.json>)
- [48a_charades_localization_probe_pre_uuid_fix.json](<reports/48a_charades_localization_probe_pre_uuid_fix.json>)
- [49_charades_localization_probe_result.json](<reports/49_charades_localization_probe_result.json>)
- [49a_charades_localization_probe_failed_uuid.json](<reports/49a_charades_localization_probe_failed_uuid.json>)
- [50_charades_localization_probe_diagnostics.json](<reports/50_charades_localization_probe_diagnostics.json>)

### 运行输出

- [charades_localization_probe_20261106_182705_399413/](<outputs/charades_localization_probe_20261106_182705_399413>)
- [charades_localization_probe_20261106_183407_371798/](<outputs/charades_localization_probe_20261106_183407_371798>)

## 使用约定

从项目根目录先执行 `source common/env.sh`。编号 Python 入口可直接以 `python -B experiments/03_optimization/charades/02_evidence_verification/文件名.py` 调用；具体参数先查脚本帮助和原文档。

历史 runner 通常拒绝覆盖已完成报告；原实验目录是证据，不是新一轮实验的空目录。新结果使用新的实验/运行目录，完成后同步记录 Markdown。

[返回实验总目录](<../../../README.md>)
