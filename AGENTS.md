# 工作目录与实验约定

用户要求按“复现、错题分析、优化”三个阶段组织实验，并以 Markdown 记录结果，供汇报使用。

- `experiments/01_reproduction/`：按 `charades/`、`didemo/`、`activitynet/` 管理发布权重复现；总表在 `summary/`。
- `experiments/02_error_analysis/数据集/NN_问题/`：错题、标注、计分和输入证据分析。
- `experiments/03_optimization/数据集/NN_方法/`：复现后的方法实验；同一方法可以分 smoke/full 等子目录。
- 新实验在相应阶段和数据集下新增编号目录，脚本直接命名为 `00_prepare.py`、`01_validate.py`、`02_run.py`、`03_analyze.py` 等，按实际步骤增减，不创建空脚本。
- 每个有效实验维护 `README.md`（脚本职责、文件地图）和 `RESULTS.md`（设置、分母、结果、成本、解释范围和下一步）。完成实验后更新 `docs/` 对应阶段汇总及 `experiments/index.json`。
- `reports/` 保存轻量来源、配置、哈希与验收；`outputs/run_id/` 保存每次独立运行；`logs/` 保存准备和启动日志。后两者已递归忽略。
- 通用环境、网络、下载和模型准备工具归 `common/`；不将 benchmark 逻辑放进 common，也不复制通用实现再修改。
- `common/` 中移入的历史工具、根 `scripts/` 中历史实现和根 `tests/test_*.py` 已绑定验收哈希。目录整理保持其字节和修改时间；方法改动放在新实验中，单独验收。
- 分类测试入口在 `tests/common/`、`tests/reproduction/`、`tests/optimization/`；统一运行 `source common/env.sh` 后执行 `python -B tests/run_tests.py`，可用 `--group` 选阶段。新增测试直接放分类目录。
- 根 `doc/`、`reports/`、`outputs/`、`logs/` 和 experiments 的旧平铺目录只作历史路径兼容；新文档与代码使用新路径。
- `docs/archive/` 保留历史详细报告原文。不要重写历史报告、测试 GT 或预测来改进主分数；区分观察、假设和已验证结论。
- 实验目录深度不再固定。新脚本用项目标记向上寻找根目录，不照抄旧脚本的 `parents[1]` 或上一版约定的 `parents[2]`。
- 目录整理仅作 CPU 验证，不因路径移动重跑模型。
