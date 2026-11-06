# 00_setup · 环境、模型与初始 smoke 准备

建立固定版本环境、官方源码来源、模型身份、数据来源及初始 smoke 的证据链。

**当前结论：环境、模型及 smoke 准备完成；保留官方源码与模型身份的历史核验。**

先读 [RESULTS.md](<RESULTS.md>) 查看可汇报的设置、结果和限制。以下编号用于理解步骤；模块文件、历史补救步骤并不要求全部依次执行。

## 脚本入口

| 编号入口 | 职责 | 原实现 |
| --- | --- | --- |
| [00_audit_stage1.py](<../audit_stage1.py>) | 初始 CPU 工程审计，盘点源码、资产和安装障碍 | [audit_stage1.py](<../audit_stage1.py>) |
| [01_install_runtime.sh](<../install_runtime.sh>) | 按 pins 安装运行依赖并检查 Torch/CUDA 配套 | [install_runtime.sh](<../install_runtime.sh>) |
| [02_repair_decord.py](<../repair_decord.py>) | 修复 decord wheel 的 Python 标签，保留二进制内容 | [repair_decord.py](<../repair_decord.py>) |
| [03_validate_runtime.py](<../validate_runtime.py>) | 仅 CPU 验收依赖、版本和必要导入 | [validate_runtime.py](<../validate_runtime.py>) |
| [04_prepare_models.py](<../prepare_models.py>) | 按固定 revision 取得权重并核验文件哈希 | [prepare_models.py](<../prepare_models.py>) |
| [05_validate_smoke_data.py](<../../scripts/validate_smoke_data.py>) | 检查 Charades 测试向量、映射与初始 smoke 资产 | [validate_smoke_data.py](<../../scripts/validate_smoke_data.py>) |
| [06_http_assets.py](<../http_assets.py>) | 通用模块：HTTPS 大文件分段续传与校验 | [http_assets.py](<../http_assets.py>) |
| [07_remote_tar.py](<../remote_tar.py>) | 通用模块：按 manifest 索引远端 tar 并按需提取 | [remote_tar.py](<../remote_tar.py>) |
| [08_restore_upstream.py](<../restore_upstream.py>) | 按来源记录恢复官方固定版本源码 | [restore_upstream.py](<../restore_upstream.py>) |
| [90_validate_layout.py](<90_validate_layout.py>) | 本次目录迁移的只读 CPU 验收 | 本目录新增脚本 |

历史编号入口为相对符号链接，编辑它会编辑同一份 `scripts/` 实现。保留实现原路径与字节，是为了使历史哈希、模块导入及 CPU 资格检查继续有效。后续新实验在相应阶段、数据集下创建独立方法目录，直接编写编号脚本。

本目录的 `env.sh` 是旧安装器经编号链接调用时使用的环境桥接。`configs`、`scripts`、`upstream` 和初始 smoke 输出的相对链接仅维持历史 Markdown 中的链接，不是额外副本。`reports/history/` 与 `00_WORKSPACE_README_HISTORY.md` 是旧 README 快照，保留当时文字和链接，不作为当前导航。

## 本实验产物

| 目录 / 文件 | 内容 |
| --- | --- |
| [RESULTS.md](<RESULTS.md>) | 汇报用结果与解释范围 |
| [reports/](<reports>) | 轻量 JSON、验收、哈希与来源；可提交 Git |
| [outputs/](<outputs>) | 原预测、逐题分析、帧图和运行内日志；仅本机 |
| [logs/](<logs>) | 准备、tmux 启动及补救日志；仅本机 |

### 轻量记录

- [53_workspace_phase_reorganization.json](reports/53_workspace_phase_reorganization.json)：按复现、分析、优化阶段再次归并的路径与冻结文件记录。
- [54_phase_layout_validation.json](reports/54_phase_layout_validation.json)：三个阶段、分类测试、参数检查、导航和递归忽略验收。
- [91_validate_phase_layout.py](91_validate_phase_layout.py)：当前阶段目录、文档、分类测试入口和 Git 忽略的只读 CPU 检查。

- [51_workspace_reorganization.json](<reports/51_workspace_reorganization.json>)：迁移、文件哈希和清理明细。
- [52_workspace_layout_validation.json](<reports/52_workspace_layout_validation.json>)：目录、入口、链接和 CPU 测试验收。

- [00_WORKSPACE_README_HISTORY.md](<reports/00_WORKSPACE_README_HISTORY.md>)
- [01_ENGINEERING_AUDIT.md](<reports/01_ENGINEERING_AUDIT.md>)
- [01_engineering_audit.json](<reports/01_engineering_audit.json>)
- [02_REPRO_READINESS.md](<reports/02_REPRO_READINESS.md>)
- [02_flash_attention_asset.json](<reports/02_flash_attention_asset.json>)
- [02_host_resources.json](<reports/02_host_resources.json>)
- [02_readiness_checks.json](<reports/02_readiness_checks.json>)
- [02_remote_metadata.json](<reports/02_remote_metadata.json>)
- [03_FIX_TO_SMOKE.md](<reports/03_FIX_TO_SMOKE.md>)
- [03_data_receipts.json](<reports/03_data_receipts.json>)
- [03_data_validation.json](<reports/03_data_validation.json>)
- [03_decord_repair.json](<reports/03_decord_repair.json>)
- [03_duplicate_queries.json](<reports/03_duplicate_queries.json>)
- [03_hf_dns_probe.json](<reports/03_hf_dns_probe.json>)
- [03_models.json](<reports/03_models.json>)
- [03_network_probe.json](<reports/03_network_probe.json>)
- [03_post_smoke_resources.json](<reports/03_post_smoke_resources.json>)
- [03_runtime_validation.json](<reports/03_runtime_validation.json>)
- [03_smoke_integrity.json](<reports/03_smoke_integrity.json>)
- [03_smoke_result.json](<reports/03_smoke_result.json>)
- [04_MODEL_AND_DATA_GUIDE.md](<reports/04_MODEL_AND_DATA_GUIDE.md>)
- [04_MODEL_IDENTITY_CHECK.json](<reports/04_MODEL_IDENTITY_CHECK.json>)
- [05_EXPERIMENT_ALIGNMENT.md](<reports/05_EXPERIMENT_ALIGNMENT.md>)
- [05_alignment_checks.json](<reports/05_alignment_checks.json>)
- [environment.conda.explicit.txt](<reports/environment.conda.explicit.txt>)
- [environment.hf_tools.freeze.txt](<reports/environment.hf_tools.freeze.txt>)
- [environment.pip.freeze.txt](<reports/environment.pip.freeze.txt>)
- [environment.smoke.pip.freeze.txt](<reports/environment.smoke.pip.freeze.txt>)
- [history/](<reports/history>)：历史快照。

### 运行输出

- [configs/](<outputs/configs>)

## 使用约定

从项目根目录先执行 `source common/env.sh`。编号 Python 入口可直接以 `python -B common/00_setup/文件名.py` 调用；具体参数先查脚本帮助和原文档。

历史 runner 通常拒绝覆盖已完成报告；原实验目录是证据，不是新一轮实验的空目录。新结果使用新的实验/运行目录，完成后同步记录 Markdown。

[返回实验总目录](<../../experiments/README.md>)
