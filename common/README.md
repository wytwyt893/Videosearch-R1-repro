# 通用工具与环境准备

这里放跨 benchmark 复用的实现；三个数据集的实验入口在 [01_reproduction](../experiments/01_reproduction/README.md)，研究实验在 [03_optimization](../experiments/03_optimization/README.md)。

| 文件 | 作用 |
| --- | --- |
| `env.sh`、`network.sh` | 项目环境与进程内网络设置 |
| `install_runtime.sh`、`runtime_constraints.txt` | 安装固定版本依赖 |
| `repair_decord.py`、`validate_runtime.py` | wheel 标签修复与运行环境验收 |
| `http_assets.py`、`remote_tar.py` | 下载、续传、远端 tar 索引与提取 |
| `prepare_models.py` | 固定 revision 的模型准备与校验 |
| `audit_stage1.py`、`restore_upstream.py` | 初始工程审计与官方代码恢复 |
| `smoke_eval.py` | 历史 smoke 评测的本地资产适配 |
| [00_setup/](00_setup/README.md) | 早期准备记录、环境快照、目录整理与只读验收 |

以上 12 个文件已实际从根 `scripts/` 迁入，旧路径保留链接；实现内容和修改时间不变。与单个 benchmark 或方法相关的历史实现仍留在根 `scripts/`，通过各实验的编号入口访问，不混入 common。

```bash
source common/env.sh
python -B tests/run_tests.py
python -B common/00_setup/90_validate_layout.py
python -B common/00_setup/91_validate_phase_layout.py
```

历史下载工具中的默认数据集路径沿用当时配置；复用时确认其参数和调用方。新增通用代码放这里，新增方法代码放对应实验目录。
