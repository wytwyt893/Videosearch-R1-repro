# VideoSearch-R1 复现记录

本仓库记录 [VideoSearch-R1](https://github.com/mlvlab/VideoSearch-R1) 发布 Stage2 权重在 Charades-FIG、DiDeMo-FIG、ActivityNet-FIG 上的全量推理评测，对应[论文 Table 2](https://arxiv.org/html/2607.00446v1#S4.T2) 的 VideoSearch-R1 行。

当前状态（2026-11-05）：**三个完整测试集均已完成，共 24,699 题。** Charades、DiDeMo 保留官方发布像素与名义时间轴；ActivityNet 使用 364 个修复视频和源时间轴。三个实验均无跳题、无视频元数据缺失，结束后 GPU 已释放。完整 SFT/GRPO 训练及表中其他 baseline 尚未复现。

**[三数据集全量总表：按论文原表分组](<doc/videosearch_r1 × 三数据集全量评估汇总-11.05.md>)** · [汇总 JSON](reports/38_three_dataset_summary.json) · [CSV](reports/38_three_dataset_table2.csv)

## 本轮结果

百分比；本地 VER 使用预先确定的末次实际验证，论文多轮取轮仍未正式确认。

| 数据集 | 来源 | VCMR 0.3 / 0.5 / 0.7 R@1 | VER Acc | VR R@1 / R@5 / R@10 / R@100 |
| --- | --- | --- | ---: | --- |
| Charades-FIG | 论文 | 16.5 / 13.4 / 8.2 | 75.7 | 24.6 / 42.9 / 52.2 / 84.7 |
| Charades-FIG | 本地 | 16.40 / 12.47 / 6.91 | 52.26 | 24.78 / 42.96 / 52.26 / 84.60 |
| DiDeMo-FIG | 论文 | 33.3 / 30.2 / 19.7 | 74.6 | 59.0 / 82.0 / 87.8 / 97.5 |
| DiDeMo-FIG | 本地 | 31.77 / 28.81 / 19.06 | 71.19 | 58.96 / 82.06 / 87.81 / 97.51 |
| ActivityNet-FIG | 论文 | 33.8 / 22.3 / 12.3 | 83.3 | 61.1 / 81.7 / 88.5 / 98.4 |
| ActivityNet-FIG | 本地修复版 | 35.76 / 23.56 / 12.98 | 83.03 | 61.20 / 81.68 / 88.42 / 98.39 |

首轮 VER 补充结果分别为 **74.84% / 74.93% / 84.83%**，不替换末次主指标。ActivityNet 有 1 条无法解析的末次回答，主 VER 为 14086/16965；按全部 16966 题计则为 83.0249%。VR、VCMR 始终使用全部测试题作为分母。

三数据集 VR 接近论文。Charades、DiDeMo 的较严格时间定位仍有差距；ActivityNet 修复版 VCMR 高于原表，但缺少同配置原版全量对照，暂不能归因于修复本身。详细分母、差值、输入差异及候选研究问题见总表。

## 实验文档

`doc/` 文件统一使用 `videosearch_r1 × 报告内容-MM.DD.md`。日期沿用原实验/报告日期，年份和完整时间见正文；ActivityNet 全量 10.30 启动、10.31 完成，新汇总整理于 11.05。

| 数据集 | 完整评测 | 其他记录 |
| --- | --- | --- |
| Charades-FIG | [全量评估-10.27](<doc/videosearch_r1 × CHARADES全量评估-10.27.md>) | [离线检查-10.27](<doc/videosearch_r1 × CHARADES离线检查-10.27.md>) |
| DiDeMo-FIG | [全量评估-10.28](<doc/videosearch_r1 × DIDEMO全量评估-10.28.md>) | [离线检查-10.28](<doc/videosearch_r1 × DIDEMO离线检查-10.28.md>) |
| ActivityNet-FIG | [修复版全量评估-10.30](<doc/videosearch_r1 × ACTIVITYNET修复版全量评估-10.30.md>) | [原版 smoke-10.29](<doc/videosearch_r1 × ACTIVITYNET原版smoke-10.29.md>) |

- 轻量全量结果：[Charades](reports/10_charades_full_summary.json)、[DiDeMo](reports/18_didemo_full_summary.json)、[ActivityNet](reports/36b_activitynet_full_summary.json)。
- 离线对照：[Charades](reports/11_charades_offline_summary.json)、[DiDeMo](reports/19_didemo_offline_summary.json)。
- ActivityNet：[修复数据概况](reports/33_activitynet_repair_summary.json)、[完整性检查](reports/34_activitynet_repair_integrity.json)、[修复版 smoke](reports/35_activitynet_repair_smoke.json)、[全量记录](reports/36_activitynet_repair_full.json)、[离线收尾恢复](reports/36a_activitynet_full_postprocess_recovery.json)。
- [重命名、原始哈希与历史路径映射](reports/37_document_renames.json)。历史审计保留当时状态，以本页和三数据集总表判断当前进展。

ActivityNet 全量在 GPU1 实测约 27 小时 50 分钟。模型进程正常退出；收尾脚本曾误把 smoke 的零无效回答要求用于全量，后续已只用 CPU 验证原预测并补齐报告。原失败日志、退出码及清单保留，模型没有重跑，唯一无效回答没有被删除。

## 共同实验协议

- 各数据集官方发布的 2B Stage2 VLM，复用 `Qwen/Qwen3-VL-Embedding-2B` 检索编码器；发布模型卡元数据的历史矛盾不用于改变实际模型规模。
- T=2、8 soft tokens、BF16/vLLM、batch=1、TP=1、eager、单卡 A100 80GB；外层 seed=42，官方 vLLM 构造器未传入该 seed，engine 默认 seed=0。
- VER 配对最后实际检查的视频；VR 使用末轮更新后的排名；VCMR 使用最后实际验证、定位的视频。保留官方触发的末轮 SQR/重排，不增加第三次验证。
- 保留官方推理视觉预算，不增加论文训练设置中的 4096 视觉 token 硬上限。ActivityNet 单独声明画面与时间轴修复。
- 原始题号、题目、GT、视频库和检索向量保留。镜像源视频不声称与作者源文件逐字节一致；原始标注边界问题与两例视觉复核例外继续记录。

## 代码来源与克隆

复现工具和实验记录提交到 [wytwyt893/Videosearch-R1-repro](https://github.com/wytwyt893/Videosearch-R1-repro)。官方源码以 `upstream/` 子模块保留，固定提交 `69234fd9783b7235164d9f92b8903d9781dc284e`。

```bash
git clone --recurse-submodules https://github.com/wytwyt893/Videosearch-R1-repro.git
cd Videosearch-R1-repro
git -C upstream rev-parse HEAD
```

已有普通 clone 可运行 `git submodule update --init upstream`。工作区 `.vscode/settings.json` 仅关闭本项目的嵌套仓库检测，让源代码管理显示根复现仓库。

| 目录 | 内容 | 提交到 Git |
| --- | --- | --- |
| `upstream/` | 固定提交的官方源码 | 子模块指针 |
| `scripts/`、`tools/`、`configs/`、`tests/` | 本地工具、冻结配置与协议测试 | 是 |
| `doc/` | 实验报告和三数据集总表 | 是 |
| `reports/` | 审计、轻量结果、哈希与历史记录 | 是；其他项目隔离快照除外 |
| `data/`、`checkpoints/` | 视频、向量、标注、模型及兼容派生权重 | 否 |
| `envs/`、`cache/` | 本地环境、下载缓存和凭据 | 否 |
| `outputs/`、`logs/`、`tmp/` | 原始预测、日志、显存采样和临时文件 | 否 |

原始输出和缓存链接依赖本机资产；GitHub 上可直接查看 `doc/`、轻量 `reports/` 和 CSV。克隆仓库不会恢复模型、数据与环境，历史 CPU 指纹也不能代替新机器验收。

## 本机入口

先准备项目环境、固定 revision 模型及完整测试数据；参考[官方说明](upstream/README.md)与本地验收工具。以下仅核对参数，不执行模型：

```bash
source scripts/env.sh
python -B scripts/run_charades_full.py --dry-run
python -B scripts/run_didemo_full.py --dry-run
python -B scripts/run_activitynet_repair_full.py
```

本轮 ActivityNet 修复版使用 `run_activitynet_repair_full.py`；`run_activitynet_full.py` 仍是官方原版输入入口。实际执行需要 `--execute --gpu N`，通过完整 CPU 检查和当前输入绑定的 smoke，并使用独立结果目录及报告路径。

三数据集总表仅依赖已发布的轻量 JSON，可用 `scripts/build_three_dataset_summary.py` 生成；现有日期的文档默认拒绝覆盖。设置、来源、协议和兼容工具均有记录，不修改其他研究项目或系统配置。

## 后续工作

先对 ActivityNet 原版、仅修复画面、画面加源时间轴做固定配置的配对对照，再根据时间定位和 SQR 轨迹分析确定优化问题。候选方向与证据见三数据集总表；方法新颖性需要进一步查阅相关工作并验证。
