本报告保留 2026-10-23 20:40 的历史审计状态。后续环境与资产问题已修复并完成 5 题 smoke，最新进度、结果和复跑入口见 [03_FIX_TO_SMOKE.md](03_FIX_TO_SMOKE.md)。

本次复核时间：2026-10-23 20:40，Asia/Shanghai。工作区：`/data/wyt/videosearch_r1_repro`。

**结论：源码和工作区隔离已准备好，正式运行环境、模型及数据尚未准备好。现在可以继续做安装和资产准备，尚不能直接启动可信的论文评测。建议先完成 Charades 发布权重推理复现，再投入 SFT / GRPO 训练。**

本次实际执行了源码/Git、依赖清单、CPU 语法和计分夹具检查、只读宿主机资源查询、少量官方网络元数据请求。没有安装软件、下载模型权重或正式数据，没有启动 CUDA 模型、推理、训练，也没有修改官方源码或现有环境。此前 `01_*` 报告保留；其 GPU 占用、可用内存等快照不能直接代表当前状态。

本次证据文件：

- [CPU、Git、依赖和资产检查](02_readiness_checks.json)
- [宿主机 GPU / 进程 / 拓扑快照](02_host_resources.json)
- [网络与轻量元数据探测](02_remote_metadata.json)
- [官方 FlashAttention wheel 元数据](02_flash_attention_asset.json)

目前的准备情况如下。

| 项目 | 本次实测 | 判断 |
| --- | --- | --- |
| 官方源码 | 167 个跟踪文件；固定提交 `69234fd9783b7235164d9f92b8903d9781dc284e`；本地工作树干净，Git 对象连通检查通过 | 通过 |
| 远端版本 | 本次通过 GitHub API 查询，官方 `main` 与本地提交相同 | 无需为启动准备盲目更新源码 |
| 获取方式 | 旧日志记载曾因 Git 传输故障，使用官方归档与原始 Git 对象恢复；本地 tree 为 `5cc0e48010a6ed0f8e4787c5079abc8c667189d6`，与远端相同 | 内容一致；获取历史并非一次成功的标准 clone |
| 语法 / CPU 检查 | 52 个 Python、85 个 Shell 文件语法通过；6 项合成计分夹具通过 | 证明静态代码可解析，不证明模型可运行 |
| Python | 项目环境为 3.11.17；官方 README 建议 3.11.14，包声明要求 >=3.11 | 补丁版本差异已知；不是当前首要阻塞 |
| 正式运行依赖 | 官方 42 项 pin 中 39 项未满足，实际均未安装；仅 packaging、psutil、ninja 三项匹配 | 阻塞 |
| 项目安装 | `videosearch-r1` 未安装为 editable 包 | 阻塞 |
| 模型 / 数据 / 输出 | `checkpoints/`、`data/`、`outputs/` 文件数均为 0 | 阻塞；无本地复现成绩 |
| 模型下载工具 | `envs/hf_tools` 中 hub 1.20.1；本地 `hf version` 和 `hf buckets --help` 成功 | CLI 可执行，远端下载仍未验证成功 |
| 网络 | 宿主机 PyPI 元数据 HTTP 200；HF 模型 API、模型 config、bucket API 均超时；127.0.0.1:10809 端口拒绝连接 | HF 资产下载通路阻塞 |
| 磁盘 | `/data` 剩余约 24.63 TiB；缓存目录约 6.3 GiB | 当前共享剩余空间充足，非独占配额 |
| CPU / 内存 | 128 逻辑 CPU，503 GiB RAM，约 475 GiB available | 当前足够开展准备工作 |
| CUDA / 编译器 | Toolkit 12.8.61，GCC 11.4.0；ffmpeg 可执行 | 基础工具存在，扩展 ABI 和模型计算尚未验收 |
| GPU0 | A100 80GB；显存已用 70599 MiB，利用率 100%；已有 VLLM 进程 | 当前不宜使用 |
| GPU1 | A100 80GB；显存已用 0 MiB、空闲 81153 MiB、利用率 0% | 可作为后续单卡 smoke 的候选，运行前须重新检查 |

GPU 数据来自宿主机只读查询。普通执行沙箱里的 `nvidia-smi` 无法访问驱动；这不表示宿主机驱动损坏。两卡拓扑显示 `NODE`，没有显示 NVLink。现有两卡也不等同于官方建议的四卡训练配置；单卡或双卡训练能否运行，需要小步实测。

特别注意：当前 `pip check` 虽然输出 `No broken requirements found`，环境中实际上只有 pip、setuptools、wheel、packaging、psutil、ninja。它检查已安装包之间的依赖关系，不能证明 requirements 已装齐。

目录可以按“官方代码、复现工具、运行资产、检查记录”来阅读。

```text
videosearch_r1_repro/
├── README.md                   本地工作区说明和现有准备状态
├── upstream/                   官方源码，保持原始提交
│   ├── README.md               官方安装、下载、推理、训练流程
│   ├── requirements.txt        42 项固定版本依赖
│   ├── pyproject.toml          项目包定义，pip install -e . 的入口
│   ├── videosearch_r1/         核心实现
│   │   ├── model/              模型与软查询相关实现
│   │   ├── search/             检索服务相关实现
│   │   ├── trainer/            训练器
│   │   ├── reward/             GRPO 奖励实现
│   │   └── utils/              公共工具
│   ├── scripts/
│   │   ├── data_construct/     下载预处理数据、构造数据
│   │   ├── inference/          用户层推理与报告入口
│   │   ├── training/           Stage1 SFT / Stage2 GRPO 入口
│   │   ├── internal/           上述入口调用的实际执行脚本
│   │   └── eval/               结果 JSON 与评估辅助代码
│   ├── preprocess/             原始视频/标注转张量、向量和训练数据
│   └── docs/                   项目展示页面、图片和视频
├── envs/
│   ├── videosearchr1/          模型运行环境，目前只有构建前置
│   └── hf_tools/               独立 HF 下载 CLI 环境
├── scripts/                    本地复现工具，不是官方 scripts 的副本
│   ├── env.sh                  路径、缓存、低并发、默认隐藏 GPU
│   ├── install_runtime.sh      正式依赖安装流程，当前被缺失 wheel 阻断
│   ├── audit_stage1.py         原阶段一 CPU 审计入口
│   └── restore_upstream.py     历史网络故障恢复工具，日常不需重跑
├── tools/bin/hf                桥接独立 hf_tools 环境的命令
├── data/                       后续数据集落盘位置，当前为空
├── checkpoints/                后续模型快照，当前为空
├── outputs/                    后续训练、推理等结果位置，当前为空
├── cache/                      pip/Conda/HF 缓存、临时文件、迁移备份
├── logs/                       先前安装、下载和审计的原始日志
└── reports/                    人工报告、机器检查结果、环境快照
```

`cache/` 中的 6.3 GiB 不代表模型已经下载。尤其 `cache/wheels/*.whl.partial` 是未完成下载，不能改名后安装。`cache/path_relocation_20261023/` 保存迁移前资料和环境备份，也不是正式 checkpoint。本次未删除这些可追溯资料。

根工作区和 `upstream/` 分别管理 Git；根 `.gitignore` 排除了源码、环境、缓存和运行资产。因此只看根目录的 `git status`，不能得知官方源码状态，需额外执行 `git -C upstream status --short`。

后续**需要下载模型**，但应按目标选择。项目有两个逻辑模型角色：VLM 负责判断候选视频是否匹配及预测时间段；Embedding 模型负责视频/文本检索和软查询更新。预计算向量可以省掉重复编码数据的工作，但不意味着多轮软查询推理不再需要检索编码器。

| 复现目标 | 需要的模型 | 当前是否具备 |
| --- | --- | --- |
| 首先完成 Charades 发布权重推理 | `VideoSearchR1/charades-stage2` 完整快照，包括 config、tokenizer、processor、模型权重、软查询相关模块和附带文件 | 未下载 |
| 同一推理流程中的检索编码器 | 优先使用 Stage2 快照自带 `query_embedder/`；若不包含，需按官方配置取得 `Qwen/Qwen3-VL-Embedding-2B` 并核验兼容性 | 内容尚无法在线读取，不能假定已内置 |
| 对比 Stage1 / 直接开展 Stage2 GRPO | `VideoSearchR1/charades-stage1`，或自己完成 SFT 后的 checkpoint | 未下载；发布权重推理阶段可先不取 |
| 从头完成论文 SFT 或做原始模型 ZS 对照 | `Qwen/Qwen3-VL-2B-Instruct`，并准备检索编码器 | 未下载 |
| 自行重新生成 reasoning 训练数据 | 源码默认 `Qwen/Qwen3-VL-30B-A3B-Thinking` teacher | 使用官方预处理数据时可跳过 |

其他数据集对应 `didemo-stage1/stage2`、`activitynet-stage1/stage2`，无需一开始把六个 checkpoint 全部下载。官方发布权重和命令映射见[固定提交 README](https://github.com/mlvlab/VideoSearch-R1/blob/69234fd9783b7235164d9f92b8903d9781dc284e/README.md)。由于本次 HF API/config 请求超时，尚未核实各 checkpoint 的实际架构、参数量、revision、文件大小及 query_embedder 内容；“论文为 2B”不能替代对发布权重 config 的核验。

模型之外，还需要数据和运行包。

| 资产 | 首轮 Charades 推理需要什么 | 准备说明 |
| --- | --- | --- |
| Charades 预处理数据 | `hf://buckets/VideoSearchR1/data/datasets/charades-sta` | 官方下载入口会取得完整 prepared bundle，不只是少量测试向量 |
| 测试标注 | `data/charades-sta/raw_annotation/test.jsonl` | 使用 FIG 细粒度描述与时间标注，不可混用任意标准 Charades-STA 标注 |
| 查询和视频向量 | `test/query_embedding/`、`test/video_embedding_1fps/` | 核对行数、维度、顺序、qid 和 video/docid 映射 |
| 视频输入 | `test/video_npy_with_meta/` 及 `meta.jsonl` | 推理仍要读取视频张量；只有 FAISS 或 embedding 文件不足以运行 VLM |
| 训练数据 | `sft_data/`、`grpo_data/`、`train/` | 后续训练才使用；prepared bundle 通常一起包含 |
| FlashAttention wheel | 与 Python 3.11、Torch 2.8、CUDA 12、CXX11 ABI 对应的官方 wheel | 当前仅 7167397 字节 partial；官方完整文件应为 256044010 字节，约 244.2 MiB |
| 其余运行包 | Torch / Transformers / vLLM / DeepSpeed 等官方 pins | Torch 2.8.0 的 PyPI cp311 x86_64 wheel 单文件约 847 MiB，另有 CUDA 依赖；小元数据请求成功不证明大包下载稳定 |

FlashAttention 的发行文件大小与 SHA256 已在本次通过[官方资产 API](https://api.github.com/repos/Dao-AILab/flash-attention/releases/assets/444171392)核实；本地安装脚本中固定的文件名和 SHA256 与之相符，当前因完整文件缺失会先退出。不要把这一退出误判为安装正在继续。

官方 README 给出的 Charades prepared manifest 总量约 107 GiB，快速推理建议至少 130 GB 空间，每个发布 checkpoint 约增加 5 GiB。这些是作者提供的预算，不是本机下载实测。下载脚本还保留分片缓存，解包数据、模型、vLLM 兼容副本与环境也占空间；应以实际 manifest、快照文件清单估算峰值。本机当前磁盘余量足够，首要问题是网络和依赖。采用 prepared 数据路线，无需额外下载原始 MP4 和运行全套特征提取；只有重新构造数据时才需要原始视频。

影响严格复现的代码差异和缺口也需要先处理。

| 优先级 | 具体发现 | 后续处理 |
| --- | --- | --- |
| P0 | HF 资产通路超时，继承代理 127.0.0.1:10809 未监听 | 确认可用代理/直连或从可联网机器下载后传入；先验证小 config、manifest，再验证实际分片/权重大文件地址 |
| P0 | 39 项正式依赖未安装，editable 包未安装 | 完成官方 wheel 校验与运行依赖安装，再做版本匹配、pip check、CPU import 和 GPU smoke |
| P0 | 模型与正式数据均为空 | 固定模型 revision、manifest 哈希，下载完整资产并核对文件；不能从空目录启动评测 |
| P1 | `scripts/common/env.bash:31` 默认 `VIDEOSEARCH_BASE_MODEL=Qwen/Qwen3-VL-4B-Instruct`，论文用 2B | 从头训练前显式指定 2B 或另行解释差异；不要以默认 4B 训练结果称为论文同规格复现 |
| P1 | SFT 脚本默认 1 epoch，`MAX_STEPS` 为空；论文是每数据集 2000 样本、2000 步 | 明确抽样/训练数据与 `MAX_STEPS=2000`、有效 batch、累积步数；对齐最终传给 Trainer 的参数 |
| P1 | 顶层推理默认 T=2，内部 Shell 直接调用默认 T=3，Python 与 Shell 部分默认也不同 | 走顶层入口并显式固定 `MAX_TURN=2`、`REFINE_TOKEN_COUNT=8`、seed、解码和视频预算 |
| P1 | Charades GRPO 入口配置加权奖励和关闭可选 latent 辅助项，不能凭默认值认定与论文完全一致 | 训练前做参数/奖励实现对照；先验证一步训练，再做完整实验 |
| P1 | 模型下载默认未固定 revision，bucket 资产可变化，下载器未显式执行完整 SHA256 验收 | 自行保存 revision、manifest、分片哈希、配置与环境快照，形成可追溯运行清单 |
| P1 | 计分代码跳过坏 JSON，重复 index 后写覆盖，VR 指标排除无效 rank | 正式计分前核对测试题数与预测数、失败/跳过记录、唯一 index 和指标分母，避免漏题导致虚高 |
| P1 | `_load_query_meta` 按完整 query 文本建字典，重复问题会覆盖 | 下载后检查真实 query/qid/video/向量行映射；当前仅合成夹具验证了行为，未证明真实数据存在异常 |
| P1 | `install_runtime.sh:5` 无条件清除代理 | 若后续必须使用代理下载包，要先使安装脚本支持实际可用通路；当前不可假设修好代理后安装器会自动使用 |

论文设置来自[论文 v1 §4](https://arxiv.org/html/2607.00446v1#S4)：2B 骨干、8 个 soft tokens、最大推理轮数 2，以及 SFT 2000 步。训练设定对齐与推理权重复现是不同验收目标；发布 Stage1 也是带软查询能力的 SFT 模型，不应直接当成论文中不含 SQR 的普通 FT baseline。

本地下载 CLI 还有一个可改进点：`tools/bin/hf` 每次都会重新 source `scripts/env.sh`，从而覆盖调用者设置的 `HF_HOME` / `HF_XET_CACHE`；官方分片下载器给每个 worker 设的独立缓存因此会被重置。它不是当前 HF 超时的证据，但恢复并行下载前应使 wrapper 保留调用者缓存设置，或先采用低并发下载。此处为静态确认，本次未改脚本。

后续工作建议按以下顺序推进，每步以验收结果作为下一步的前提。

1. **修通下载通路。** 先使 HF checkpoint config、模型 metadata 和 bucket manifest 能从运行机器读取；再确认实际大文件/CDN 地址可访问。可使用可联网机器预下载并传入，但必须保留官方来源、revision 和哈希。
2. **补全环境。** 取得完整官方 FlashAttention wheel，校验 SHA256；检查安装脚本的代理策略后执行安装；在 `upstream/` 安装 editable 包。所有 42 项 pin 满足后，再验收 `pip check` 与 Torch、Transformers、vLLM、DeepSpeed、FlashAttention、视频解码等必要导入。冻结实际依赖。若要求 Python 补丁版本完全一致，此时另建 3.11.14 环境。
3. **只准备 Charades 首轮资产。** 先下载 `charades-stage2` 完整快照和 Charades prepared 数据；检查快照是否自带 `query_embedder/`。记录模型 revision、config、特殊 token、SQR 模块、向量/标注映射及数据 manifest。
4. **做 5–10 题单卡 smoke。** 运行前复查 GPU1 空闲情况，用单进程、独立输出目录，先测模型加载、视频读取、多轮检索、显存与耗时。限制 query 数即可，保持完整测试视频库，不通过缩小语料库降低检索难度。CPU 审计环境默认隐藏 GPU，正式 runner 必须显式选择物理卡；官方脚本会自行重设 CUDA_VISIBLE_DEVICES。
5. **完成全量推理与报告。** 使用相同模型、完整 corpus 和固定参数，单独保存正式结果目录；确认无未解释漏题/坏行/覆盖，再生成 result JSON，同时保存终端 VER 统计和准确分母。smoke 预测不要与正式输出混用。
6. **再扩展到训练或其他数据集。** 如只需论文推理指标，到第 5 步即可；如需完整训练复现，再补 2B 基座或 Stage1 权重、训练集及 SFT/GRPO 参数对照。先做一步反向传播、保存重载和一步 GRPO，再确定 GPU、batch 与完整训练预算。两张 A100 的可用性、通信与单步显存都需实测，不能用“2B 可加载”替代训练验证。

可以先熟悉以下后续命令；这些命令本次**未执行**。只有环境、网络与资产就绪后才应运行下载或模型入口。

```bash
cd /data/wyt/videosearch_r1_repro
source scripts/env.sh

# 完整 wheel 已存在且下载通路配置正确后，补全正式运行依赖
bash scripts/install_runtime.sh

# 正式依赖就绪后，做现有 CPU 导入审计
# 此脚本会更新 reports/01_engineering_audit.json 和 pip freeze，运行前可归档旧快照
envs/videosearchr1/bin/python -B scripts/audit_stage1.py --imports

cd upstream
# 以下官方数据命令会下载完整 prepared bundle
bash scripts/data_construct/download_preextracted_data.bash charades
# 默认下载可变 revision；严格复现应先固定快照，再以本地路径传给推理入口
bash scripts/download_checkpoint.bash charades-stage2

# 仅为后续 smoke 的示例，先重新检查 GPU1；LOGS_DIR 显式写到 outputs
EVAL_GPUS=1 NUM_PROCESSES_PER_GPU=1 MAX_SAMPLES=5 MAX_TURN=2 \
  REFINE_TOKEN_COUNT=8 SEED=42 RESUME_FROM_JSONL=False \
  LOGS_DIR="$VIDEOSEARCH_OUTPUT_ROOT/charades_smoke" \
  bash scripts/inference/inference.bash charades \
  --checkpoint "$VIDEOSEARCH_CHECKPOINT_ROOT/VideoSearchR1__charades-stage2"

# 输出验收后，给 report.bash 传入实际生成的完整 JSONL 路径
# 全量运行另建 LOGS_DIR，并设 MAX_SAMPLES=0
```

官方推理脚本在未指定 `LOGS_DIR` 时，默认将日志/预测写到 checkpoint 父目录下的 `logs/`，不会自动全部写进根 `outputs/`。显式设置 `LOGS_DIR` 后更便于区分 smoke、正式评测与不同模型。训练输出默认由 `VIDEOSEARCH_OUTPUT_ROOT` 管理。

Charades 首个可对照的目标来自[官方论文表 2](https://arxiv.org/html/2607.00446v1#S4.T2)：VCMR 的 IoU0.3/0.5/0.7 R@1 为 16.5 / 13.4 / 8.2%，VR R@1 为 24.6%，VER 为 75.7%。这些仅是论文参考值，本地目前没有任何推理分数。模型来源、数据版本和评测口径核验后，才可评价复现偏差。
