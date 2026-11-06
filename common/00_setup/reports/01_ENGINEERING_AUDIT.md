# VideoSearch-R1：阶段一工程审计

日期：2026-10-23，Asia/Shanghai。当前只开展工程审计；**没有运行视频向量编码、推理、SFT 或 GRPO，也没有下载正式数据或模型权重**。本报告明确区分已验证事实、官方声明、静态风险和待验证项。

迁移补记（2026-10-23）：当前根目录为 `/data/wyt/videosearch_r1_repro/`。自有脚本改为按自身位置解析相对路径，两套环境按原版本重建，旧根入口已移除。迁移后 CPU 复核通过，正式运行依赖仍未就绪。以下早期 GPU、容量和资源记录仍为原验收快照；对应原 JSON 保留在 `../cache/path_relocation_20261023/source_backup/reports/01_engineering_audit.json`，本目录中的 JSON 已更新为迁移后复核结果。

## 1. 阶段边界和结论

| 阶段 | 本次状态 | 边界 |
|---|---|---|
| 一：工程审计 | 源码、资源、独立 Conda 及 CPU 静态检查完成；正式运行依赖未装完，明确未就绪 | 不使用 GPU，不自动进入阶段二 |
| 二：Charades 推理复现 | 尚未开始 | 先解决 HF 资产可达性、固定模型版本和数据清单，再申请单卡 smoke |
| 三：训练可行性 | 仅记录静态入口及风险，未做训练验证 | 不能用“2B可加载”推断完整 SFT/GRPO 可训练；完整训练必须人工确认 |

**建议路线：先用官方 Charades Stage2 权重和预处理数据做推理复现，再决定训练投入。当前没有任何本地 benchmark 成绩。**

最终验收时间：2026-10-23 16:48（Asia/Shanghai）。状态为 **AUDIT_COMPLETE / RUNTIME_BLOCKED**，不是“基线复现成功”。

| 最终检查 | 实际结果 |
|---|---|
| Python / Shell语法 | 52个Python文件、85个Shell文件通过 |
| CPU夹具 | 6项通过，包含异常计分与重复问题映射的行为验证 |
| 官方源码 | 固定SHA、Git tree、commit对象与工作区干净检查通过 |
| SetR保护 | 257个代码文件哈希及两个原环境的核心包版本不变 |
| 独立CLI | hub1.20.1的 `hf buckets --help` 通过 |
| 正式运行依赖 | 39项官方pin尚未满足；`runtime_ready_cpu=false`；没有进行模型库完整导入或GPU加载验收 |
| 数据 / checkpoint | 0个正式数据文件、0个模型文件；未开始下载 |
| 本项目空间 | 当前约6.3GiB，主要为源码、包缓存及未完成安装的临时文件 |
| 遗留任务 | 本项目安装/下载进程均已退出；没有GPU等待、推理或训练任务 |

## 2. 主机与共享资源：实际检查

| 项目 | 已验证结果 | 解读 |
|---|---|---|
| 工作区（迁移后） | `/data/wyt/videosearch_r1_repro/` | 不在 SetR 目录下混放新课题 |
| 目标挂载 | `/data` → `/dev/sdb`，ext4 | 已检查目标盘，不是根据用户预估填写 |
| 磁盘 | 总量约58TiB；检查时可用约24.6TiB，`df -h` 显示25T | 是共享剩余空间，不是本项目独占配额；未确认用户级配额 |
| CPU / RAM | Xeon Platinum 8352Y；128逻辑CPU；503GiB RAM，约450GiB available | 本项目安装/编译并发限制为2，不启动满CPU预处理 |
| GPU0 | A100-SXM4-80GB，81920MiB；检查时约69859MiB已占用 | PID2082123 为现有 SetR 的 `VLLM::EngineCore`；不干预 |
| GPU1 | A100-SXM4-80GB，81920MiB；检查时约69817MiB已占用 | PID724073 为其他项目 Python；不干预 |
| 驱动 | NVIDIA 590.48.01 | 驱动支持能力不等同于实际安装的 CUDA Toolkit |
| CUDA Toolkit | `/usr/local/cuda-12.8`；nvcc12.8.61 | 仅通过本项目进程变量只读复用，不改 `/usr/local/cuda` 或系统路径 |
| 双卡拓扑 | `nvidia-smi topo -m` 中 GPU0↔GPU1 为 `NODE`，未显示 `NV#` | 未测通信带宽，不按“具有高速NVLink”估算训练加速 |
| 编译器 | 系统 GCC11.4.0 | 与 Conda Python 的构建编译器不是同一个概念 |

资源是时间点快照，GPU随其他任务变化；完整原始信息见 `01_engineering_audit.json`。本项目未启动 GPU 进程，也未停止任何其他任务。

## 3. 官方源码获取与完整性

| 项目 | 记录 |
|---|---|
| 官方来源 | <https://github.com/mlvlab/VideoSearch-R1> |
| 固定 commit | `69234fd9783b7235164d9f92b8903d9781dc284e` |
| 官方 Git tree | `5cc0e48010a6ed0f8e4787c5079abc8c667189d6` |
| 本地工作区 | `upstream/`；`git status --porcelain` 为空 |
| 实际获取方式 | 标准 `git clone` 和进程级直连浅克隆均发生 TLS 终止；改用 GitHub 官方 codeload 固定提交归档＋官方 Git Data API 恢复浅工作副本。**不是一次成功的标准 clone** |
| 校验 | 全工作树 `git write-tree` 与官方 tree 相同；导入的原始 commit 对象按 Git SHA1 规则精确匹配官方 SHA；`git fsck --connectivity-only` 通过 |
| 源码归档 SHA256 | `ccea3a0f1e86ea03d01c364397c2b883075b7d0ea4f755598aaee485fe248bea` |
| 修改范围 | 官方文件未修改；隔离配置、安装及审计脚本放在工作区根的 `scripts/` |
| Git 操作 | 只导入原作者的已发布提交，没有创建本人的研究提交，没有 push |
| 许可 | 代码仓库 LICENSE 为 Apache-2.0；不能自动推定视频、标注和模型也使用同一许可 |

恢复只接受与官方 commit 完全相同的对象哈希，不用第三方代码镜像，也不伪造替代 commit。可复核入口为 `scripts/restore_upstream.py`，已有工作副本时拒绝覆盖。

## 4. 独立环境及安装状态

| 项目 | 记录 |
|---|---|
| Conda 前缀（迁移后） | `/data/wyt/videosearch_r1_repro/envs/videosearchr1` |
| Python | 实际3.11.17；README推荐3.11.14。完整索引传输失败，小索引无3.11.14，采用可用3.11补丁版创建；这是已记录的环境差异，不宣称逐版本完全一致 |
| Conda 隔离 | `CONDA_PKGS_DIRS`、新前缀均在本项目；`CONDA_REGISTER_ENVS=false`，不写全局环境注册表，不运行 `conda init` |
| 官方核心依赖 | Torch2.8.0、Transformers4.57.1、vLLM0.11.0、DeepSpeed0.16.8、FlashAttention2.8.3.post1、huggingface-hub0.36.0 |
| 构建前置 | 本环境固定 setuptools80.9.0、wheel0.45.1、packaging25.0、ninja1.13.0、psutil7.2.1；先装Torch，再解析官方 requirements，避免空环境下源码包找不到Torch |
| 本地缓存 | HF、pip、Conda、Torch扩展、Triton、CUDA JIT、临时文件、WandB均使用本项目路径；WandB offline |
| GPU隔离 | 阶段一命令设置 `CUDA_VISIBLE_DEVICES=''`；不创建模型或LLM引擎 |
| 运行依赖验收 | **未通过就绪验收**：构建前置已安装，Torch大包未下载完成，官方运行依赖尚未进入完整安装；已停止本项目安装，未安排自动续装。以最终JSON的 `runtime_requirements_mismatch` 和 `runtime_ready_cpu=false` 为准 |
| 下载工具环境 | 独立 `envs/hf_tools/` 固定hub1.20.1；本地 `hf buckets --help` 成功；仅通过 `tools/bin/hf` 桥接CLI，不把新SDK塞进模型环境 |
| 版本记录 | `environment.pip.freeze.txt`；最终另存 Conda explicit 清单 |

环境安装准备脚本已增加前置门禁：只接受官方FlashAttention发行资产ID444171392对应的完整wheel，SHA256应为 `3807faf34d263ad0c3dff6f07c7ebdd27aef4a2614ef124db2afee4578ef7232`；当前只有`.whl.partial`，不会被安装。wheel缺失/校验失败会立即退出，不自动回退长时间源码编译。安装Torch后还会核验CUDA12.8与CXX11 ABI，再安装该wheel；依赖安装脚本本身不是环境已经通过的证明。

SetR 隔离检查：对原项目257个代码文件做前后哈希对照，并检查 `env311`、`qwen_env` 中 Torch/Transformers/vLLM/DeepSpeed 的版本。初次复核无变化；最终结果见JSON。SetR正在运行的预测、日志和自动实验报告允许自然更新，不将其冻结或停止。

## 5. 数据、模型、执行入口

| 环节 | 官方入口 / 路径 | 审计结论 |
|---|---|---|
| 预处理数据下载 | `scripts/data_construct/download_preextracted_data.bash charades` → `scripts/download_data.bash` | 从 `hf://buckets/VideoSearchR1/data/datasets/charades-sta` 下载 manifest 和分片；此阶段未执行下载 |
| 推理模型 | `VideoSearchR1/charades-stage2`；`scripts/download_checkpoint.bash` | 官方声明已发布；本机HF元数据请求超时，尚未证实资产可下载、模型规格和许可 |
| 视频检索器 | `Qwen/Qwen3-VL-Embedding-2B`；若checkpoint有 `query_embedder/`，入口优先采用它 | 必须核验发布模型的嵌入器、软查询投影模块及特殊token，不能拿普通Qwen模型代替 |
| 官方推理 | `scripts/inference/inference.bash charades` → `scripts/internal/stage1/scripts/charades/test_checkpoint_temporal_grounding.bash` → 公共同名入口 → `eval_verified_test_rerank_temporal_grounding.py` | 顶层设置最大轮数2；低层直接运行的默认轮数是3，后续必须走固定入口并显式记录2 |
| 正式计分 | `scripts/inference/report.bash` → `temp_report_temporal_grounding.bash` → `scripts/eval/write_temporal_result.py` | 同时生成终端分析和 `.result.json`；自动结果JSON没有完整包含终端中的验证准确率字段，后续需明确提取口径 |
| Stage1 SFT | `scripts/training/stage1/train.bash charades` → Charades `verification_setting.bash` → 公共训练入口 | 默认基座4B、每卡1条、累积1、1epoch、学习率2e-5；默认没有显式锁2000更新步 |
| Stage2 GRPO | `scripts/training/stage2/train.bash charades` → Charades `stage2.bash` → `train_stage2.bash` | 从发布的Charades Stage1初始化；默认每卡2条、8个采样候选、学习率5e-7、ZeRO1、vLLM colocate；未执行 |

| 资产 | 实际内容 / 后续核对点 |
|---|---|
| `raw_annotation/train.jsonl`、`test.jsonl` | FIG细粒度描述 `fig_desc`、视频ID `video`、标注时间 `time`；不是普通“视频已给定”的QA |
| `test/query_embedding/` | 已发布问题向量和行号/问题/视频映射；与标注逐题对齐 |
| `test/video_embedding_1fps/` | 视频向量与 `docid2row.json`；核验ID唯一、维度、顺序、有限值 |
| `test/video_npy_with_meta/` | 已提取的视频张量及元数据，不等同于只下载几个小向量文件 |
| `test/index/` | 官方提供FAISS资产；实际推理脚本还直接使用视频矩阵打分，后续核验不能只检查FAISS文件存在 |
| `sft_data/`、`grpo_data/` | 作者的训练中间数据；推理复现无需重新生成Teacher数据 |

官方README建议Charades推理至少130GB空闲空间，完整数据清单约107GiB，checkpoint约5GiB/模型。**这些是官方声明，不是本机下载测量值。** 下载脚本保留压缩分片缓存，并解包到数据目录；峰值应同时计入缓存、解包、模型和兼容副本，不能把130GB当保证上限。此处磁盘余量充足，但下载前仍要读取实际manifest预算。

## 6. 已发现的问题、修复与后续门禁

| 优先级 | 事实 / 风险 | 本次处理 | 后续要求 |
|---|---|---|---|
| P0 | 继承代理为未监听的 `127.0.0.1:10809` | 只对本次安装子进程使用直连；没有更改系统代理 | HF直连仍未通过，阶段二前需一个实际可用且获准使用的网络路径 |
| P0 | 论文为Qwen3-VL-2B-Instruct，代码公共默认为4B | 保留原代码，记录冲突 | 发布checkpoint配置到手后先验证规格；训练复现显式选择论文2B并记录，不静默采用4B对照2B表格 |
| P0 | 数据脚本使用 `hf buckets cp`，但依赖固定huggingface-hub0.36.0 | 已创建独立hub1.20.1工具环境，CLI help实测成功；旧0.36模型依赖仍按原计划保留 | 本地CLI兼容问题已有隔离方案；实际HF资产连接、认证和下载尚未通过 |
| P0 | 大包下载吞吐不稳定，不能据小包或初始峰值估整库成本 | Torch15分钟安装流程后下载仍未完成；FlashAttention官方256044010字节wheel在约100秒仅取得7167397字节并超时；已停本项目安装 | 先建立可靠网络、测试持续吞吐，再下载107GiB数据；不在阶段一无限等待，不把部分wheel当成功缓存 |
| P0 | `_load_query_meta` 以完整问题文字为字典键，同名问题后写覆盖 | CPU夹具2条相同Q映射为1条、最后qid保留，已验证函数行为 | 下载后检查同Q的qid、向量行号、视频ID；不能假定qid自然唯一，也不能直接删官方测试题 |
| P0 | 推理遇到缺元数据、缺文档或非法时间会跳过；计分也会跳过坏JSON行、覆盖重复index，检索指标会排除无效rank | CPU夹具已复现行为，未修改官方函数 | 正式计分前强制检查 expected IDs = written + failed + skipped；同题恰一条、零坏行、零未解释跳过、各指标分母明确；不靠删题抬分 |
| P1 | 模型下载未固定HF revision；数据bucket可变；下载脚本未执行SHA256校验 | 保存代码commit与源码哈希 | 固定模型revision，保存manifest哈希、分片/解包文件校验、query与corpus顺序、实际命令 |
| P1 | 直接调用低层推理入口可能用3轮；顶层为2轮；help文本与实际显存利用率默认值也不完全相同 | 跟踪了实际赋值链，未运行入口 | 以最终解析参数为准：T=2、seed42、温度0、8个soft tokens；冻结输入、帧采样和视觉token预算 |
| P1 | 官方合并/续跑以index去重且后写覆盖，未绑定完整模型/数据哈希 | 未复用任何预测，未自动续跑 | smoke/full独立输出；只允许相同模型、数据、提示词、解码参数的续跑与复用 |
| P1 | 训练脚本和论文并非逐项相同：SFT默认1epoch，Charades GRPO使用加权奖励 | 仅记录差异，不私自改奖励或训练步数 | 阶段三单独列参数对照；未确认前不称为严格训练复现 |
| P1 | 当前Python3.11.17不同于README3.11.14 | 如实记录并冻结实际环境 | 后续所有方法使用同环境；若要求逐补丁版本一致，先另建3.11.14环境再冻结基线 |

以上重复/跳过问题是**源码静态行为及人工构造的小夹具结果**，不是声称正式Charades数据已经存在这些异常。正式数据尚未下载，train/test问题交集、实际QA数、视频数、时间越界和重复标注目前都不能填写为“0”或“已通过”。

## 7. 官方Benchmark与论文参考值

来源：[论文v1 §4、表2](https://arxiv.org/html/2607.00446v1#S4.T2)。这里只摘录相同骨干的ZS（未微调）、FT（普通微调）、VideoSearch-R1三方法；不是全表全部竞争方法。

| 指标 | 中文含义 | 注意 |
|---|---|---|
| VR R@K | 正确视频进入前K的比例，K=1/5/10/100 | 衡量视频检索，不等同于回答正确率 |
| VCMR IoU阈值/R@1 | 找对第一名视频，且预测时间段与正确片段的重合度达到阈值 | 视频找错时该题应为0；阈值0.3/0.5/0.7 |
| VER Acc | 对当前候选视频是否匹配问题的判断准确率 | 后续明确采用哪一轮/哪些有效判断的分母，不混同VR或VCMR |

下表全部为**论文百分数**；本地JSON若输出0–1比例，汇报时乘100，不能混用。

| 数据集 | 方法 | VCMR0.3/R@1 | 0.5/R@1 | 0.7/R@1 | VER Acc | VR R@1 | R@5 | R@10 | R@100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Charades-FIG | Qwen3-VL-2B ZS | 12.2 | 7.2 | 2.9 | 30.0 | 21.6 | 41.8 | 51.5 | 84.2 |
| Charades-FIG | Qwen3-VL-2B FT | 12.9 | 10.4 | 7.2 | 74.7 | 21.6 | 41.8 | 51.5 | 84.2 |
| Charades-FIG | VideoSearch-R1 | 16.5 | 13.4 | 8.2 | 75.7 | 24.6 | 42.9 | 52.2 | 84.7 |
| DiDeMo-FIG | Qwen3-VL-2B ZS | 22.0 | 10.6 | 4.0 | 62.8 | 54.8 | 79.3 | 85.6 | 97.0 |
| DiDeMo-FIG | Qwen3-VL-2B FT | 23.6 | 22.1 | 16.7 | 73.1 | 54.8 | 79.3 | 85.6 | 97.0 |
| DiDeMo-FIG | VideoSearch-R1 | 33.3 | 30.2 | 19.7 | 74.6 | 59.0 | 82.0 | 87.8 | 97.5 |
| ActivityNet-FIG | Qwen3-VL-2B ZS | 17.2 | 10.1 | 5.8 | 63.0 | 55.1 | 78.8 | 86.7 | 98.1 |
| ActivityNet-FIG | Qwen3-VL-2B FT | 29.1 | 19.2 | 11.4 | 83.1 | 55.1 | 78.8 | 86.7 | 98.1 |
| ActivityNet-FIG | VideoSearch-R1 | 33.8 | 22.3 | 12.3 | 83.3 | 61.1 | 81.7 | 88.5 | 98.4 |

FT的检索列与ZS相同是论文明确说明的共享检索结果，不是本地补测。本文使用的是VERIFIED派生的FIG细粒度标注；`charades` 是脚本别名、`charades-sta` 是数据目录名，不应与任意标准Charades-STA评测直接混用。

| 本地正式结果 | 数值 |
|---|---|
| Charades完整推理与指标 | 未运行，不能用论文数值代填 |
| 单样本耗时 / 显存峰值 | 未测，不能用模型参数量推算为实测 |
| SFT或GRPO完整训练成功 | 未验证 |

## 8. 后续具体建议：仅规划、不启动

| 顺序 | 工作 | 验收条件 |
|---:|---|---|
| 1 | 完成运行依赖/CPU导入验收，解决HF连接和bucket CLI | 新环境 `pip check`、依赖锁、无CUDA上下文导入通过；能获取轻量manifest和checkpoint config |
| 2 | 固定Charades数据manifest、模型revision及许可 | 原文/张量/向量/标注行号一致；核验2B规格、软查询特殊token与投影权重；记录实际QA/视频数 |
| 3 | 数据完整性、重复问题和官方train/test隔离审计 | 同Q分组检查；test不参与调参或训练；如有官方异常单独报告，不擅自删除或重划测试来优化指标 |
| 4 | 申请一张真正空闲的A100做少量smoke | 先汇报卡号、样本数、时间预算；固定问题、完整检索语料、T=2、温度0；不缩小corpus冒充全量baseline |
| 5 | smoke通过后完整推理和官方计分 | 保存原始逐题JSONL、原始官方JSON、终端完整报告、配置/哈希/显存采样；排除漏题/坏行/重复覆盖；官方值与本地值分列 |
| 6 | 单独阶段三训练审计与最小smoke | 仅在明确授权后测试1–2个更新步；至少验证反向传播、优化器更新、GRPO组内奖励、保存/恢复和显存峰值；不自动启动完整训练 |

双卡GRPO的一个静态约束：默认每卡2条、2进程时生成批量为4，不能被8个采样候选整除；脚本会把 `steps_per_generation` 从1自动调至2，默认累积步数也随之变成2，vLLM TP可自动取2。**这只证明代码有相应配置逻辑，不证明160GB总显存足够、通信快或训练成功。** 官方建议4张A6000级显卡；当前双卡拓扑还需要实测。Stage1默认无DeepSpeed配置、序列上限46384，Stage2则有训练模型/参考策略/嵌入器/vLLM与rollout等开销，不能只看2B权重大小。

## 9. 本次实际入口与日志

| 文件 | 用途 |
|---|---|
| `logs/01_git_clone_direct.log` | 标准Git传输故障 |
| `logs/01_upstream_commit_api.json`、`01_upstream_git_commit.json`、`01_upstream_provenance.json` | 官方SHA、Git对象和归档校验取证 |
| `logs/01_conda_create.log`、`01_conda_create_direct.log`、`01_conda_create_current.json` | 代理、完整索引、小索引推荐版本的失败记录 |
| `logs/01_conda_create_py311.json` | 实际成功创建的独立Python3.11环境记录 |
| `logs/01_runtime_install.log` | 官方运行依赖安装过程和退出原因 |
| `logs/01_runtime_pause.json` | 阶段一暂停原因、仅停止本项目安装PID、未安排自动续装 |
| `logs/01_hf_buckets_help.txt`、`reports/environment.hf_tools.freeze.txt` | 分离下载CLI的实际验收和固定依赖 |
| `logs/01_flash_attention_release.json` | 官方预编译轮子名称、大小、发行资产ID及SHA256；实际下载未完成 |
| `logs/01_cpu_audit.log` | 实际CPU审计输出 |
| `reports/01_engineering_audit.json` | 机器可读资源、源码/依赖、夹具和隔离验收；以最后时间戳为准 |
| `reports/setr_isolation_before.json` | 原SetR代码/环境只读保护基线 |

不自动提交或push。本阶段结束后暂停，等待用户确认下一阶段；没有挂起任何等待GPU后会自行启动的新实验。
