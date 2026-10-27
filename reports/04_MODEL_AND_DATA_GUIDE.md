我们当前集中复现论文表 2 中 **Charades-FIG 数据集、VideoSearch-R1 方法这一行**。目前完成的是作者发布权重的 5 题推理检查，正式成绩需要跑完整的 3720 道测试题。其他方法、其他数据集和从头训练属于后续工作。

VCMR 是任务：根据一句话，从整个视频库中找到对应视频，再给出动作发生的起止时间。Charades-FIG 是用来测试这项能力的数据集。命令中的 `charades`、本地目录 `charades-sta` 是官方代码沿用的名称；当前读的是作者发布的 FIG 细粒度问题描述与时间标注。

| 表 2 指标 | 通俗解释 | Charades-FIG / VideoSearch-R1 论文参考值 |
| --- | --- | --- |
| VR R@1/5/10/100 | 正确视频有没有进入搜索结果前 1/5/10/100 名 | 24.6 / 42.9 / 52.2 / 84.7% |
| VER Acc | 模型判断“这段视频是否符合问题”是否正确 | 75.7% |
| VCMR 0.3/0.5/0.7 R@1 | 第一名视频找对，且时间段重合程度达到对应标准 | 16.5 / 13.4 / 8.2% |

以上为论文数字。本地的 5 题检查用于验证程序链条，尚未完成这组正式分数的复现。

本地有两个不同分工的模型：

| 模型 | 作用 | 位置（相对项目根） |
| --- | --- | --- |
| `VideoSearchR1/charades-stage2` | 看候选视频、判断匹配、定位时间，并生成软查询更新 | `checkpoints/VideoSearchR1__charades-stage2/` |
| `Qwen/Qwen3-VL-Embedding-2B` | 将文字/软查询表示用于检索，与视频向量比较，找到候选视频 | `checkpoints/Qwen__Qwen3-VL-Embedding-2B/` |

推理过程可以理解成：文字描述 → 检索器找视频 → 主模型看片 → 匹配则输出时间段；不匹配则更新查询、再次检索。

两个目录中真正存储学习参数的文件都叫 `model.safetensors`。主模型文件为 4908892434 字节（约 4.9 GB），检索器文件为 4255140312 字节（约 4.3 GB）。当前两份完整权重都已下载并核对官方哈希。

- `config.json`：说明模型结构。
- `tokenizer.json`、`vocab.json` 等：将文字转换为模型能处理的 token。
- `preprocessor_config.json`、`video_preprocessor_config.json`：视频/图像输入处理配置。
- `_vllm_compat/`：从主模型生成的 vLLM 兼容副本，用于文本生成；软查询相关参数由官方评测代码从原 checkpoint 加载。
- `.partial`、`.chunks/`：早期下载留下的缓存。已成功运行的完整权重是 `model.safetensors`。

关于“是不是论文中的 Qwen 2B”，本次额外核验发现了一个发布信息矛盾，需要如实保留：

1. 论文写明主模型基于 `Qwen3-VL-2B-Instruct`，检索器是 `Qwen3-VL-Embedding-2B`。
2. 本地 Stage2 的 config 与实际权重张量显示：28 层、hidden size 2048、intermediate size 6144，与本次读取的官方 2B 配置一致。官方 4B 则是 36 层、hidden size 2560、intermediate size 9728。
3. 但作者发布的 Stage2 模型卡 `README.md` 的 `base_model` 字段写的是 `Qwen/Qwen3-VL-4B-Instruct`。这与实际配置不符，不能忽略，也不能凭推测改成“已证明来自某个基础权重”。

**目前可以确认：运行的是作者发布的 Charades Stage2 checkpoint，实际结构属于 2B 系列。精确训练来源的标注矛盾仍待核实；完整论文分数也仍待全量评测。** 证据见 [模型身份核验](04_MODEL_IDENTITY_CHECK.json)。

2B 表示约 20 亿级参数的模型规格，和 2GB 文件大小是不同概念。Qwen3 是模型家族版本，VL 表示处理视觉与语言，2B 表示规模。

论文描述的训练路线是：基础 `Qwen3-VL-2B-Instruct` → Stage1 监督微调（SFT）→ Stage2 强化学习（GRPO）。我们当前直接加载作者给出的 Stage2 发布权重做推理，没有自行完成这两阶段训练，也没有另外下载原始 2B-Instruct 基座。

表中的 `Qwen3-VL-2B (ZS)` 是原始模型的零样本对照；`Qwen3-VL-2B (FT)` 是普通微调对照。论文的 FT 不包含 SQR，不能直接把作者发布的 Stage1 当成该 FT 对照。本轮优先验证 VideoSearch-R1 这一行。

基础准备的目录可以这样对应：

| 目录 | 放的东西 | 现状 |
| --- | --- | --- |
| `upstream/` | 如何运行模型的官方程序 | 固定提交，保持原样 |
| `envs/videosearchr1/` | Python、Torch 等运行软件 | 已安装、已实际推理验证 |
| `checkpoints/` | 作者训练好的模型权重及配套配置 | 主模型和检索器均已取得 |
| `data/charades-sta/` | 题目、正确答案、搜索向量、视频张量 | 3720 题及 1334 视频的检索库齐全；实际视频张量当前只有 8 个 |
| `scripts/` | 本地准备、检查和执行工具 | 已有环境、数据、模型与 smoke 入口 |
| `outputs/` | 模型实际输出与分数 | 当前有 5 题 smoke 结果 |

视频向量是方便搜索的数字表示；主模型看片仍需要视频张量。后续核心工作是补齐测试视频、核对重复问题编号并完成 3720 题评测。现阶段的主模型和检索器无需重新下载。

参考：[论文实验设置与表 2](https://arxiv.org/html/2607.00446v1#S4)、本地模型配置、权重头部及作者模型卡。
