# Charades Stage2 实验配置对齐

当前任务是复现表 2 的 Charades-FIG / VideoSearch-R1 发布权重推理结果。本轮只处理配置：不下载数据、不等待 GPU、不启动模型，也不启动训练。

结论：论文明确的推理设置已核对；正式评测配置已显式固定。论文没有交代的设置沿用官方实现并列出不确定性。2026-10-26 已按用户规则决定：论文没有明确规定推理必须限制 4096 个视觉 token，因此不新增该上限。当前不能宣称全部协议和训练来源均已证实与论文一致。

## 已核对并固定的设置

| 项目 | 论文 / 资料依据 | 之前的 smoke | 新的正式评测配置 | 处理结果 |
| --- | --- | --- | --- | --- |
| 主模型规模 | 论文 Qwen3-VL-2B-Instruct；发布权重实际配置属于 2B 结构 | 官方 Charades Stage2 | 同一份固定 revision 权重及 processor | 保留；模型卡的 4B 字段单列为发布信息矛盾 |
| 检索器 | 论文 Qwen3-VL-Embedding-2B | 已校验的本地检索器 | 同一份固定 revision；初始检索使用官方预计算向量 | 已对齐模型规格 |
| 最大轮数 T | 论文 T=2 | 2 | `MAX_TURN=2` | 显式固定，避免内部 Shell 的默认 3 轮 |
| 每次软查询更新 N | 论文 N=8 | 8，实际 rollout 为 8×2048 | `REFINE_TOKEN_COUNT=8` | 显式固定，避免直接 Python 入口默认 1 |
| 视频帧采样 | 论文训练 1 FPS、最多 64 帧；发布测试张量也采用这一目标 | 使用官方 NPY，读取已有帧 | 继续使用官方 NPY 与元数据，不重新抽帧 | 1334 条元数据均为目标 1 FPS / 上限 64 帧；与训练采样规格一致，不证明完整视觉预算一致 |
| VR | 论文 R@1/5/10/100 | 同左 | `TOPK=1,5,10,100` | 已对齐 K；统计时点见后文 |
| VCMR | 论文 IoU 0.3/0.5/0.7 下的 R@1 | 同左 | `TEMPORAL=True`、`IOU_THRESHOLDS=0.3,0.5,0.7` | 已对齐阈值；使用官方 `final_iou_r1`，错误视频或未成功定位为 0 |
| 评测范围 | 官方完整测试集 3720 题、1334 视频检索库 | 前 5 题，完整检索库 | `MAX_SAMPLES=0`、`EVAL_LIMIT_RATIO=1.0`、`RERANKS=all` | 正式配置已取消 smoke 的 5 题限制；旧 smoke 入口和结果保留 |

固定权重版本：

- `VideoSearchR1/charades-stage2`：`c78dc49c6b2cffdf9ae36bd01b738f9b9cdce283`。
- `Qwen/Qwen3-VL-Embedding-2B`：`9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`。
- 官方源码：`69234fd9783b7235164d9f92b8903d9781dc284e`，保持干净。

## 论文没有给出、我们需要明确固定的设置

以下设置不是已发现的论文偏差。它们来自官方实现或本地执行选择，不能在实验记录中写成“论文规定”。

| 设置 | 固定值 | 来源与处理 |
| --- | --- | --- |
| 初始 query 向量 | `USE_UPDATED_QUERY=False` | 官方默认：沿用发布向量，不现场重编码初始 query |
| SQR 查询编码器 | `USE_QUERY_EMBEDDER_PATH=True` | 沿用官方 Shell；当前 checkpoint 无单独 tuned `query_embedder/` |
| 查询前缀 | `Represent the user's input` | 官方数据集 wrapper 默认；显式固定，防止外部环境覆盖 |
| Query pooling / 归一化 | `latent_last` / True | 沿用官方入口 |
| Query 文字最大长度 | 128 | 沿用官方入口 |
| Refine gate | False | 沿用官方 Shell；直接 Python 默认 True，不能混用入口 |
| 整体序列预算 | `MODEL_MAX_LENGTH=20384`，生成最多 1024 | 官方 Shell；vLLM 的上下文上限由二者相加得到 21408。不是 4096 视觉 token 上限 |
| 生成采样 | temperature=0、top_p=1、top_k=-1、repetition penalty=1 | 官方推理代码的贪心设置；论文 GRPO 温度 1.0 属于训练，不能套用到这里 |
| 精度与后端 | BF16、vLLM | 沿用已成功的官方入口路径 |
| 随机种子 | 外层脚本 42；vLLM 未显式接收 seed，既有日志为 0 | 保留并准确记录；论文未规定，不能声称所有组件均为 seed42 |
| 本地执行设置 | batch1、TP1、eager=True、显存比例 0.45 | 保留 smoke 的执行设置；官方 Shell 后两项默认为 False / 0.6。属于本地选择，不宣称不同执行方式必然逐 token 一致 |

正式 profile 还显式固定单 worker、`RESUME_FROM_JSONL=False`；后续正式运行应创建独立输出目录，不能混用旧 smoke 预测。

## 已核实并决定：推理不新增 4096 个视觉 token 上限

论文 §4 的设置限定在训练语境：训练时视觉 token 数设置为 4096，视频以 1 FPS、最多 64 帧采样。这里对 4096 使用“设置为”的表述，没有明确规定推理阶段同样必须满足该硬上限。

按用户 2026-10-26 给出的规则：只有论文明确要求推理限制时才严格添加；没有明确要求时保留官方推理脚本。该项的配置决策现已确定：**保持官方推理像素预算、processor 和发布视频张量，不新增 4096 token 限制，不把整体上下文长度改成 4096。**

此决定不等于证明作者原始推理恰好使用了多少 token；它表示当前复现按论文已明确的语境和官方实现选择参数。后续记录实际输入规模属于可追溯性检查，不再把缺少 4096 推理上限视为当前必改项。

## 仍需区分或确认的事项

| 事项 | 证据与问题 | 当前采用的处理 | 何时调整 |
| --- | --- | --- | --- |
| VER 取轮和分母 | 论文仅定义 match/not-match accuracy；官方脚本分别报告首轮和末次，并排除无效答案 | 保留官方报告；额外审计首轮、末次实际验证视频的正确配对，以及无效答案数/覆盖率。首轮仅作为核对表 2 的候选口径，不宣称作者已确认 | 明确原始论文协议后选定对应统计；无需为改变统计口径重跑模型 |
| 末轮重排 | 官方第 2 轮仍可 SQR；最终检索结果可能未被验证。`final_answer` 与 `final_top1_is_gt` 可能来自不同视频 | 保留官方 VR 的 `final_rank` 和 VCMR 的 `final_iou_r1`；审计 VER 时按 `turns[-1].answer` 与同一轮 `top1_is_gt` 配对。另列最后实际验证前的排名 | 明确原始轮次定义后决定是否增加独立的“末轮不更新”对照；不直接替换官方基线 |
| Stage2 训练来源 | 模型卡写 4B，config 与张量为 2B 结构；已下载权重本身与发布文件一致 | 保留原始发布文件和矛盾记录，继续以“官方发布 Stage2 权重推理”描述实验 | 获得训练记录或作者澄清；不能通过修改 config 字段证明训练来源 |

官方推理像素参数保持：`image_patch_size=16`、`video_min_pixels=100352`、`video_max_pixels=602112`、`video_total_pixels=90316800`。这些不是直接以视觉 token 为单位的配置；也不把 processor 中 `shortest_edge=4096` 解释成 token 上限。

## VER、重排与模型卡的基础说明

首轮指每道题第一次检索出候选视频后，VLM 对该视频做的第一次匹配判断。末次指这道题停止前的最后一次实际判断：第一轮就停止时，首轮与末次是同一次；继续到第二轮时，末次为第二轮。它不是测试集的第一题与最后一题，也不必然等于第 2 轮。

VER 检查“对候选视频的匹配判断是否正确”。候选视频为标注视频时回答匹配才正确；不是标注视频时回答不匹配才正确。首轮 VER 每题取一次首轮判断，末次 VER 每题取一次末次判断。官方脚本排除无法解析的回答，报告必须同时写分母和覆盖率。

例如两道题均成功输出可解析答案：题 1 的第一候选即正确视频，回答匹配后停止；题 2 第一候选错误，先正确回答不匹配，第二轮面对另一个错误视频却回答匹配。首轮准确率为 2/2，末次为 1/2。题 2 首轮的正确拒绝有 VER 得分，但不表示已经检索成功。

重排的是视频检索结果列表。SQR 更新查询表示后，重新计算该查询向量与全部 1334 个视频向量的相似度，再按分数排序。视频向量库和视频内部帧顺序保持原样。末轮继续重排，表示第 2 次验证后仍可更新列表，但新第一名没有第 3 次 VLM 验证；新排名不能作为先前回答所针对的视频。

模型卡是 Hugging Face 模型页面的说明书，对应模型仓库的 `README.md`，包含发布者填写的模型来源与用途等元数据。本地 Charades Stage2 模型卡的 `base_model` 写为 4B，而论文、checkpoint config 和实际权重张量支持 2B 结构。**当前模型规模可以确定为 2B 系列；尚待确认的是来源字段为什么写为 4B，不是当前加载的模型规模无法识别。** 不因这个字段改换 4B 模型或修改 checkpoint 结构。依据见 [模型身份核验](04_MODEL_IDENTITY_CHECK.json)。

13 组同文 query 沿用官方文本映射，保留全部题目。正确视频与时间来自各条原始标注，正式完整性验收按 `index` 进行，不按可能重复的输出 qid 去重。该选择属于保留官方基线，不能宣称已消除了文本映射歧义。

## 本轮已完成的具体处理

1. 新增 [正式评测参数文件](../configs/charades_stage2_eval.bash)，将关键参数显式固定；未修改原 smoke 入口或官方源码。
2. 新增 [配置检查工具](../scripts/check_charades_alignment.py)。它通过真实的官方 Shell 调用链，将模型 Python 替换成参数记录程序，验证最终传参；不会执行评测模块。
3. 检查故意传入错误的继承值（如 T=17、N=1、5 题限制和不同 query 前缀），验证正式 profile 覆盖它们。40 项关键传参核对通过；额外确认单 worker、shard 0 和不续跑旧结果。
4. 以静态解析核验官方视觉像素默认值、生成采样与 vLLM seed 处理，核对模型配置、配置/tokenizer 哈希和全部发布文件大小；大权重 SHA256 继承此前的官方校验记录，本轮不重新计算。
5. 保存 [机器可读检查结果](05_alignment_checks.json)。`configuration_checks_passed=true` 表示上述配置检查通过，不表示完整数据已下载或论文所有未公开设置均已确认。

只检查配置：

```bash
cd /data/wyt/videosearch_r1_repro
python -B scripts/check_charades_alignment.py
```

后续正式入口应依次读取 `scripts/env.sh` 和 `configs/charades_stage2_eval.bash`，再使用 `VIDEOSEARCH_STAGE2_CHECKPOINT` 调用官方 Charades 入口。本轮未启动正式评测，未创建 GPU 等待任务，也未启动数据下载。

SFT 步数、学习率、GRPO 的 G、KL、训练温度和训练视觉预算属于从头训练复现。当前直接评测发布权重，不把这些训练参数误当成推理时待补的开关，也不能仅凭 Stage2 命名宣称训练过程已经复现。

依据：[论文 §3.3 / §4](https://arxiv.org/html/2607.00446v1#S4)、[官方入口](../upstream/scripts/inference/inference.bash)、[内部 Shell](../upstream/scripts/internal/stage1/test_checkpoint_temporal_grounding.bash)、[评测实现](../upstream/scripts/internal/stage1/eval_verified_test_rerank_temporal_grounding.py)、[模型身份证据](04_MODEL_IDENTITY_CHECK.json)。
