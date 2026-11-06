本轮目标：修复环境与资产准备问题，完成 Charades 官方发布权重的 5 题单卡 smoke，并保存预测、计分与资源记录。当前用户已授权推进到 smoke；不启动完整训练或全量 benchmark。

工作区：`/data/wyt/videosearch_r1_repro`。官方源码保持固定提交，本地兼容工具放在根 `scripts/`，不修改其他项目或系统配置。

| 步骤 | 工作 | 验收条件 | 当前状态 |
| --- | --- | --- | --- |
| 1 | 修复下载通路与 CLI 缓存覆盖 | 官方模型 metadata、config、bucket manifest 可读取；调用者缓存设置保留 | 已完成 |
| 2 | 补全运行环境 | 完整 FlashAttention wheel SHA256 正确；依赖 pins、editable 安装、pip check 与必要导入通过 | 已完成：42 项 pins 匹配，必要 CPU 导入、pip check、editable 安装通过 |
| 3 | 固定并取得模型 | Stage2 revision 固定；模型/config/tokenizer 完整；检索编码器及软查询模块核验 | 已完成：两个完整模型通过官方大小与哈希核验；Stage2 包含 16 个软查询相关 tensors |
| 4 | 准备 smoke 数据 | 完整测试检索库向量、查询/标注映射正确；smoke 所需视频张量可读取；数据清单及哈希保存 | 已完成：完整测试向量/元数据验收通过，全部测试视频可定位，5 个初始候选张量可读取 |
| 5 | 单卡 smoke | 运行前检查空闲 GPU；固定 T=2、8 soft tokens、seed42；独立输出目录；5 题预测及计分完整 | 已完成：5/5 有效预测，零跳题、零元数据缺失；3 题触发 SQR |
| 6 | 记录结果与剩余工作 | 保存实际命令、版本、模型/数据来源、日志、预测、result JSON、耗时及峰值显存 | 已完成：预测、报告、版本/哈希和 GPU 释放证据均已保存 |

2026-10-23 初始进展：

- 发现可用代理 `http://127.0.0.1:7890`。仅通过本项目进程变量使用，不修改系统代理。
- 官方 Stage2 revision：`c78dc49c6b2cffdf9ae36bd01b738f9b9cdce283`；公开、非 gated；主 safetensors 大小 4908892434 字节。
- 官方 Charades manifest 为 108 个逻辑 tar shard、6709 个文件，声明总大小 114703834511 字节。首先分析分片内容，确定能否只下载 smoke 所需资产；测试检索 corpus 不缩小。
- 修复安装器无条件清除代理的问题；修复 `tools/bin/hf` 重置 worker 的 HF 缓存变量问题。
- 本轮原始网络证据保存在 `03_network_probe.json`，模型与 manifest 在 `cache/manifests/`。

执行进展补记：

- 完整 FlashAttention wheel 与 Torch wheel 均通过官方 SHA256；12 个 CUDA 依赖缓存也与 PyPI 哈希比对通过，已复用约 2.7 GiB。
- 实际 Torch 为 2.8.0+cu128，FlashAttention 2.8.3.post1，CXX11 ABI=True；CPU 导入未初始化 CUDA。
- 取得 3720 个测试问题及其 2048 维向量、1334 个完整测试视频向量、FAISS 与视频元数据。
- 实际有 13 种重复 query 文本，其中 1 组涉及不同视频，向量最大绝对差约 7.8e-7；前 5 条原始样本均不重复且视频映射正确。详见 `03_duplicate_queries.json`，未删除或替换测试题。
- 仅为 smoke 按需取得候选视频张量；没有下载全部训练视频和整个 107 GiB bundle。检索用的完整 1334 视频向量库保持原样。
- SDK 元数据头不兼容通过固定 revision 的官方 HTTPS 文件与哈希校验解决；整文件低速传输改为可续传分段下载。
- 数据 manifest 对 00000、00021、00107 指定了拆分资产，旧完整 tar 内容与之不同。索引严格遵循 manifest；旧完整 tar 的早期索引已归档，不用于运行。
- 后续核实：已完成的索引覆盖全部 1334 个测试视频，缺失测试视频索引数为 0。停止后续训练分片索引，TLS 失败的剩余训练资产不再作为 smoke 的前置条件；完整训练数据仍未准备。
- 完整测试数据索引实际只需前 22 个逻辑分片（1400 个文件，其中 1334 个测试视频张量）。模型与数据完整性证据分别为 `03_models.json`、`03_data_validation.json`、`03_data_receipts.json`。
- Hugging Face 主域名新连接出现 TLS EOF，官方 `hf.co` 域名可正常提供同一模型和 bucket；本项目 `HF_ENDPOINT` 默认使用该官方域名，可通过 `VIDEOSEARCH_HF_ENDPOINT` 覆盖，不修改系统 DNS。
- decord 官方 wheel 文件名标为 py3-none、内部 WHEEL 误标 cp36-cp36m。制作 build tag 为 `1py311fix` 的本地 wheel，仅修改 WHEEL 与 RECORD；Python 和 `libdecord.so` 内容均未改变，CPU 解码两帧测试通过。原件、修复件和哈希证据均保留，详见 `03_decord_repair.json`。
- requests 2.32.5 与传递依赖 chardet 6.x 不兼容，已在 `scripts/runtime_constraints.txt` 固定 chardet 5.2.0；官方 42 项版本均未改动。当前环境快照：`environment.smoke.pip.freeze.txt`。
- smoke 首次模型加载时，官方代码成功加载全部 16 个 refine tensors；随后 vLLM 0.11 对 GPU UUID 字符串执行 int() 而失败。本地 runner 改为数字编号，并在运行前用 Torch UUID 与 NVML UUID（归一化 GPU- 前缀）交叉核验物理卡。失败日志保留，不复用失败输出。

本轮最终状态为 **COMPLETE_TO_SMOKE**。成功运行于 2026-10-23 21:30:15 开始，结果目录为 `outputs/charades_smoke_20261023_213015/`。

| 实际验收项 | 结果 |
| --- | --- |
| GPU | 物理 GPU1，A100 80GB；CUDA / NVML UUID 核验一致 |
| 样本 | 官方测试集原始 index 0–4，未换题 |
| 输入检索库 | 完整 1334 个视频向量；3720 个 query 向量 |
| 参数 | T=2、8 soft tokens、seed42、BF16、vLLM、batch1、eager、GPU memory utilization 0.45 |
| 预测完整性 | 5/5 条，index 唯一，所有 skip 为 0 |
| 视频元数据 | 8/8 次输入命中，缺失 0；当前取得 8 个视频张量 |
| SQR | 3 题使用软查询更新，实际应用 5 次 |
| 耗时 | 123.03 秒，包含模型初始化、运行中 3 个视频按需下载及退出 |
| 设备显存采样峰值 | 45583 MiB，约 44.51 GiB；包含 vLLM KV 缓存预留，不代表最低显存需求 |
| 计分链条 | result JSON 已生成；VR 与 VCMR 分母均为 5 |
| 退出检查 | GPU1 显存回到 0 MiB、利用率 0%；本轮无安装、下载或推理任务残留 |
| 官方源码 | 保持固定 commit，工作树干净 |

5 题工程检查的 VR R@1 为 20%，VCMR IoU0.3 / 0.5 / 0.7 R@1 为 20% / 20% / 0%。这些结果用于确认预测与计分链条工作；论文指标需要对全部 3720 题正式评测。

可直接打开的结果和证据：

- [运行摘要](03_smoke_result.json)
- [完整性、代码及输入哈希](03_smoke_integrity.json)
- [结束后的 GPU / 进程检查](03_post_smoke_resources.json)
- [依赖验收](03_runtime_validation.json)、[数据验收](03_data_validation.json)、[模型验收](03_models.json)
- [预测 JSONL](../outputs/charades_smoke_20261023_213015/external_verified_test_temporal_grounding_VideoSearchR1__charades-stage2.jsonl)
- [计分 JSON](../outputs/charades_smoke_20261023_213015/external_verified_test_temporal_grounding_VideoSearchR1__charades-stage2.result.json)
- [实际推理日志](../outputs/charades_smoke_20261023_213015/inference.log)、[计分日志](../outputs/charades_smoke_20261023_213015/report.log)、[显存曲线 CSV](../outputs/charades_smoke_20261023_213015/gpu_memory.csv)

复跑 smoke 的入口如下。runner 会先检查依赖、资产和 GPU 空闲状态；每次创建独立输出目录。

```bash
cd /data/wyt/videosearch_r1_repro
source scripts/env.sh
python -B scripts/run_charades_smoke.py --gpu 1
```

当前工具的职责：

| 入口 | 用途 |
| --- | --- |
| `scripts/network.sh` | 本项目代理 / 官方 HF 域名配置；支持 proxy、direct、inherit 三种模式 |
| `scripts/install_runtime.sh` | 固定版本环境安装、CUDA/Torch 检查、decord 标签修复 |
| `scripts/prepare_models.py` | 固定 revision 下载模型并核验官方大小、blob / LFS 哈希 |
| `scripts/http_assets.py` | 官方大文件分段续传与最终 SHA256 校验 |
| `scripts/remote_tar.py` | 按 manifest 索引 tar；通过 HTTP Range 提取指定文件，记录来源及哈希 |
| `scripts/validate_runtime.py` | 独立的 CPU 运行环境验收，避免改写阶段一历史报告 |
| `scripts/validate_smoke_data.py` | 完整测试向量、FAISS、query/video 映射、元数据和 smoke 样本检查 |
| `scripts/run_charades_smoke.py` | 物理 GPU 检查、调用官方入口、显存采样、预测/计分验收 |
| `scripts/smoke_eval.py`、`tools/bin/smoke_python` | 加载官方评测模块，仅在查找缺失视频文件时按需获取官方张量；模型、检索和计分逻辑沿用官方实现 |

后续工作范围：

1. 全量 Charades 推理仍需覆盖剩余 1326 个测试视频张量，可按现有索引按需取得或预取；正式输出目录应独立于 smoke。
2. 全量前明确 13 组同文 query 的 qid 映射口径、完整题数和计分分母，保留官方测试题。
3. 完整训练还需要 train 数据、2B 基座或 Stage1 checkpoint，并对齐论文的 SFT 2000 步及 GRPO 参数。此次没有准备全部训练数据，也没有启动训练。

日志保留了首次 vLLM UUID 兼容性失败，以及成功退出时官方 NCCL 未显式 destroy 的提示；退出后的设备与进程检查确认资源已释放。早期 `01_*`、`02_*` 文件继续作为当时的审计快照。

2026-10-26 补记：本地 Stage2 的配置与实际张量形状匹配官方 Qwen3-VL-2B 结构，但发布模型卡的 `base_model` 字段标为 4B。这是尚待核实的发布元数据矛盾；模型结构核验和 smoke 成功不能证明完整训练来源或表 2 分数已经复现。通俗说明见 [模型与数据基础说明](04_MODEL_AND_DATA_GUIDE.md)，配置对照证据见 [模型身份核验](04_MODEL_IDENTITY_CHECK.json)。
