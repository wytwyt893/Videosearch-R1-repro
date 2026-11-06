# 00_data_repair · ActivityNet-FIG 数据审计与原视频修复

确认 ActivityNet 发布视频张量是否覆盖全片；对异常视频寻找原视频来源、检查内容对应关系、修复采样，并形成有独立来源记录的数据视图。

**当前结论：364 个疑似异常视频完成来源核验和采样修复；原版资产与来源证据保留。**

先读 [RESULTS.md](<RESULTS.md>) 查看可汇报的设置、结果和限制。以下编号用于理解步骤；模块文件、历史补救步骤并不要求全部依次执行。

## 脚本入口

| 编号入口 | 职责 | 原实现 |
| --- | --- | --- |
| [00_prepare_activitynet_assets.py](<../../../../scripts/prepare_activitynet_assets.py>) | 下载 ActivityNet 模型及完整测试资产 | [prepare_activitynet_assets.py](<../../../../scripts/prepare_activitynet_assets.py>) |
| [01_fetch_activitynet_metadata.py](<../../../../scripts/fetch_activitynet_metadata.py>) | 取得原始元数据，为时长与标签核查提供来源 | [fetch_activitynet_metadata.py](<../../../../scripts/fetch_activitynet_metadata.py>) |
| [02_activitynet_eda.py](<../../../../scripts/activitynet_eda.py>) | 统计题数、视频时长、区间与数据形状 | [activitynet_eda.py](<../../../../scripts/activitynet_eda.py>) |
| [03_check_activitynet_alignment.py](<../../../../scripts/check_activitynet_alignment.py>) | 核对 ActivityNet 原版入口与参数 | [check_activitynet_alignment.py](<../../../../scripts/check_activitynet_alignment.py>) |
| [04_preflight_activitynet_full.py](<../../../../scripts/preflight_activitynet_full.py>) | 原版完整测试资产的 CPU 预检 | [preflight_activitynet_full.py](<../../../../scripts/preflight_activitynet_full.py>) |
| [05_audit_activitynet_sampling.py](<../../../../scripts/audit_activitynet_sampling.py>) | 检查发布 NPY 的帧序列及元数据覆盖范围 | [audit_activitynet_sampling.py](<../../../../scripts/audit_activitynet_sampling.py>) |
| [06_fetch_activitynet_raw_sample.py](<../../../../scripts/fetch_activitynet_raw_sample.py>) | 取得一个可核查的原视频示例 | [fetch_activitynet_raw_sample.py](<../../../../scripts/fetch_activitynet_raw_sample.py>) |
| [07_compare_activitynet_raw_npy.py](<../../../../scripts/compare_activitynet_raw_npy.py>) | 比较原视频解码帧与发布张量的对应关系 | [compare_activitynet_raw_npy.py](<../../../../scripts/compare_activitynet_raw_npy.py>) |
| [08_index_activitynet_raw_mirror.py](<../../../../scripts/index_activitynet_raw_mirror.py>) | 索引原视频镜像并定位待修复视频 | [index_activitynet_raw_mirror.py](<../../../../scripts/index_activitynet_raw_mirror.py>) |
| [09_download_activitynet_repair_sources.py](<../../../../scripts/download_activitynet_repair_sources.py>) | 下载待修复视频的候选原视频来源 | [download_activitynet_repair_sources.py](<../../../../scripts/download_activitynet_repair_sources.py>) |
| [10_retry_activitynet_source_downloads.py](<../../../../scripts/retry_activitynet_source_downloads.py>) | 对未完成来源下载做有记录的续试 | [retry_activitynet_source_downloads.py](<../../../../scripts/retry_activitynet_source_downloads.py>) |
| [11_fetch_activitynet_repair_alternatives.py](<../../../../scripts/fetch_activitynet_repair_alternatives.py>) | 查找并取得替代视频来源 | [fetch_activitynet_repair_alternatives.py](<../../../../scripts/fetch_activitynet_repair_alternatives.py>) |
| [12_review_activitynet_source_alignment.py](<../../../../scripts/review_activitynet_source_alignment.py>) | 复核来源旋转、时间与帧序号对应，输出诊断 | [review_activitynet_source_alignment.py](<../../../../scripts/review_activitynet_source_alignment.py>) |
| [13_align_activitynet_repair_sources.py](<../../../../scripts/align_activitynet_repair_sources.py>) | 对候选来源做画面/方向对齐并记录证据 | [align_activitynet_repair_sources.py](<../../../../scripts/align_activitynet_repair_sources.py>) |
| [14_qualify_activitynet_repair.py](<../../../../scripts/qualify_activitynet_repair.py>) | 检查来源时长与内容对应，验收修复候选 | [qualify_activitynet_repair.py](<../../../../scripts/qualify_activitynet_repair.py>) |
| [15_extract_video_frames_verified.py](<../../../../scripts/extract_video_frames_verified.py>) | 可靠解码计数并导出覆盖全片的帧与索引 | [extract_video_frames_verified.py](<../../../../scripts/extract_video_frames_verified.py>) |
| [16_build_activitynet_repair_views.py](<../../../../scripts/build_activitynet_repair_views.py>) | 建立原版、修复像素及源时间轴数据视图 | [build_activitynet_repair_views.py](<../../../../scripts/build_activitynet_repair_views.py>) |
| [17_check_activitynet_repair_integrity.py](<../../../../scripts/check_activitynet_repair_integrity.py>) | 对修复视图、哈希和文件映射做完整性检查 | [check_activitynet_repair_integrity.py](<../../../../scripts/check_activitynet_repair_integrity.py>) |
| [18_validate_activitynet_repair_cpu.py](<../../../../scripts/validate_activitynet_repair_cpu.py>) | 修复版进入 GPU 前的 CPU 资格检查 | [validate_activitynet_repair_cpu.py](<../../../../scripts/validate_activitynet_repair_cpu.py>) |
| [19_activitynet_repair_time.py](<../../../../scripts/activitynet_repair_time.py>) | 推理适配模块：透传修复版源采样时间轴 | [activitynet_repair_time.py](<../../../../scripts/activitynet_repair_time.py>) |

历史编号入口为相对符号链接，编辑它会编辑同一份 `scripts/` 实现。保留实现原路径与字节，是为了使历史哈希、模块导入及 CPU 资格检查继续有效。后续新实验在相应阶段、数据集下创建独立方法目录，直接编写编号脚本。

## 本实验产物

| 目录 / 文件 | 内容 |
| --- | --- |
| [RESULTS.md](<RESULTS.md>) | 汇报用结果与解释范围 |
| [reports/](<reports>) | 轻量 JSON、验收、哈希与来源；可提交 Git |
| [outputs/](<outputs>) | 原预测、逐题分析、帧图和运行内日志；仅本机 |
| [logs/](<logs>) | 准备、tmux 启动及补救日志；仅本机 |

### 轻量记录

- [20_activitynet_download_plan.json](<reports/20_activitynet_download_plan.json>)
- [21_activitynet_eda.json](<reports/21_activitynet_eda.json>)
- [22_activitynet_assets.json](<reports/22_activitynet_assets.json>)
- [23_activitynet_alignment.json](<reports/23_activitynet_alignment.json>)
- [24_activitynet_cpu_preflight.json](<reports/24_activitynet_cpu_preflight.json>)
- [27_activitynet_time_metadata_audit.json](<reports/27_activitynet_time_metadata_audit.json>)
- [28_activitynet_sampling_coverage.json](<reports/28_activitynet_sampling_coverage.json>)
- [29_activitynet_sampling_fix_validation.json](<reports/29_activitynet_sampling_fix_validation.json>)
- [30_activitynet_raw_correspondence.json](<reports/30_activitynet_raw_correspondence.json>)
- [31_activitynet_repair_cohort.json](<reports/31_activitynet_repair_cohort.json>)
- [31a_activitynet_source_visual_review.json](<reports/31a_activitynet_source_visual_review.json>)
- [32_activitynet_repair_cpu_preflight.json](<reports/32_activitynet_repair_cpu_preflight.json>)
- [33_activitynet_repair_summary.json](<reports/33_activitynet_repair_summary.json>)
- [34_activitynet_repair_integrity.json](<reports/34_activitynet_repair_integrity.json>)
- [34a_activitynet_label_observations.json](<reports/34a_activitynet_label_observations.json>)

### 运行输出

- [activitynet_cpu_preflight_20261029_143914_585464/](<outputs/activitynet_cpu_preflight_20261029_143914_585464>)
- [activitynet_eda_20261028/](<outputs/activitynet_eda_20261028>)
- [activitynet_raw_correspondence_20261029/](<outputs/activitynet_raw_correspondence_20261029>)
- [activitynet_repair_integrity_20261030_002802_860745/](<outputs/activitynet_repair_integrity_20261030_002802_860745>)
- [activitynet_repair_integrity_20261030_002834_327578/](<outputs/activitynet_repair_integrity_20261030_002834_327578>)
- [activitynet_repair_source_review_v1/](<outputs/activitynet_repair_source_review_v1>)
- [activitynet_repair_visual_review_v1/](<outputs/activitynet_repair_visual_review_v1>)
- [activitynet_sampling_audit_20261029/](<outputs/activitynet_sampling_audit_20261029>)

## 使用约定

从项目根目录先执行 `source common/env.sh`。编号 Python 入口可直接以 `python -B experiments/01_reproduction/activitynet/00_data_repair/文件名.py` 调用；具体参数先查脚本帮助和原文档。

历史 runner 通常拒绝覆盖已完成报告；原实验目录是证据，不是新一轮实验的空目录。新结果使用新的实验/运行目录，完成后同步记录 Markdown。

[返回实验总目录](<../../../README.md>)
