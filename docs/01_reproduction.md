# 01 · 三个 benchmark 的发布权重复现

完成 Charades-FIG、DiDeMo-FIG、ActivityNet-FIG 共 **24699 题**，对应论文 Table 2 的 VideoSearch-R1 行。完整 SFT/GRPO 训练和其他 baseline 尚未复现。

| Benchmark | 论文 VCMR@0.3 / 0.5 / 0.7 R@1 | 本地复现基线 | 脚本与产物 |
| --- | --- | --- | --- |
| Charades-FIG | 16.5 / 13.4 / 8.2 | 16.40 / 12.47 / 6.91 | [charades/](../experiments/01_reproduction/charades/README.md) |
| DiDeMo-FIG | 33.3 / 30.2 / 19.7 | 31.77 / 28.81 / 19.06 | [didemo/](../experiments/01_reproduction/didemo/README.md) |
| ActivityNet-FIG | 33.8 / 22.3 / 12.3 | 35.76 / 23.56 / 12.98 | [activitynet/](../experiments/01_reproduction/activitynet/README.md) |

百分比。Charades/DiDeMo 基线采用官方像素与名义时间标签；ActivityNet 为 364 个修复视频和源时间标签版本，不能将其分数差异全部归因于修复。Charades 后续时间戳优化单列在阶段 03，不静默替换这张复现基线表。

完整 VR/VER/VCMR 分组、百分点差值及分母见 [三数据集总表](../experiments/01_reproduction/summary/RESULTS.md) 与 [历史详细总表](<archive/videosearch_r1 × 三数据集全量评估汇总-11.05.md>)。

共同设置为实际 2B Stage2 VLM、2B 检索器、T=2、N=8、BF16/vLLM、单卡。主 VER 配对最后实际验证的视频；VR 使用末轮更新后的排名；VCMR 使用最后实际验证/定位的视频。论文 VER 取轮仍未正式确认，首轮 VER 仅作补充。

ActivityNet 有 1 条无效末次回答：主 VER 为 14086/16965，全部题计为 83.0249%；VR/VCMR 分母均为完整 16966 题。CPU 收尾恢复未重跑模型或删除这条回答。

[返回文档目录](README.md)
