# videosearch_r1 × 三数据集全量评估汇总-11.05

整理日期：2026-11-05；时间均为北京时间。本轮完成 Charades-FIG、DiDeMo-FIG、ActivityNet-FIG 三个完整测试集，共 **24,699 题**的发布权重推理评测。
对应[论文 Table 2](https://arxiv.org/html/2607.00446v1#S4.T2) 的 **VideoSearch-R1 行**。下表保留原表的 Dataset、Method、VCMR、VER、VR 分组及列顺序；其他 baseline 尚未复现。

百分比：论文保留原表一位小数，本地保留两位小数。本地主 VER 始终为预先确定的末次实际验证；论文多轮取轮未明确确认，VER 数值并列不代表其口径已被证明相同。

<table>
<thead>
<tr><th rowspan="2">Dataset</th><th rowspan="2">Method</th><th colspan="3">VCMR</th><th>VER</th><th colspan="4">VR</th></tr>
<tr><th>0.3/R@1</th><th>0.5/R@1</th><th>0.7/R@1</th><th>Acc</th><th>R@1</th><th>R@5</th><th>R@10</th><th>R@100</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">Charades-FIG</td><td>VideoSearch-R1（论文）</td><td>16.5</td><td>13.4</td><td>8.2</td><td>75.7</td><td>24.6</td><td>42.9</td><td>52.2</td><td>84.7</td></tr>
<tr><td>VideoSearch-R1（本地）</td><td><strong>16.40</strong></td><td><strong>12.47</strong></td><td><strong>6.91</strong></td><td><strong>52.26</strong></td><td><strong>24.78</strong></td><td><strong>42.96</strong></td><td><strong>52.26</strong></td><td><strong>84.60</strong></td></tr>
<tr><td rowspan="2">DiDeMo-FIG</td><td>VideoSearch-R1（论文）</td><td>33.3</td><td>30.2</td><td>19.7</td><td>74.6</td><td>59.0</td><td>82.0</td><td>87.8</td><td>97.5</td></tr>
<tr><td>VideoSearch-R1（本地）</td><td><strong>31.77</strong></td><td><strong>28.81</strong></td><td><strong>19.06</strong></td><td><strong>71.19</strong></td><td><strong>58.96</strong></td><td><strong>82.06</strong></td><td><strong>87.81</strong></td><td><strong>97.51</strong></td></tr>
<tr><td rowspan="2">ActivityNet-FIG</td><td>VideoSearch-R1（论文）</td><td>33.8</td><td>22.3</td><td>12.3</td><td>83.3</td><td>61.1</td><td>81.7</td><td>88.5</td><td>98.4</td></tr>
<tr><td>VideoSearch-R1（本地修复版†）</td><td><strong>35.76</strong></td><td><strong>23.56</strong></td><td><strong>12.98</strong></td><td><strong>83.03</strong></td><td><strong>61.20</strong></td><td><strong>81.68</strong></td><td><strong>88.42</strong></td><td><strong>98.39</strong></td></tr>
</tbody>
</table>

† ActivityNet 本地结果使用修复版：364 个异常视频从公开镜像完整解码后重新采样，全部 4,866 个视频使用源时间轴；其中 364 个使用新解码 PTS，其他 4,502 个使用发布的源采样元数据。Charades、DiDeMo 本次保留官方发布像素与名义时间轴。三个测试集都保留原始题目和完整检索库，但输入修复方式不同，不能把本轮当作三数据集上的统一修复消融。

## VER 取轮、分母与补充结果

| 数据集 | 论文 VER | 本地末次（主指标） | 本地首轮（补充） | 主 VER 正确/有效 | 无效回答 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Charades-FIG | 75.7% | 52.26% | 74.84% | 1944/3720 | 0 |
| DiDeMo-FIG | 74.6% | 71.19% | 74.93% | 2857/4013 | 0 |
| ActivityNet-FIG | 83.3% | 83.03% | 84.83% | 14086/16965 | 1 |

末次 VER 与该轮实际检查的视频配对；第二轮后若重排得到新视频，不把旧判断配给新视频。VR 使用最后更新后的排名，VCMR 使用最后实际检查、定位的视频。保持 T=2、N=8，触发时执行官方末轮查询更新与重排，不增加第三次验证。
ActivityNet 唯一无效末次回答位于原始 index 10237：主 VER 分母为 16,965；若将它按错误计入全部 16,966 题，末次 VER 为 83.0249%。VR、VCMR 分母始终为全部题数。
Charades、DiDeMo 的作者公开预测，其首轮 VER 更接近论文数值；DiDeMo 附件实际为 T=3。已保存的前两轮快照只能作离线诊断，不能代替新的 T=2 受控实验。我们没有按分数接近程度更换主 VER 口径。

## 与论文的差值

以下均为本地未四舍五入数值减论文公布值，单位为百分点；VER 取轮仍有上述限制。

| 数据集 | VCMR 0.3 / 0.5 / 0.7 | 末次 VER | VR R@1 / R@5 / R@10 / R@100 |
| --- | --- | ---: | --- |
| Charades-FIG | -0.10 / -0.93 / -1.29 | -23.44 | +0.18 / +0.06 / +0.06 / -0.10 |
| DiDeMo-FIG | -1.53 / -1.39 / -0.64 | -3.41 | -0.04 / +0.06 / +0.01 / +0.01 |
| ActivityNet-FIG | +1.96 / +1.26 / +0.68 | -0.27 | +0.10 / -0.02 / -0.08 / -0.01 |

三数据集的 VR 各项与论文最大绝对差约 0.18 个百分点。Charades、DiDeMo 的 VCMR 仍低于论文，已有离线对照主要指向正确接受视频后的时间定位差异；ActivityNet 修复版 VCMR 高于论文约 0.68–1.96 个百分点。缺少同配置原版全量对照，不能将其更高分数直接归因于修复。

## 数据、运行与验收

| 数据集 | 问题数 | 完整视频库 | 输入版本 | 单卡推理实测耗时 |
| --- | ---: | ---: | --- | --- |
| Charades-FIG | 3,720 | 1,334 | 官方发布像素与名义时间轴 | 4 小时 33 分 52 秒 |
| DiDeMo-FIG | 4,013 | 1,002 | 官方发布像素与名义时间轴 | 4 小时 36 分 40 秒 |
| ActivityNet-FIG | 16,966 | 4,866 | 364 个修复像素视频；全部视频透传源时间轴 | 27 小时 50 分 5 秒 |

公共设置：各数据集官方发布 Stage2 2B VLM，复用 2B 检索编码器；单卡 A100 80GB、BF16、vLLM、batch=1、TP=1、eager、T=2、8 soft tokens。外层 seed=42，官方 vLLM 构造器未传入 seed，engine 默认 seed=0。保留官方推理视觉处理，不另加 4096 视觉 token 硬上限；论文的 4096 描述属于训练设置。
三个完整测试集均无跳题、无视频元数据缺失，结束后 GPU 释放检查通过。ActivityNet 于 10 月 30 日启动、10 月 31 日完成，文件名保留启动日期 10.30。其推理返回码为 0，原收尾因错误套用 smoke 零无效回答检查而退出；随后仅用 CPU 补齐报告，原预测、错误日志和失败清单均保留。IoU=0.5 的一个 float32 边界以原始 JSONL 与官方离线计分为准。
原始标注边界问题保留，ActivityNet 的两个镜像来源保留显式视觉复核例外。公开镜像编码不声称与作者原文件逐字节一致。完整训练、其他 baseline 和论文其他表格尚未完成。

## 实验报告与机器可读结果

- Charades-FIG：[完整报告](<videosearch_r1 × CHARADES全量评估-10.27.md>)；[结果摘要](../reports/10_charades_full_summary.json)。
- DiDeMo-FIG：[完整报告](<videosearch_r1 × DIDEMO全量评估-10.28.md>)；[结果摘要](../reports/18_didemo_full_summary.json)。
- ActivityNet-FIG：[完整报告](<videosearch_r1 × ACTIVITYNET修复版全量评估-10.30.md>)；[结果摘要](../reports/36b_activitynet_full_summary.json)。
- [Charades 最小离线检查](<videosearch_r1 × CHARADES离线检查-10.27.md>)；[DiDeMo 最小离线检查](<videosearch_r1 × DIDEMO离线检查-10.28.md>)。
- [汇总 JSON](../reports/38_three_dataset_summary.json)、[Table 2 CSV](../reports/38_three_dataset_table2.csv)、[文件重命名映射与原始哈希](../reports/37_document_renames.json)。

## 下一阶段研究起点

1. **先分离数据修复因素。** ActivityNet 已保留原版、仅修复画面、修复画面并透传源时间轴三个视图。固定题目、权重、预算和评估协议，做配对对照，确认帧覆盖与时间轴各自的影响。
2. **聚焦时间定位。** Charades、DiDeMo 的检索接近论文，而正确接受 GT 后的区间预测仍有差异。先检查时间标签与源采样，再研究预算分配或局部二次定位，分别验证各因素。
3. **研究迭代成本与收益。** 已保存首轮/末次轨迹，可先离线分析何时 SQR 改善排名、何时需要继续验证，再设计自适应停止或更新门控，并同时报告准确率与推理成本。

以上是由现有证据引出的候选问题，尚未证明方法新颖性，也没有启动新的 GPU 实验；后续需查阅相关工作并设计受控实验。
