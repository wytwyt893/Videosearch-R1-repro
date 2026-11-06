# videosearch_r1 × CHARADES定位失败案例分析-11.05

本轮完成第一步：固定选取 5 个定位失败案例，加 1 个成功对照，重建当前流水线的视觉输入并逐帧检查。全程 CPU，无模型权重加载、前向推理或 GPU 任务；原始预测、标签、模型与配置保持不变。

先打开[交互式逐帧浏览器](../outputs/charades_localization_cases_20261105/case_browser.html)，可切换案例、拖动帧、跳到 GT/本地/作者区间起止点、播放或展开全部帧。HTML 自包含，不需网络；浏览器直接打开本地文件即可。
![六案例总览](../outputs/charades_localization_cases_20261105/case_overview.png)

## 输入重建与证据边界

使用固定官方提交及全量运行绑定的 NPY、metadata、processor 配置，在 CPU 复现 `_build_generation_inputs` 与实际 HF 视频处理。将 `pixel_values_videos` 的空间/时间 patch 排列还原，逆归一化成显示用 RGB；回打包张量完全一致，独立视频 placeholder 路径与完整 prompt 路径的视觉张量也完全一致。图像仅作展示缩放，没有增强、裁剪或生成内容。
历史 GPU 运行没有保存视觉中间张量，因此这是冻结流水线的 CPU 重建，不冒称历史 GPU 截图或作者像素输入。HF 和 vLLM 的时间标签计算一致；没有再次抽帧、没有输入截断。两帧共用一个时间组标签不等于两帧被平均成一张图，两个原始帧仍在 temporal patch 中。

## 选样与病例表

前三例沿用此前已固定的极端差异案例；第四例按 GT 超出 annotation duration 且配对 IoU 下降最大选取；第五例按本地 IoU 最接近但低于 0.69 选取，排除纯浮点阈值；成功对照取共同达到 0.7、GT 持续 3.5–10 秒且不越界的最小 index。规则在看图前保存于 selection.json。五个失败例都在首轮接受 GT，没有进入 SQR；这是有目的的案例诊断，不是无偏抽样。

| index / 视频 | GT（秒） | 本地（秒） | 作者（秒） | 本地 / 作者 IoU | 帧数 / 视觉 token |
| --- | --- | --- | --- | --- | --- |
| [1252 / BEAAA](../outputs/charades_localization_cases_20261105/case_001252_BEAAA/contact_sheet.png) | 0–3.5 | 0–27 | 0–3 | 0.130 / 0.857 | 28 / 2660 |
| [870 / 0Z1PC](../outputs/charades_localization_cases_20261105/case_000870_0Z1PC/contact_sheet.png) | 25–32.5 | 2.5–13 | 26–34.1 | 0.000 / 0.714 | 38 / 2052 |
| [2233 / V6H2O](../outputs/charades_localization_cases_20261105/case_002233_V6H2O/contact_sheet.png) | 8.1–14.5 | 1.6–13 | 7–14.6 | 0.380 / 0.842 | 36 / 3420 |
| [2344 / 1QP1S](../outputs/charades_localization_cases_20261105/case_002344_1QP1S/contact_sheet.png) | 21.9–35 | 17–29 | 18–33.93 | 0.394 / 0.708 | 34 / 3230 |
| [1516 / TDAY1](../outputs/charades_localization_cases_20261105/case_001516_TDAY1/contact_sheet.png) | 23.4–35 | 25–33 | 26–34.37 | 0.690 / 0.722 | 34 / 1836 |
| [26 / 7IPW7](../outputs/charades_localization_cases_20261105/case_000026_7IPW7/contact_sheet.png) | 0–6.3 | 0–5 | 0–5 | 0.794 / 0.794 | 28 / 2660 |

## 案例 1252：BEAAA

原始查询：A person in black jacket and pants sits in a chair, eating a sandwich.

- 0–1 秒附近可见食物靠近嘴；约 3–20 秒间有脱、整理或穿外套的不同阶段；23–27 秒又出现进食。
- 本地 0–27 秒把前后相似动作及中间的外套操作合并；GT 只标 0–3.5 秒的早期实例。

**待验证解释：**目标动作实例没有与持续坐姿、人物外观和其他动作阶段分开；重复实例也带来单一 GT 的选择歧义。

**观察限制：**稀疏帧不能确定每一口的精确起止；不据此修改 GT，也不声称后来进食必定满足全部细粒度描述。

GT 区间内有 4 个源时刻对应的输入帧，涉及 2 个时间组；总输入 3044 tokens，未达到 20384 上限。
仅将已保存输出秒数按 metadata 映射到源时钟，再按原 duration 裁剪，诊断 IoU 为 0.1248，原值 0.1296。这是输出坐标诊断，没有重跑模型，不能代替输入时间戳单变量实验或当作新成绩。

[完整帧图](../outputs/charades_localization_cases_20261105/case_001252_BEAAA/contact_sheet.png) · [PDF](../outputs/charades_localization_cases_20261105/case_001252_BEAAA/contact_sheet.pdf) · [帧与时间标签 CSV](../outputs/charades_localization_cases_20261105/case_001252_BEAAA/frames.csv)

## 案例 870：0Z1PC

原始查询：The person, dressed in a striped shirt and plaid shorts, closes the closet doors.

- 约 3–12 秒主要是走近、站在柜前；约 12–14 秒柜门从关闭变成打开，之后整理物品。
- 约 26–32 秒可见柜门逐渐合上；GT 25–32.5 秒与这一阶段相符，本地 2.5–13 秒落在前面的接近/打开阶段。

**待验证解释：**人物和物体匹配较好，但相反状态变化（开门/关门）及其先后关系被混淆。

**观察限制：**这是大幅错位案例，不能代表所有 79 条差异；约 1 秒时间轴修正无法把前段移动到后段。

GT 区间内有 7 个源时刻对应的输入帧，涉及 4 个时间组；总输入 2483 tokens，未达到 20384 上限。
仅将已保存输出秒数按 metadata 映射到源时钟，再按原 duration 裁剪，诊断 IoU 为 0.0000，原值 0.0000。这是输出坐标诊断，没有重跑模型，不能代替输入时间戳单变量实验或当作新成绩。

[完整帧图](../outputs/charades_localization_cases_20261105/case_000870_0Z1PC/contact_sheet.png) · [PDF](../outputs/charades_localization_cases_20261105/case_000870_0Z1PC/contact_sheet.pdf) · [帧与时间标签 CSV](../outputs/charades_localization_cases_20261105/case_000870_0Z1PC/frames.csv)

## 案例 2233：V6H2O

原始查询：A person is shown drinking from a glass in an indoor setting, with a black car parked nearby.

- 约 3 秒人物走入；8–9 秒有持杯/看手中物体的阶段；10.5、12.6 秒可见杯子靠嘴，31.6 秒附近又有类似饮水。
- 本地起点 1.6 秒包含大量走近、持物的前导片段；GT 为 8.1–14.5 秒。

**待验证解释：**把持杯、人物出现或准备阶段提前算入饮水；重复动作的实例选择需要更明确证据。

**观察限制：**GT 的确切边界仍比稀疏视觉动作更宽，不能把人工看到的举杯帧直接改成新 GT。

GT 区间内有 6 个源时刻对应的输入帧，涉及 3 个时间组；总输入 3844 tokens，未达到 20384 上限。
仅将已保存输出秒数按 metadata 映射到源时钟，再按原 duration 裁剪，诊断 IoU 为 0.4351，原值 0.3798。这是输出坐标诊断，没有重跑模型，不能代替输入时间戳单变量实验或当作新成绩。

[完整帧图](../outputs/charades_localization_cases_20261105/case_002233_V6H2O/contact_sheet.png) · [PDF](../outputs/charades_localization_cases_20261105/case_002233_V6H2O/contact_sheet.pdf) · [帧与时间标签 CSV](../outputs/charades_localization_cases_20261105/case_002233_V6H2O/frames.csv)

## 案例 2344：1QP1S

原始查询：Two men are present in the image, one laughing at a picture and the other looking on.

- 早段以站立者的动作更明显；约 20–22 秒后，两人围绕图片、转头与互动的画面更明显。
- 本地 17–29 秒提前结束；GT 21.9–35 秒，annotation duration 为 33.93 秒，源视频最后采样帧约 33.64 秒。

**待验证解释：**多主体关系与细微表情的时段判断较难，同时存在标注末端超出可观测视频的问题。

**观察限制：**低位/遮挡的人体可能在更早帧已出现；不能断言早段完全没有第二个人，也无法仅凭 1 FPS 帧精确定义开始发笑的时刻。

GT 区间内有 13 个源时刻对应的输入帧，涉及 7 个时间组；总输入 3644 tokens，未达到 20384 上限。
仅将已保存输出秒数按 metadata 映射到源时钟，再按原 duration 裁剪，诊断 IoU 为 0.5003，原值 0.3944。这是输出坐标诊断，没有重跑模型，不能代替输入时间戳单变量实验或当作新成绩。

[完整帧图](../outputs/charades_localization_cases_20261105/case_002344_1QP1S/contact_sheet.png) · [PDF](../outputs/charades_localization_cases_20261105/case_002344_1QP1S/contact_sheet.pdf) · [帧与时间标签 CSV](../outputs/charades_localization_cases_20261105/case_002344_1QP1S/frames.csv)

## 案例 1516：TDAY1

原始查询：A young boy with dark hair, dressed in a striped shirt and shorts, stands up from a bench.

- 人物前段持续倚躺；约 26–30 秒可见调整腿部、转身并起身，31 秒后已站立。
- 本地 25–33 秒与作者 26–34.37 秒都覆盖该事件；GT 23.4–35 秒包含更早准备阶段并延伸到声明时长之后。

**待验证解释：**更像小幅边界、名义时钟和标注容差问题，而不是完全看错动作；适合单独测试时间轴。

**观察限制：**本地 IoU 0.6897 接近 0.7，但并非浮点舍入边界；不能仅凭跨阈值就声称语义明显变好。

GT 区间内有 11 个源时刻对应的输入帧，涉及 6 个时间组；总输入 2252 tokens，未达到 20384 上限。
仅将已保存输出秒数按 metadata 映射到源时钟，再按原 duration 裁剪，诊断 IoU 为 0.7184，原值 0.6897。这是输出坐标诊断，没有重跑模型，不能代替输入时间戳单变量实验或当作新成绩。

[完整帧图](../outputs/charades_localization_cases_20261105/case_001516_TDAY1/contact_sheet.png) · [PDF](../outputs/charades_localization_cases_20261105/case_001516_TDAY1/contact_sheet.pdf) · [帧与时间标签 CSV](../outputs/charades_localization_cases_20261105/case_001516_TDAY1/frames.csv)

## 案例 26：7IPW7

原始查询：a person walks through a doorway, dressed in a purple sweater and black pants with a floral pattern, holding a red cup.

- 人物前几秒穿过门口，随后大部分时间站在门边持杯。
- 本地和作者均给出 0–5 秒，GT 为 0–6.3 秒；没有把后续持续站立、持杯全部算入穿门动作。

**待验证解释：**同一处理流水线能够区分明显的动作阶段，失败不是所有视频都被统一输出为整段。

**观察限制：**这是按规则选出的一个成功对照，不是总体准确率的估计。

GT 区间内有 7 个源时刻对应的输入帧，涉及 4 个时间组；总输入 3053 tokens，未达到 20384 上限。
仅将已保存输出秒数按 metadata 映射到源时钟，再按原 duration 裁剪，诊断 IoU 为 0.8307，原值 0.7937。这是输出坐标诊断，没有重跑模型，不能代替输入时间戳单变量实验或当作新成绩。

[完整帧图](../outputs/charades_localization_cases_20261105/case_000026_7IPW7/contact_sheet.png) · [PDF](../outputs/charades_localization_cases_20261105/case_000026_7IPW7/contact_sheet.pdf) · [帧与时间标签 CSV](../outputs/charades_localization_cases_20261105/case_000026_7IPW7/frames.csv)

## 共同问题与总体范围

案例提示三个可分离的问题：
场景/人物匹配后没有准确切分目标动作实例；
相反动作或准备阶段与目标动作混淆；
小幅边界差、稀疏采样及标注时长问题。

没有发现这六例因视频后半段缺失、再次抽帧或上下文截断而丢失 GT。没有证据证明单独增加视觉 token 就能解决这些错误。

- 为避免被极端案例误导，另统计全部 79 条“双方接受 GT、作者命中 0.7、本地未中”的定位损失：
66 条首轮结束，只有 1 条完全无重叠；预测长度 >1.5 倍 GT 的有 7 条，<2/3 的有 5 条；起点/终点误差中位数分别为 -1.7/-2.2 秒。总体上多数不是完全认错时段，小幅偏早和边界精度也值得优先验证。该子集按结果选出，不能作总体因果估计。

## 后续验证顺序

1. 时间戳唯一变量对照：同一像素 NPY、模型、视觉预算、后端、T/N、完整候选库与主计分，只切换名义时间/源时间标签。使用预先固定的评测范围，不按逐题 GT 调整补丁。
2. 在上述结果基础上，再单独研究动作证据约束、实例分割或局部边界复核，尤其区分开/关、拿着/饮用、准备/动作本体；先验证单项，再组合。
3. 局部更密集采样需要更密集的真实源视频帧；重复或插值现有 NPY 不会增加新动作证据。先确认资产可用性，再与增加 token/更换后端分开比较。

[机器可读案例与观察](../reports/39_charades_localization_cases.json) · [六例总览 PDF](../outputs/charades_localization_cases_20261105/case_overview.pdf) · [重建脚本](../scripts/inspect_charades_localization_cases.py) · [观察记录与汇总脚本](../scripts/review_charades_localization_cases.py)

本轮未修改主实验分数，没有启动时间戳对照或组合优化的 GPU 推理。可视化资产保留在本机 outputs/，不自动纳入 Git。
