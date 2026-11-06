# 回归测试

现有 **17 个测试模块、76 项用例**都是后续仍有用途的回归检查，当前未发现可直接删除的临时测试脚本。

| 分类 | 内容 | 模块数 |
| --- | --- | ---: |
| [common/](common/) | 原视频解码/取帧、耗时估计与 GPU 释放判断（模拟） | 2 |
| [reproduction/charades/](reproduction/charades/) | VR/VER/VCMR 取轮和分母 | 1 |
| [reproduction/didemo/](reproduction/didemo/) | 计分、下载重试、耗时统计 | 3 |
| [reproduction/activitynet/](reproduction/activitynet/) | 计分与无效回答、采样/修复视图、来源与方向、tar 分片、耗时 | 8 |
| [optimization/charades/](optimization/charades/) | processor 像素往返重建、时间标签、全量配对统计 | 3 |

这些测试主要使用小型合成数据和模拟调用；它们不代表完整数据集通过，更不代替真实 smoke/全量评测。涉及协议的测试会读取本地少量标注或已保存资格文件，需要项目环境和相关资产。

```bash
source common/env.sh
python -B tests/run_tests.py
python -B tests/run_tests.py --group optimization
```

分组入口目前链接到根 `tests/test_*.py` 的同一份已验收源码，保留其哈希与路径。根平铺文件在 VS Code 中隐藏；没有复制第二套测试。新增测试直接写在分类目录，统一 runner 会递归收集一次。旧 `unittest discover -s tests` 仅兼容旧平铺集，不作为后续统一入口。
