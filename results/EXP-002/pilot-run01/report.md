# EXP-002 试运行：8 个问题的预算扫描

状态：完成（流程试运行；不作为正式效果结论）  
运行编号：`score_run07` + `qa_run02`  
代码提交：服务器实际执行版本未记录  
数据与配置：MVVBench，EgoExo 与 MMPTrack；服务器本机配置未归档  
原始结果：服务器 `artifacts/EXP-002/score_run07/` 与 `artifacts/EXP-002/qa_run02/`，未纳入 Git

## 目的

确认“候选证据打分 → 预算计划 → 多选 QA → 汇总分析”流程可在服务器运行，并初步观察不同传输选择策略的成本与准确率。样本仅 8 题，因此结果用于排查流程和形成下一轮假设，不支持泛化结论。

## 样本与运行

共 8 题：EgoExo 4 题，MMPTrack 4 题。题目为：

- EgoExo：`sfu_basketball_05_35-q00000`、`sfu_basketball_06_24-q00004`、`sfu_basketball_10_31-q00023`、`sfu_cooking023_8-q00036`
- MMPTrack：`63am_industry_safety_2-q01180`、`63am_lobby_1-q01186`、`63am_lobby_3-q01190`、`63am_office_2-q01196`

比较策略：`full_input`、`uniform_temporal_sampling`、`fixed_view_sampling`、`query_driven_selection`。预算比例为 0.1、0.2、0.3、0.4、0.6、0.8、1.0。成本按候选 JPEG 帧文件字节数计算，是离线传输代理值，不是真实网络传输测量。

模型为 Qwen3-VL-8B-Instruct；服务器记录的运行环境为 PyTorch 2.9.1+cu128、Transformers 4.57.6、NVIDIA RTX 5090。精确模型修订、随机种子、运行时间和服务器代码提交没有留存。

## 汇总结果

准确率分母为 8 题。`full_input` 不受预算上限约束，作为全量输入基准；其成本在每个预算行都为 432,265 B。

| 预算比例 | 策略 | 准确率 | 平均上传字节 |
|---:|---|---:|---:|
| 0.1 | uniform / fixed / query-driven | 不可行（0/8 可评估） | 0 B |
| 0.1–1.0 | full_input | 0.375 | 432,265 B |
| 0.2 | uniform_temporal_sampling | 0.375 | 51,602 B |
| 0.2 | fixed_view_sampling | 0.375 | 51,602 B |
| 0.2 | query_driven_selection | 0.375 | 51,278 B |
| 0.3 | uniform_temporal_sampling | 0.500 | 107,362 B |
| 0.3 | fixed_view_sampling | 0.125 | 104,424 B |
| 0.3 | query_driven_selection | 0.500 | 105,133 B |
| 0.4 | uniform_temporal_sampling | 0.500 | 160,183 B |
| 0.4 | fixed_view_sampling | 0.250 | 156,665 B |
| 0.4 | query_driven_selection | 0.500 | 156,854 B |
| 0.6 | uniform_temporal_sampling | 0.500 | 216,115 B |
| 0.6 | fixed_view_sampling | 0.250 | 208,850 B |
| 0.6 | query_driven_selection | 0.500 | 225,626 B |
| 0.8 | uniform_temporal_sampling | 0.375 | 324,014 B |
| 0.8 | fixed_view_sampling | 0.250 | 208,850 B |
| 0.8 | query_driven_selection | 0.375 | 322,233 B |
| 1.0 | uniform_temporal_sampling | 0.375 | 432,265 B |
| 1.0 | fixed_view_sampling | 0.250 | 208,850 B |
| 1.0 | query_driven_selection | 0.375 | 432,265 B |

## 初步观察

- 10% 预算下三种受限策略都不可行，说明当前最小传输单元已经超过该预算。后续正式评测应将 20% 作为首个可行扫描点，或调整候选单元粒度后重新定义成本版本。
- 20%–60% 区间内，查询驱动策略没有稳定优于均匀采样。30% 和 40% 两者准确率相同，传输字节也接近；60% 时查询驱动比均匀采样多传约 9.5 KB，准确率相同。
- 全量输入准确率为 0.375；部分较低预算结果达到 0.5。小样本波动、问题分布以及上下文变化都可能造成这种现象，不能据此认为压缩提升了模型能力。
- 分域结果显示 EgoExo 四题各策略准确率均为 0.25；MMPTrack 结果波动较大，是总体差异的重要来源。每域只有四题，不能据此比较数据集难度或方法优劣。

## 限制与下一步

这次只覆盖 8 个问题，且策略的选择器仍是启发式基线。记录中缺少服务器实际代码提交、配置快照、完整逐题预测文件和精确运行时间；分析结果可复现性有限。下一轮计划扩大到盘点出的 EgoExo 88 题和 MMPTrack 16 题，保持场景级去重，并排除当前未纳入实验的 Panoptic 问题。QA 运行若被四小时作业时限中断，可使用新版 `run_video_qa.py --resume` 从 `predictions.partial.json` 继续；服务器需要先拉取包含该参数的代码版本。

大规模结果完成后，应重新导出并归档实际配置、逐题结果、汇总指标和代码提交号，再更新本报告或建立新的运行目录，不覆盖本次试运行记录。
