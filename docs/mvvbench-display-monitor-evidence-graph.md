# MVVBench 显示器案例：查询相关局部证据图

## 目的

以 `63am_lobby_3-q01190` 的显示器开关状态差值问题为单题诊断，拆分检查：单视角观察、直接多视角问答、端侧实体/状态提取、跨视角匹配与去重、最终计数和计算。这个单案例用于定位错误，不作为方法有效性的总体证据。

端侧不接收候选选项，只针对问题抽取本视角证据；中心端收到各端 JSON 证据图后融合。局部实体记录包含显示器类别、外观、画面位置、屏幕状态、可见性、帧来源和简短依据。屏幕状态允许 `uncertain`；黑色屏幕不直接等于关机。中心端匹配状态为 `same`、`different` 或 `uncertain`，只有 `same` 才去重。

## 首轮条件

- `single_view_<id>`：单视角图像直接回答原题。
- `direct_all_views`：全部视角图像直接回答原题。
- `local_graph_view_<id>`：单视角端侧证据图，不提供候选选项。
- `candidate_entity_pairs.json`：从两视角局部图生成所有跨视角候选屏幕配对。
- `cross_view_entity_matching_text`：中心端只看文字证据图，对每个候选对逐一判断 same/different/uncertain。
- `cross_view_entity_matching_visual`：中心端看文字证据图和每个 view 的一张原始帧，对同一批候选对逐一判断。各 view 的 local ID 是局部编号，编号相同不构成匹配证据；漏答配对按 uncertain 处理。
- `graph_count_no_dedup`：脚本将所有 view 内实体分别保留，再确定性汇总状态和差值，用作不去重对照。
- `graph_count_dedup_text`、`graph_count_dedup_visual`：脚本分别按文字匹配、视觉匹配的 `same` 候选合并，再确定性汇总。状态冲突或不确定实体保留在未决计数中，不强制选答案。

所有条件使用相同标注时间窗和每视角相同帧数。该脚本默认每视角 2 帧，基线 QA 只要求输出选项字母、上限 64 tokens；视觉匹配条件额外向中心端提供每视角第一张原始帧。零长度时间戳扩展为中心 2 秒窗口；这属于 timestamp-oracle 诊断。输出保留原始响应；若生成触及 token 上限，会标记为 `truncated`。运行可用相同参数加 `--resume` 续跑。

## 服务器运行

在仓库根目录、激活 `vlm_redundancy` 环境后运行：

```bash
python scripts/diagnose_mvvbench_evidence_graph.py \
  --config configs/exp002.local.json \
  --questions artifacts/EXP-002/score_run07/evaluation_questions.json \
  --output artifacts/EXP-002/display_monitor_graph_run04 \
  --query-id 63am_lobby_3-q01190 \
  --frames-per-view 2
```

若运行中断：

```bash
python scripts/diagnose_mvvbench_evidence_graph.py \
  --config configs/exp002.local.json \
  --questions artifacts/EXP-002/score_run07/evaluation_questions.json \
  --output artifacts/EXP-002/display_monitor_graph_run04 \
  --query-id 63am_lobby_3-q01190 \
  --frames-per-view 2 \
  --resume
```

若该题的评测问题路径或 query ID 与服务器实际文件不同，替换对应参数。脚本只读取配置中的标注、映射和视频路径，不修改源数据。

## 输出

- `sampled_frames.json`：输入帧来源及时间戳。
- `frames/`：抽取后的图片。
- `calls/<step>/prompt.txt`、`raw_response.txt`、`metrics.json`：各次调用的提示词、原始输出和模型指标。
- `predictions.json`：所有模型调用和脚本计算条件的结构化结果、解析状态及逐条件正确性。
- `evidence_graphs.json`：中心实体匹配实际使用的各视角局部图。
- `candidate_entity_pairs.json`：送给两种匹配条件的相同候选对清单。
- `cross_view_matches_text.json`、`cross_view_matches_visual.json`：逐候选配对的判断、依据和漏答列表；只有 same 才参与对应条件的去重。
- `manifest.json`、`run.json`：设置签名、运行状态和条件摘要。

## 人工分析顺序

1. 看 `local_graph_view_*`：是否漏掉笔记本屏幕、重复识别同一屏幕，或把黑屏错误标成关机。
2. 对比两份 `cross_view_matches_*.json`：图像 grounding 是否带来更具体的匹配依据，所有候选配对是否得到判断，是否错误地把相同 local ID 当作证据。
3. 对比 `graph_count_no_dedup`、`graph_count_dedup_text` 和 `graph_count_dedup_visual`：判断视觉 grounding 是否改变实体去重与最终计数。
4. 看计数、差值和答案是否与选项一致；最后对照单视角与直接全视角基线。

VLM 的自报置信度不是校准概率。模型输出的实体和匹配仍需人工核查；单题的正确性不能说明跨题泛化效果。

