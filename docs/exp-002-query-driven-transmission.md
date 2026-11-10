# EXP-002：带宽预算下的查询驱动视频证据选择

状态：8 题流程试运行已完成；扩展评测尚未完成。试运行结果见 [EXP-002 试运行报告](../results/EXP-002/pilot-run01/report.md)。

## 研究问题

在固定通信预算下，能否根据用户问题选择更有用的摄像头视角与时间片段，使多视角问答质量接近全量输入，同时降低上行传输字节数？

首轮只研究离线传输内容选择，不把网络延迟、丢包或实时性作为结论。MVVBench 的 `views`、`timestamp_start`、`timestamp_end` 和正确答案只用于样本组织、诊断和 oracle 上界；真实选择策略不得读取这些答案侧标注。

## 预算定义

以每道问题的全量视频输入实际字节数为基准，使用预算比例扫描：`0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00`。这些比例是探索网格，不代表某个现实网络的 Mbps。先在开发样本检查传输量—准确率曲线，再据曲线确定后续重点预算点。

当前实现将每段固定时长片段均匀抽取的 JPEG 帧作为传输单元，最长边缩放至统一尺寸并使用固定 JPEG 质量参数，按 JPEG 文件字节数计量。全量成本是该问题候选池内所有生成单元的 JPEG 字节总和。初始示例将每路视频最多均匀选 4 个时间单元，所以“全量输入”指当前候选池的全量帧，不是原始整段视频的每一帧；正式扩大评测时把 `max_chunks_per_view` 设为 `null`。该指标是可复现的离线传输代理，不代表 H.264/H.265 连续视频码流或真实网络包字节数。若后续改为重新编码视频片段，须使用统一编码器和参数，并作为新的成本定义版本报告。

预算是单个查询的总上传字节数。策略输出不得超过预算；无法整除时向下取整并记录未使用额度。除字节数外，同时记录传输帧数、选中视角数、推理输入帧数及估算 token，避免把通信成本和模型上下文成本混为一谈。

## 数据与划分

- 开发与主评估覆盖 MVVBench 的 Ego-Exo4D 和 MMPTrack 子集；先统计各域有效 QA、场景、视角数、时长及缺失视频，再冻结样本清单。
- 按 `sample_name` / 场景分组划分，避免同一场景的问题跨训练、验证和测试集。数据较小时优先采用固定的场景级验证/测试划分并报告样本数；不能把 QA 随机拆分后称为跨场景泛化。
- 所有方法使用相同问题、模型版本、提示词、解码设置和随机种子。按来源域分别报告结果，并报告汇总值。
- `views` 与时间标注可用于构造全量输入和 oracle 诊断上界，但不能输入实际选择器。

## 对照方法

1. 全量输入：质量上界与 100% 成本锚点。
2. 均匀时间采样：各视角轮询抽样。
3. 固定视角采样：只取排序后第一个视角的时间单元，不利用查询。
4. 查询驱动选择：只用问题文本及运行时可见的视频侧信息。
5. Oracle 诊断上界（可选）：利用标注所需视角或时间窗，仅用于衡量选择空间，不作为可部署基线。

## 指标与报告

- QA 准确率（总体及数据源、问题类别分组）。
- 实际上传字节数及相对全量成本比例。
- 准确率—字节数曲线、曲线下面积，以及各预算点相对全量准确率的变化。
- 配对问题级差值和置信区间；统计单位按场景聚类，避免把同场景 QA 当作完全独立样本。
- 失败案例：关键视角漏选、时间片段错选、信息压缩损失、模型推理失败分别归因。

## 首轮执行顺序

1. 用服务器本地路径配置完成数据盘点，导出可用样本清单和缺失项。
2. 固定模型/提示词/视频编码参数，跑全量输入与均匀采样小样本，确认成本和准确率可复现。
3. 对预算网格运行基线和查询驱动策略，保存逐题选择、输入成本、答案及运行配置。
4. 检查曲线与错误样本，再决定是否增加学习式选择策略或网络仿真。

## 当前实现接口

`scripts/score_video_evidence.py` 读取 MVVBench 问题标注、视频映射和整理后视频路径；按配置做场景级划分，在每个问题指定的候选视角内抽取 JPEG 帧并调用本地 Qwen3-VL 给每个视角/时间单元打相关性分数。标注 `views` 只用于定义基准指定的候选池；模型看到的是实际输入帧、摄像机编号和采样时间，不会看到答案、正确选项或标注时间。分数是同一个 VLM 的启发式判断，需要用均匀抽帧、随机/固定基线对照，不能直接视为真实证据价值。

`scripts/select_video_evidence.py` 接收打分器输出的候选清单。每个单元包含 `item_id`、`view_id`、`start_s`、`end_s`、`bytes` 和 `relevance_score`。脚本生成全量、均匀时间轮询、固定单视角和分数/字节贪心四种选择计划。

计划生成后，`scripts/run_video_qa.py` 对每个计划选择的 JPEG 帧运行多选 QA，将回答、正确选项、置信度和帧字节代理写入逐题记录；正确选项只供离线分析脚本使用。`scripts/analyze_video_qa.py` 对完全配对的计划与预测计算准确率、平均上传字节和预算利用率，并按数据来源域分别汇总。当前打分器是可复现的 VLM 启发式基线，选择器是贪心基线，不应当作最终研究方法或学得的查询理解能力。

候选单元示意：

```json
[
  {
    "query_id": "example-q1",
    "items": [
      {"item_id": "view0-000", "view_id": "0", "start_s": 0.0, "end_s": 1.0,
       "bytes": 18420, "relevance_score": 0.73}
    ]
  }
]
```

## 时间戳与视角互补诊断

主实验的策略不读取 `timestamp_start/end`。另设 `scripts/diagnose_mvvbench_evidence.py` 作为 oracle 诊断：按每题标注时间窗截取证据，只把问题、选项和抽出的图像交给 VLM，正确答案只用于离线统计。它分两组条件：

1. **视角消融**：所有标注指定视角、每个单视角、以及逐个移除一个视角；每个保留视角固定抽 4 帧。用于观察多视角是否优于最佳单视角，以及移除某个视角的影响。
2. **帧数扫描**：固定使用所有指定视角，每个视角均匀抽 1、2、4、8 帧。用于观察增加时间证据是否提升答案正确率，同时记录实际 JPEG 字节数。

非零长度时间窗内，采样点在各等分子区间中心；当标注时间戳起止相同，默认扩展为以该时间点为中心、总长 2 秒的窗口，并在视频边界截断。此参数可用 `--point-window-seconds` 调整。视角与帧数扫描会复用完全相同的“全视角、每视角 4 帧”条件，以便核对结果。输出含 `diagnostic_manifest.json`、可断点续跑的 `predictions.partial.json`、完成版 `predictions.json` 和逐条件任务目录。

服务器先用少量问题确认接口和耗时：

```sh
python scripts/diagnose_mvvbench_evidence.py \
  --config configs/exp002.local.json \
  --questions artifacts/EXP-002/score_run08/evaluation_questions.json \
  --output artifacts/EXP-002/diagnostic_run01_pilot \
  --questions-per-domain 2
```

确认后运行完整问题集；若四小时作业中断，使用同一命令、同一路径并加 `--resume`：

```sh
python scripts/diagnose_mvvbench_evidence.py \
  --config configs/exp002.local.json \
  --questions artifacts/EXP-002/score_run08/evaluation_questions.json \
  --output artifacts/EXP-002/diagnostic_run01 --resume
```

可用 `--modes view_ablation` 或 `--modes frame_sweep` 单独运行一组；可用 `--frame-counts 1 2 4 8` 和 `--ablation-frames 4` 显式调整采样量。诊断结论仅适用于“已知正确时间窗”的 oracle 设置，不能据此声称系统能够定位相关事件，也不能把 oracle 条件与不使用时间戳的策略做同信息量比较。

汇总完成的结果：

```sh
python scripts/analyze_mvvbench_diagnostic.py \
  --predictions artifacts/EXP-002/diagnostic_run01/predictions.json \
  --output artifacts/EXP-002/diagnostic_run01/report.json
```

服务器路径已在试运行配置中完成设置。路径通过 `*.local.json` 配置；配置中的数据集域可设为 `egoexo`、`mmptrack` 或二者。当前扩展评测计划使用场景去重的问题集（EgoExo 88 题、MMPTrack 16 题），保持 `one_question_per_scene`，并将 `max_questions_per_domain`、`max_chunks_per_view` 设为 `null`。当前只适配 MVVBench 注释格式，扩展到不同 schema 的数据集时需新增适配器。

## 当前限制

MVVBench 可评估多视角问答质量和离线通信量折中，但不包含网络链路带宽变化、并发流、丢包和端到端时延。真实 Mbps、动态带宽分配和实时性能需要后续单独建模/测量。按 MVVBench 与原始视频数据集各自许可使用数据。

