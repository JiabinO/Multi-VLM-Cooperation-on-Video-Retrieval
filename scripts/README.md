# 运行入口

后续提供数据检查、小样本运行、服务器批处理与结果导出入口。每个脚本注明参数、工作目录、输入输出、失败恢复和日志位置。
首个入口：video_smoke.py，执行单路视频抽帧、Qwen3-VL 图像序列推理及证据校验。参数和命令见 ../docs/video-smoke.md；多端联合推理尚未实现。

Assembly101 事件预处理：phase1_annotations.py 使用 Python 标准库导入官方 fine/coarse action labels，按序列匹配事件到候选时间窗，并生成 8/24 帧三视角人工可见性审核表。字段和服务器命令见 ../docs/phase1-event-screening.md；候选时间映射不会自动标为已验证。

MVVBench 下载：先在 Hugging Face 数据集页面接受其非商业研究及遵守来源数据集许可的条件，再运行 `python scripts/download_mvvbench.py`。脚本将问答标注、视频映射和上游整理脚本保存到 `artifacts/mvvbench/`（不进 Git）；优先读取 `HF_TOKEN` 环境变量，也会使用已安装 huggingface_hub 的登录令牌。源视频不包含在 MVVBench 发布包中，须按三套原始数据各自的访问条件下载；脚本生成的 `DOWNLOAD_README.md` 记录后续目录和准备步骤。

EXP-002 VLM 打分：复制 `configs/exp002_budget_sweep.example.json` 为本机忽略文件（如 `configs/exp002.local.json`），填入本地模型、标注、映射、视频和输出路径。`dataset.domains` 可设为 `['egoexo']`、`['mmptrack']` 或二者。先用少量样本检查：

```sh
python scripts/score_video_evidence.py --config configs/exp002.local.json --output artifacts/EXP-002/score_run01
python scripts/select_video_evidence.py --candidates artifacts/EXP-002/score_run01/candidate_units.json --config configs/exp002.local.json --output artifacts/EXP-002/score_run01/plans.json
python scripts/run_video_qa.py --plans artifacts/EXP-002/score_run01/plans.json --assets artifacts/EXP-002/score_run01/candidate_assets.json --questions artifacts/EXP-002/score_run01/evaluation_questions.json --config configs/exp002.local.json --output artifacts/EXP-002/qa_run01
python scripts/analyze_video_qa.py --plans artifacts/EXP-002/qa_run01/plans.json --predictions artifacts/EXP-002/qa_run01/predictions.json --output artifacts/EXP-002/qa_run01/report.json
```

`score_video_evidence.py` 默认要求输出目录不存在。若运行中断且该目录里已有 `run.json`，可使用相同配置和输出目录加 `--resume` 继续；它会读取已保存的每个候选片段 `score.json`，跳过已经完成的 VLM 打分，并补齐最终候选文件。抽帧会重新检查/生成，尚未写完的打分会重新执行。

```sh
python scripts/score_video_evidence.py --config configs/exp002.local.json --output artifacts/EXP-002/score_run01 --resume
```

`run_video_qa.py` starts a fresh run by default and requires a new output directory. To continue an interrupted run, pass the same plans, assets, questions, config, and output directory again with `--resume`; it reads `predictions.partial.json` and skips completed query/policy/budget conditions. It verifies that the requested plans match the saved `plans.json` before continuing. Without `--resume`, an existing output directory is an error, which avoids accidental overwrites.

```sh
python scripts/run_video_qa.py --plans artifacts/EXP-002/score_run01/plans.json --assets artifacts/EXP-002/score_run01/candidate_assets.json --questions artifacts/EXP-002/score_run01/evaluation_questions.json --config configs/exp002.local.json --output artifacts/EXP-002/qa_run01 --resume
```

预算成本是固定质量 JPEG 抽帧文件的字节数，用于离线策略对比，不等于网络实际传输。`score_video_evidence.py --extract-only` 只抽帧，不生成可用计划。运行参数和数据限制见 `../docs/exp-002-query-driven-transmission.md`。

MVVBench 时间戳诊断：`diagnose_mvvbench_evidence.py` 专门检查“标注时间窗内，多个 view 是否互补”以及“帧数增加是否改善回答”。它读取 `evaluation_questions.json` 和原始标注中的 `timestamp_start/end`，在标注时间窗内均匀取帧；视角消融固定每个 view 4 帧，帧数扫描固定全部指定 view，默认每 view 取 1/2/4/8 帧。零长度时间戳默认扩展为以该时刻为中心的 2 秒窗口。该诊断使用 oracle 时间戳，结果不能当作无时间先验的部署策略或与主实验直接等价的基线。完整说明和命令见 `../docs/exp-002-query-driven-transmission.md`。用 `analyze_mvvbench_diagnostic.py --predictions .../predictions.json --output .../report.json` 导出按数据集和条件汇总的准确率、帧数、字节数以及与全视角条件的配对差值。




自动筛查逐题复核候选：

```sh
python scripts/screen_mvvbench_diagnostic.py \\
  --predictions artifacts/EXP-002/diagnostic_run01/predictions.json \\
  --output-dir artifacts/EXP-002/diagnostic_run01/screening
```

默认依据运行清单中的 `prompt_mode` 过滤同类预测，并检查条件覆盖、全视角/单视角/留一视角正确性差异、答案变化、置信度异常和帧数影响。输出 `review_queue.csv`、`screening_report.json` 与说明文件；报告包含题目级触发原因、预测条件、帧时间及对应模型输入图像路径。它只负责排序人工复核候选，不自动证明视角互补或干扰。中断运行可将输入改为 `predictions.partial.json`，但需先查看条件覆盖度。
