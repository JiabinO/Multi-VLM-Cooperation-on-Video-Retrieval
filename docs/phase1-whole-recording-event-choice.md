# 整段视频官方事件选项筛查

## 实验问题

对同一 Assembly101 长序列的 C10095、C10115、C10118 三路视频分别输入完整录制的均匀采样帧。模型看到同一组从该序列三路官方细粒度标签合并出的动作选项，逐项判断当前机位是否有画面证据、在采样中未观察到，或无法确定，并给出自评置信度和证据帧。

这是一项**多标签事件盘点**，不是单选：完整录制包含多个动作。选项次序使用固定随机种子，并在三路保持一致；模型 prompt 不含官方时间段和答案。官方标注只在模型输出后用于候选对照。

## 重要解释

- `visible` 必须有当前输入帧证据；`not_observed` 仅表示本次抽样画面没有观察到，不能证明整段没有发生；`uncertain` 表示画面或时间采样不足。
- 事件标签的采样覆盖由标注区间与实际输入帧时间计算。区间内有至少两帧才作为较强的候选；只有一帧或零帧命中时，不能简单计为模型错误。
- 某动作未出现在一个机位的官方细粒度标注里，不代表官方已证明该机位不可见。人工审核需要查看原始视频和模型引用帧，才能判断是视角差异、标注遗漏还是模型错误。
- 模型自报置信度不是校准概率。应先人工审核若干高低置信度正确/错误例子，再讨论置信度是否有区分性。
- 全片等间隔抽样会漏掉很短的动作。此实验用于发现候选标签与机位差异；如果目标事件没有足够输入帧覆盖，下一步需单独做事件级密集采样，不能仅凭整段盘点得出结论。
- 默认每个视角 64 张输入图像；实际视频帧率仍是 30 fps 左右，64 是每路整段视频的抽样张数，不是 64 fps。

## 服务器操作

在仓库根目录、使用已有的 `vlm_redundancy` 环境运行，不安装或修改 Conda 环境。先将 `ANNOT_DIR` 指向此前导入官方标注的输出目录，其中应有 `annotation_manifest.json`：

```bash
conda activate vlm_redundancy
ANNOT_DIR=artifacts/phase1/annotation_import_run01
VIDEO_DIR="$HOME/vlm-data/phase1/assembly101/recordings/nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724"
```

先做不加载模型的完整视频采样与覆盖率预检：

```bash
python scripts/phase1_event_choice.py \
  --manifest "$ANNOT_DIR/annotation_manifest.json" \
  --videos "$VIDEO_DIR" \
  --frames 64 \
  --output artifacts/phase1/event_choice_extract01 \
  --extract-only
```

查看 `sampling_coverage_summary.json`，重点看哪些标签是 `at_least_two_sampled_frames`、哪些只有一帧或没有命中；同时打开三路 `preview.jpg` 确认整段均匀抽样覆盖。再以新输出目录运行模型：

```bash
python scripts/phase1_event_choice.py \
  --manifest "$ANNOT_DIR/annotation_manifest.json" \
  --videos "$VIDEO_DIR" \
  --model "$HOME/event_casual_rag/models/Qwen3-VL-8B-Instruct" \
  --frames 64 \
  --max-new-tokens 4096 \
  --output artifacts/phase1/event_choice_run01
```

若 64 帧在现有 GPU 上显存不足，可用 32 帧另开新 run；必须同时对比覆盖率，不能把 32 帧结果与 64 帧结果当作纯视角差异。

## 输出与审核

- `run.json`：输入配置、视频时长/帧率、运行环境、任务状态。
- `event_options.json`：同一组随机排序的事件选项及 ID。
- `<view>/frames.json`、`<view>/preview.jpg`：实际抽帧位置与预览。
- `<view>/raw_response.txt`、`parsed_response.json`、`metrics.json`、`error.log`：原始回答、解析结果、token/耗时和错误。
- `<view>/event_sample_coverage.json` 与 `sampling_coverage_summary.json`：各机位官方事件区间被抽样覆盖的情况。
- `gold_reference.json`：运行后供人工审核的官方标签对照；不注入模型 prompt，也不作为未审核的最终真值。
- `review.md`：逐机位对照模型选择、置信度、证据帧和官方标签的审核表。

优先人工复核三类项：官方有标注但模型选 `not_observed`；模型选 `visible` 但机位官方未标注该事件；模型对短时事件给高置信度但只引用一帧。复核后，只把“一个机位确有可见证据、另一机位确实缺少该信息”的事件交给下一步 QA 设计。
