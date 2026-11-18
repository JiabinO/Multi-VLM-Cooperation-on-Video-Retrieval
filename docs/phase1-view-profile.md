# 机位视角描述能力实验

目的：测试 VLM 能否从单一机位跨多个时间窗的抽样帧中，归纳稳定的场景布局、画面覆盖范围和重复遮挡，同时把窗口内临时动作与机位特征分开。每次推理仅输入一个机位，模型不能看到其他摄像头，因此不能自行断言信息“独有”或“其他视角不可见”；共享/独有归类留给后续跨机位对齐和人工审核。

## 运行方式

沿用现有 `vlm_redundancy` Conda 环境、`video_smoke.py` 抽帧方式和 `w02_qa.QwenRuntime` 模型加载方式，不安装依赖、不改环境。输入是已有的五秒 `Wxx_Cxxxxx.mp4` clips。默认 W03–W06 每窗抽 2 帧，即每机位 8 帧、共 3 次生成；建议先验证抽帧，再做 GPU 推理。

```bash
python -m compileall -q scripts tests
python -m unittest discover -s tests -v

CLIPS=artifacts/phase1/assembly101_9011_a01/clips
python scripts/phase1_view_profile.py \
  --clips "$CLIPS" --windows W03 W04 W05 W06 \
  --frames-per-window 2 \
  --model /home/scc/pb22111667/event_casual_rag/models/Qwen3-VL-8B-Instruct \
  --max-new-tokens 1024 \
  --output artifacts/phase1/view_profile_run01
```

抽帧单测使用独立的新输出目录：

```bash
python scripts/phase1_view_profile.py \
  --clips "$CLIPS" --windows W03 W04 W05 W06 \
  --frames-per-window 2 --extract-only \
  --output artifacts/phase1/view_profile_extract_check01
```

输出目录不能预先存在。每个机位任务保留其跨窗口帧清单、各窗预览、prompt、原始回答、JSON 草稿、token/耗时指标、任务状态和错误日志；根目录有 `run.json`、`review.json`、`review.md` 和错误日志。审核时先看 `review.md` 中的帧证据，再打开相应窗口预览；确认某描述跨多个窗口重复出现后，才归入稳定机位信息。单窗动作不算固定背景。模型置信度是自评，不能直接解释为正确概率。

这只是局部场景/视野概括能力的探索性实验；并不验证摄像机物理标定、完整视频理解或真实网络传输压缩率。当前输出尚未与问题相关性、跨视角共享信息和端到端 token 压缩率绑定。
