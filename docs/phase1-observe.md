# VLM 辅助观察与人工审核

状态：已完成本地离线验证；新入口尚未进行服务器 GPU 验证。沿用现有
vlm_redundancy 环境，不安装依赖。模型加载和生成复用 w02_qa.QwenRuntime，
仅增加可选 prompt 参数，W02 默认提示词不变；抽帧复用 video_smoke 流程。

## 执行

在仓库根目录、现有 Conda 环境中执行。输入为已有的五秒片段，不能传完整视频。
窗口原视频偏移固定为 W01=20、W02=60、W03=100、W04=140、W05=180、W06=220 秒。
偏移是数据准备约定，未经脚本独立验证。每路 8 帧，只观察采样画面，不声称看完连续视频。

```bash
python -m compileall -q scripts tests
python -m unittest discover -s tests -v
python scripts/phase1_observe.py \
  --clips artifacts/phase1/assembly101_9011_a01/clips \
  --windows W03 W04 W05 W06 \
  --model /home/scc/pb22111667/event_casual_rag/models/Qwen3-VL-8B-Instruct \
  --frames 8 --max-new-tokens 768 \
  --output artifacts/phase1/observe_run01
```

可先用 --windows W03 试跑三路；全部执行共 12 次生成，模型只加载一次。
仅验证抽帧时加 --extract-only 并换输出目录。任何任务失败整体返回非零，
其他任务仍继续；失败项必须查看错误，不能当作“没有观察到事实”。
已有输出目录拒绝覆盖。Ctrl+C 保留此前已完成的任务与审核表。

## 输出与需要审核的内容

- review.md：适合直接阅读并修改的中文审核表。
- review.json：相同草稿的结构化审核模板，初始 is_gold=false，所有条目 pending。
- 每个 Wxx_camera 子目录：preview.jpg、frames/、frames.json、原始回答、
  prompt.txt、metrics.json、draft.json、task.json、error.log 和抽帧日志。
- 根 run.json：执行代码版本、环境、参数、整体及逐任务状态。

优先打开 review.md，同时查看对应预览、必要时回看完整片段。每条选择：
confirm（确认）、correct（修正）、delete（错误或无依据）、uncertain（无法判断）。
correct 时写出修正后的事实；pending 保留为尚未处理，不能当作确认。

你需要重点检查：
1. 组件是否认错，名称不同是否实际指向同一组件。
2. 动作是否看得出来；“拧动”不能无依据升级为“拧松”“拧紧”或“拆下”。
3. 前后状态变化是否有画面支持，是否把姿态猜成完整动作。
4. 左右手的描述是否正确，有无遮挡。
5. 模型遗漏了哪些事实，尤其是该视角独有或更清楚的信息。
6. 看不清的细节；“看不清”不能写成“没有发生”。

每条观察现在包含模型自评置信度（0–1）、对支持画面的文字证据、支持帧编号和不确定点。审核时核对文字证据是否确实出现在所列帧中、这些证据支持 claim 到什么程度，以及模型是否把推测写成事实。置信度只是模型自评，不是校准后的概率或准确率；不能只按高低接受或删除事实。帧编号仅作证据核查，不作为主要理解能力评分。三路独立生成后再人工对照，不能用另一视角替当前视角补证据。

review.md 与 review.json 不自动双向同步。建议只编辑 review.md，另存
review.human.md 后发回（保留草稿）；需要程序化处理时编辑 review.json，
另存 review.human.json。后续再转换为确认后的事实集及候选 QA，本脚本不自动出题，
不把模型输出或人工尚未处理的条目升级为真值。

建议回传 review.human.md、run.json、各路 raw_response.txt 与 preview.jpg；
如有失败，再带对应 error.log。完整采样 PNG 留服务器，有疑问再取单张。

同一个 Qwen 辅助出题与参加评测会产生模型偏向；本阶段作为开发标注辅助，
需保留人工补充和修正来源，正式评测另做独立人工审核及其他录制。
