# Assembly101 W02 多视角问答

状态：W02_qa_run01 已完成服务器 GPU 推理；用户人工复核为四组正确、full 错误。
不安装、升级或修改现有 Conda 环境。入口复用 video_smoke.py 的 extract-only
流程和帧校验，模型采用相同的本地 AutoProcessor / Qwen3VLForConditionalGeneration、
BF16、CUDA:0、SDPA、apply_chat_template 多图片输入及 greedy generate 方式。

固定窗口 [60,65) 秒；必须传入同一 Assembly101 录制的三路完整同步视频。
用户需核实 recording/session 身份和相机标识；本实现不凭摄像机编号猜测具体录制。
未提供真实数据路径，不能声称已经验证样本身份或视频内容。
默认每路 8 帧；full=24 帧，三个单路各 8 帧，C10095+C10115=16 帧。
按视角再按时间排列；各组引用同一份采样 PNG，保留帧索引、名义时间和 SHA256。
沿用 FPS 名义时间，并未验证真实 PTS 或跨相机同步偏差。

问题：操作者从推土机模型上拆下了哪个部件？
人工复核后的参考表达：铲斗部件。原始运行使用“钩子部件”，原始记录保留不改。
参考答案仅写入运行元数据，不进入提示词，也不用于自动字符串匹配评分。
按是否指向实际拆下的组件评判；同义名称或足以确定同一组件的描述可接受，
含糊描述记为部分正确或无法判定，指向其他组件记为错误。不得仅凭文本相似度评分。

## W02_qa_run01 人工复核

用户于 2026-10-26 回看视频，确认 C10095、C10115、C10118、C10095+C10115
均正确指向铲斗；full 回答“红色的履带部件”，判错。C10115 对帧 6、7 中
操作者持握拆下组件的描述符合画面，作为附加事实核查，不额外提高答案得分。
记录见 results/Assembly101-W02-QA/W02_qa_run01/human_review.json。
本记录是已查看输出后的人工复核，不冒充事先制定的盲评或数据集官方标注。
本题四个条件正确不等于跨样本 80% 准确率，也不足以证明某视角冗余或有害。
选帧、帧引用用于核查证据，不作为当前慢变化装配素材的主要理解能力指标。

## 服务器命令

在现有 Conda 环境和仓库根目录执行。以下三个环境变量应指向同一录制的完整视频。
已截取的 5 秒 W02 文件需要加 --w02-clips：按片段内 [0,5) 秒采样，
frames.json 同时保存 clip_time_sec 和加上 60 秒后的原视频名义时间。
此偏移是用户声明的剪辑来源，不代表脚本验证了剪辑边界。

```bash
conda activate vlm_redundancy
git status --short
git fetch origin
git switch --detach <本次提交SHA>
export W02_C10095=/absolute/path/to/session/C10095.mp4
export W02_C10115=/absolute/path/to/session/C10115.mp4
export W02_C10118=/absolute/path/to/session/C10118.mp4
python -m compileall -q scripts tests
python -m unittest discover -s tests -v
python scripts/w02_qa.py --help
python scripts/w02_qa.py \
  --C10095 "$W02_C10095" --C10115 "$W02_C10115" --C10118 "$W02_C10118" \
  --output artifacts/W02/extract_run01 --extract-only
# 人工检查 views/*/preview.jpg、frames.json 后执行：
python scripts/w02_qa.py \
  --C10095 "$W02_C10095" --C10115 "$W02_C10115" --C10118 "$W02_C10118" \
  --model /home/scc/pb22111667/event_casual_rag/models/Qwen3-VL-8B-Instruct \
  --frames 8 --max-new-tokens 128 --output artifacts/W02/qa_run01
```

每个组输出 run.json、metrics.json（实际展开后的 input_tokens、生成耗时、
输出 token、峰值显存）、frames.json、prompt.txt、raw_response.txt 和 error.log。
失败或只抽帧时没有原始回答，尚未测量的指标为 null；不伪造数字。
采样图片及预览位于 views/<camera>/，各组 frames.json 记录其绝对路径。
抽帧进程 stdout/stderr 单独保存，组错误不中断其他组，任一失败整体非零退出。
模型只加载一次，加载耗时单独记录；按五组固定顺序运行，第一组冷启动，
后续组可能受预热影响，单次耗时不能视作严格的速度对照。
输出目录存在时拒绝覆盖；重跑必须换 run ID。

## 离线验证

```bash
python -m compileall -q scripts tests
python -m unittest discover -s tests -v
```

test_w02_qa.py 用显式 offline-test-double-NOT-GPU 模拟抽帧和推理，
验证五组输入数量、答案隔离、失败日志、失败后继续、禁止覆盖及 extract-only
不加载模型。此测试不证明 OpenCV 解码、CUDA 可用或 Qwen 回答正确。

2026-10-26 本地验证：Python 3.14，compileall 通过，完整 unittest 20/20 通过，
CLI --help 通过。未安装任何依赖；本地缺少 cv2 和 torch，未执行真实解码或 GPU 推理。
