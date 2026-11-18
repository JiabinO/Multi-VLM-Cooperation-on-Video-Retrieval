# W03–W06 开发集问答实验 v1

本轮问题由用户审核观察草稿并补充澄清后整理，在问答运行前固定。
配置为 configs/phase1_qa_v1.json，6 题、五组，合计 30 次生成；同一窗口只抽帧一次。
模型只加载一次，每题都是独立单轮请求，不包含观察草稿、人工说明或参考答案。
W02 入口和历史输出保留。该批题不是独立测试集，同一模型辅助预描述可能带来题目偏向。
新入口仅完成静态与离线测试，服务器 GPU 尚待执行，不改变现有 Conda 环境。

## 已固定的问题

| ID | 问题 | 参考答案 |
| --- | --- | --- |
| W03-Q1 | 右手使用什么工具，对哪个部位做什么操作？ | 螺丝刀拧动模型底部红色螺丝 |
| W04-Q1 | 拧动黑色组件上的螺丝后，是否将螺丝取下？ | 是，取下螺丝 |
| W05-Q1 | 从桌面拿起的红色零件是什么？ | 螺丝 |
| W05-Q2 | 拿走一颗螺丝前后，红色螺丝数量如何变化？ | 七颗变六颗 |
| W06-Q1 | 安装中的零件发生了什么？ | 零件掉到桌面 |
| W06-Q2 | 零件掉落后怎样放置车身，接着做什么？ | 车身放回桌面固定，继续安装 |

W04 是带先验的是非诊断题；W05-Q2 提示减一，必须答对起止数量才算计数正确。
W06-Q2 已提示掉落，不能将该题结果当成自行发现掉落的能力。
每题语义评分，correct / partial / incorrect / uncertain，支持同义表达，
多项问题拆分记录工具、对象、动作及顺序；未看清与明确判断错误分开记录。
帧引用不额外加分；不预先认定某个视角冗余或必须跨视角才能回答。
相同录制/窗口的问题相关，不能将 30 次调用当成 30 个独立视频样本。

人工依据：W03 右手操作底部红色螺丝；W04 用户补充帧 4、5 取下螺丝；
W05 右手拿红色螺丝、数量七变六，C10095 较早伸手与黑色零件投影重叠而非拿起黑色零件；
W06 完整顺序为悬空持车、安装零件、零件掉落、车身放回桌面固定、继续安装。
W03_C10115 的具体螺丝身份是否独立可见尚待确认，参考依据采用其他已审核视角。
用户审核表中完整片段回看字段为否，支持帧及各步骤逐视角可见性未全部核定，
因此定位为开发实验，不声称建立了精确证据标注或官方答案。

## 服务器执行

先激活现有环境，避免误用 base；不安装任何依赖。

```bash
conda activate vlm_redundancy
python -c "import cv2, torch; print('OpenCV:', cv2.__version__); print('CUDA:', torch.cuda.is_available())"
python -m compileall -q scripts tests
python -m unittest discover -s tests -v
python scripts/phase1_qa.py \
  --config configs/phase1_qa_v1.json \
  --clips artifacts/phase1/assembly101_9011_a01/clips \
  --model /home/scc/pb22111667/event_casual_rag/models/Qwen3-VL-8B-Instruct \
  --frames 8 --max-new-tokens 128 \
  --output artifacts/phase1/qa_v1_run01
```

可用 --question-ids W03-Q1 先运行一题，须更换输出目录；仅抽帧检查加 --extract-only。
固定组顺序为 full、C10095、C10115、C10118、C10095+C10115，首调用冷启动，
不同回答长度及共享 GPU 负载都影响耗时，单次调用不作稳定速度结论。
为保证可比性仍使用每路 8 帧与现有 640×360 五秒片段；偏移根据窗口配置约定，
未独立验证源剪辑 PTS 和同步。某路未覆盖某动作时可答性要另记，不改问题迁就输出。

## 回传和审核

输出根目录包含 run.json、questions.snapshot.json、review.json；每个题目/组目录
保存原始回答、实际 prompt、参考配置、输入 token、耗时、采样引用及错误日志。
图片和预览在 views/Wxx_camera/ 中，各问题复用同一路采样。
任一任务失败整体非零，其他任务继续；既有输出目录拒绝覆盖。

运行后回传 review.json、run.json、questions.snapshot.json 以及所有组的
raw_response.txt、metrics.json、error.log；有证据疑问再补预览与原图。
review.json 中所有评分初始为 pending，不自动生成正确率，失败任务不能当作正常答案。
更改题目或参考答案需另开配置版本并记录原因，不覆盖本轮题目快照或历史输出。
