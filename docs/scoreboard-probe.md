# PROBE-001：复用 8 帧逐帧读取比分

目的：在同样的采样图像上，测试带问题的证据提取和逐帧对应。此阶段是单视频诊断，不是多端实验。

## 人工参考答案
data/manifests/video_mme_scoreboard_review.json 记录本次聊天的人工核查，非 Video-MME 官方标注：
- 帧 1、2、3：ENG 0:0 DEN；比赛计时分别 5:39、5:43、5:48。
- 帧 0：FOX SPORTS 片头已确认，但未确认是否没有比分牌，因此不评分。
- 帧 4–7：疑似进球回放，暂定；比分和计时未核查，因此不评分。

参考答案不进入提示词，模型输入只有图像、全局帧编号、视频时间与通用读取要求。
复用原始 PNG，核对其 SHA256、原视频文件名、帧索引；不重新抽帧，也不修改原运行目录。
这些检查并非原始视频的密码学身份证明；正式实验需进一步完善数据版本标识。

## 执行
在已激活 vlm_redundancy 的仓库根目录执行：
```sh
python scripts/scoreboard_probe.py \
  --source artifacts/SMOKE-001/video_mme_first30_run01 \
  --model /home/scc/pb22111667/event_casual_rag/models/Qwen3-VL-8B-Instruct \
  --output artifacts/PROBE-001/scoreboard_run01
```
source 指向上次包含 frames.json、run.json 和 frames/*.png 的目录；如果当时用了其它目录，请替换。输出目录必须不存在，重跑用新编号。
默认 max-new-tokens=768，与上次保持一致；若截断或 JSON 不完整，保留失败记录，再另开运行增加上限。
只核查输入文件，不加载模型：追加 --validate-only 并使用新的输出目录。

## 输出
- raw_response.txt：原始回答，不自动修复。
- readings.json：每个输入帧恰好一条；缺帧、重复编号、无效字段或非法计时导致 parse_failed，非零退出并保留原文。
- evaluation.json：仅对帧 1–3 逐字段比较，记录预期、实际、正确与否；分母固定为人工核查的 3 帧。其它帧不作负例。
- report.md：中文简报，解释评分范围。
- metrics.json：生成时间、token、文本字节、峰值显存、是否达到生成上限。
- comparison.json：如原运行有 metrics.json，列出新旧测量及差值；任务和提示词不同、各只有一次运行，不能据此宣称加速或性能提高。
- prompt.txt、reference.json、frames.json、run.json：实际配置、人工记录、输入索引、代码与环境版本；运行状态完成不等于答案正确。

原图仍保存在 source。结果请结合原运行 preview.jpg 检查，特别核查未评分帧。两种时间分别保存：视频时间由输入清单提供，比赛计时由模型读取，禁止相互填充。
全字段匹配率只描述这 3 帧，不是 Video-MME 准确率。多帧一起输入意味着模型仍可能受其它帧影响；提示词禁止补全并不能证明模型没有跨帧推断，后续可用逐帧独立调用验证。

## 本地验证
```sh
python -m unittest discover -s tests -v
```
测试不需要下载模型或安装 GPU 依赖，覆盖部分标注评分、非法输出拒绝、时间格式、图片校验及无模型输入验证。GPU 推理需要在服务器验证。
