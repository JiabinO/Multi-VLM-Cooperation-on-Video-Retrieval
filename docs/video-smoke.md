# SMOKE-001：单路视频采样帧验证

目的：检查真实视频抽帧、证据引用和结果保存。不是完整视频理解评测，也不是多摄像头冗余实验。

已由服务器端人工验证的环境：Python 3.11.17、PyTorch 2.9.1+cu128、Transformers 4.57.6、OpenCV 4.11.0、A100 80GB；单图推理正常。本脚本的 GPU 推理仍需服务器首次验证。

## 环境
```sh
conda activate vlm_redundancy
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-smoke.txt
python -m pip check
```
已有环境无需重复安装 PyTorch。依赖文件约束直接依赖；每次运行后另存 pip freeze，形成实际环境快照。

## 执行
在仓库根目录，用新的输出目录执行：
```sh
python scripts/video_smoke.py \
  --video /home/scc/pb22111667/event_casual_rag/datasets/Video-MME/videos/_8lBR0E_Tx8.mp4 \
  --model /home/scc/pb22111667/event_casual_rag/models/Qwen3-VL-8B-Instruct \
  --start 0 --end 30 --frames 8 \
  --output artifacts/SMOKE-001/video_mme_first30_run01
```
仅检查解码与抽帧时，添加 --extract-only，使用另一个输出目录。输出目录已存在则拒绝覆盖。调整时间范围或帧数后使用新的运行编号。

## 产物与检查
- preview.jpg、frames/*.png：8 帧预览与原图；人工检查引用是否支持事实。
- frames.json：帧编号、名义时间、解码器报告时间和图片 SHA256。
- prompt.txt、raw_response.txt：实际任务说明及未经修复的原始回答。
- evidence.json：通过结构与帧编号检查的观察；不代表事实已获人工验证。
- metrics.json：生成耗时、token、响应字节、显存和达到长度限制的提示。
- run.json、error.txt：配置、版本、代码提交、状态及错误；解析失败非零退出并保留原文。

时间区间为 [start,end)，最后一帧不会越过 end。时间用帧索引/FPS 推算，适合本轮近似恒定帧率测试；不是经核对的真实 PTS。不得据此对变帧率数据作精确时间定位。
该脚本以多张带时间标签的图片输入模型，没有使用原生视频接口或音频。采样时间包络不是事件真实起止时间。只看 8 帧不能推断帧间未见的事件。
输入 token 含视觉占位展开，响应字节是文本 UTF-8 大小；两者都不是实际设备通信测量。

运行后可在输出目录保存环境：
```sh
python -m pip freeze > artifacts/SMOKE-001/video_mme_first30_run01/environment.txt
```
先回传 run.json、metrics.json、raw_response.txt 和 preview.jpg，再决定是否扩大窗口。模型目录当前记录本地路径，正式实验还需补齐模型文件版本/校验清单。
