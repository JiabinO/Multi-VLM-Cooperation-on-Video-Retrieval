# 机位视野描述作为 Prompt 先验的配对实验

## 目的与对照

测试加入经人工审核的 `view_coverage` 是否改善当前窗口的事件观察，同时监测先验是否引入无当前图像证据的复述。每个“窗口 × 机位”只抽一次帧，两种条件使用相同帧、模型、生成参数和观察任务：

- `no_profile`：不提供机位历史描述。
- `approved_profile`：只增加该机位人工审核通过的 `view_coverage`，并明确它是定位区域的历史先验，不是当前事实。

两组都输出当前帧直接支持的事实和仍无法确认的信息。置信度是模型自评，不是校准概率。该实验是配对的提示消融，不代表真实网络传输节省。

## 准备已审核配置

复制 `configs/camera_view_profiles.example.json` 为单独的运行配置。只填人工看过并确认的视野描述；对应机位完成审核后才将 `approved` 设为 `true`。不要把未审核的 raw response、临时动作或猜测的“独有信息”放入配置。脚本会校验状态，并在输出中保存配置哈希和本次实际使用的描述快照。

配置形状如下：

```json
{
  "schema_version": "camera-view-profile-v1",
  "profiles": {
    "C10095": {"approved": true, "view_coverage": "人工审核后的画面覆盖描述"},
    "C10115": {"approved": true, "view_coverage": "人工审核后的画面覆盖描述"},
    "C10118": {"approved": true, "view_coverage": "人工审核后的画面覆盖描述"}
  }
}
```

## 服务器执行

先激活已有环境；不需要安装或修改 Conda 包。建议先用 W03 三机位做 6 次生成的配对试跑，检查审核表和错误日志后，再扩展到 W03–W06（24 次生成）。

```bash
conda activate vlm_redundancy
python -c "import cv2, torch; print('OpenCV:', cv2.__version__); print('CUDA:', torch.cuda.is_available())"
python -m compileall -q scripts tests
python -m unittest discover -s tests -v

CLIPS=artifacts/phase1/assembly101_9011_a01/clips
PROFILES=configs/camera_view_profiles_run01.json
MODEL=/home/scc/pb22111667/event_casual_rag/models/Qwen3-VL-8B-Instruct

python scripts/phase1_view_prompt_ab.py \
  --clips "$CLIPS" --profiles "$PROFILES" --windows W03 \
  --frames 8 --model "$MODEL" --max-new-tokens 1024 \
  --output artifacts/phase1/view_prompt_ab_w03_run01
```

试跑后确认生成和审核流程正常，再对其余窗口执行完整批次；输出目录必须使用新名称：

```bash
python scripts/phase1_view_prompt_ab.py \
  --clips "$CLIPS" --profiles "$PROFILES" --windows W03 W04 W05 W06 \
  --frames 8 --model "$MODEL" --max-new-tokens 1024 \
  --output artifacts/phase1/view_prompt_ab_run01
```

只做抽帧验证不要求模型或 profiles 文件：加 `--extract-only` 并使用新的 `--output` 目录。脚本仍会保存两条件的配对输入清单，但不运行推理。

## 审核与解释

根目录 `review.md` 按配对列出两组回答；每组目录有原始回答、实际 prompt、帧清单、token/生成耗时和错误日志。逐对记录：

1. 机位提示组新增了哪些有直接帧证据的事实、减少了哪些遗漏。
2. 是否把 profile 文字直接复述为当前事实，且当前帧没有支持。
3. 两组各自的错误、共同遗漏及不确定表达变化。
4. 该差异是否影响事件问题所需事实，而非只让回答更长或更像提示。

先查看 W03 小试结果，确认该问题和标注口径有区分度后再扩展。比较答案时以人工事实审核为准；不要仅凭 token 数或自评置信度宣称效果。当前 `w02_qa.QwenRuntime` 的 token 是模型输入 token；如要声称通信字节节省，还需另行测量序列化后实际传输的完整消息。
