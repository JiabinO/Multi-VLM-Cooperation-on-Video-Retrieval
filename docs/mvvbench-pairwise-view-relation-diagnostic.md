# MVVBench 两图空间关系诊断

## 目的

用一对尽量时间对齐的单帧图像，定性区分 VLM 的跨视角能力：能否自己发现共同实体；获知“同场景、不同机位”后能否利用稳定地标描述粗略视角关系；人工提供已核验对应物后，这种关系判断是否改善。此诊断不要求精确相机位姿，也不输入原始 QA 问题、选项或答案。

## 条件

脚本对每种条件分别以 A→B 和 B→A 顺序运行一次：

1. `blind_pair`：不告知是否同一场景，检查模型是否提出有图像依据的共同实体与关系。
2. `same_scene_known`：告知同一场景、不同机位，检查提示先验是否帮助它识别稳定地标并描述视角差异。
3. `verified_anchors`：再提供人工确认的跨图对应物，隔离实体匹配失败与关系推理失败。

交换输入顺序只是稳定性检查，不是额外独立样本。人物会移动，因此优先使用球门、墙角、固定设备等静态地标；人物和球可作为辅助对应物，不能单独据此断言机位移动。

## 服务器运行

将脚本放入服务器仓库的 `scripts/` 后，在 `vlm_redundancy` 环境运行。以下路径使用此前足球案例已经抽取的 View 2 / View 3 单帧；若文件位置不同，替换成对应的人工检查过的图片。

```bash
python scripts/diagnose_pairwise_view_relation.py \
  --image-a artifacts/EXP-002/diagnostic_run01/tasks/c09a0454c987525ea4/view_2/frame_000.jpg \
  --image-b artifacts/EXP-002/diagnostic_run01/tasks/865a709c00cebdf52e/view_3/frame_000.jpg \
  --model-path /home/scc/pb22111667/models/Qwen3-VL-8B-Instruct \
  --anchors "the same man wearing the black number-7 jersey (identity anchor only; his position may change)" \
  --output artifacts/EXP-002/diagnostic_run01/pairwise_view_relation_qwen8b.json
```

脚本加载本地 Qwen3-VL-8B，不需要 API key。每次请求完成后即写入 JSON；中断后使用相同参数和输出路径加 `--resume` 继续。若更换图片、模型目录、锚点或生成长度，使用新的输出路径。

若人工确认两图中有可一一对应的固定地标（例如同一球门或同一围栏拐角），应把它加入 `--anchors`。不要把“都是草地/都是围栏”当成精确对应点；若只有球员可匹配，也要注明他会移动，避免把人物位移直接解释成机位变化。

## 人工判读

每个条件只记录三项即可：

| 条件 | 共同实体是否识别正确 | 视角关系是否有静态图像证据 | 顺序交换后结论是否一致 |
|---|---|---|---|
| blind_pair | 是 / 部分 / 否 | 是 / 部分 / 否 / 不适用 | 是 / 否 |
| same_scene_known | 是 / 部分 / 否 | 是 / 部分 / 否 | 是 / 否 |
| verified_anchors | 已提供 | 是 / 部分 / 否 | 是 / 否 |

解释重点是条件之间的变化：若从 blind 到 same-scene 提示后明显改善，说明模型需要场景先验；若只有提供对应物后改善，主要瓶颈可能在跨图实体匹配；若给了对应物仍无法用稳定地标描述关系，才更像空间关系推理困难。单个图像对只能产生案例级证据，不用于统计总体能力或证明多视角 QA 的因果机制。

