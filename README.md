# VLM 多端视频协作研究

研究多端联合视频理解中的通信效率与答案质量。EXP-001 关注协作消息中的冗余与证据保留；当前准备推进的 EXP-002 关注带宽预算下，按查询选择要传输的视角和时间片段。

当前状态：MVVBench 的 Ego-Exo4D 与 MMPTrack 数据已由用户下载。EXP-002 已有 MVVBench 数据适配、候选帧提取、Qwen3-VL 相关性打分、预算计划、QA 运行和离线分析入口；真实数据运行仍待服务器路径配置。

## 从哪里开始
- [逐帧比分读取与人工参考答案对照](docs/scoreboard-probe.md)
- [单路视频测试与运行命令](docs/video-smoke.md)
- [研究与首轮实验](docs/experiment-plan.md)
- [EXP-002 查询驱动传输选择](docs/exp-002-query-driven-transmission.md)
- [EXP-002 预算选择与分析脚本](scripts/README.md)
- [本地开发、GitHub 与服务器运行](docs/workflow.md)
- [目录与存储约定](docs/repository-layout.md)
- [运行与结果记录](results/README.md)
- [实验报告模板](docs/templates/experiment-report.md)
- [论文阅读记录库](data/reading_library/README.md)
- [Notion 新研究区](https://app.notion.com/p/40282682621581c9b750f4a08e1d3cf4)

Notion 用于想法、讨论和可读的结果解释；GitHub 保存代码、版本化配置、数据清单和精选实验结果。两处通过实验编号、运行编号和代码版本相互引用，目前不自动同步。

## 目录
```text
docs/                  研究计划、流程、决策和报告模板
configs/               可提交的实验配置
data/manifests/        数据版本与样本清单，不放原始视频
data/reading_library/  论文索引、研究卡片与阅读方法
src/                   后续 Python 实现
scripts/               本地检查、服务器启动及结果整理入口
tests/                 后续关键功能测试
results/<experiment_id>/<run_id>/
                       精选运行元信息、指标、小型逐题结果与报告
artifacts/             本机或服务器的大型输出，不进 Git
```

## 当前下一步
1. 检查一个多摄像头数据源的小样本与标注。
2. 确定端点与中心模型、结构化证据格式。
3. 跑通少量样本并完整记录消息、答案和成本。
4. 人工核查重复、互补与独立确认，再比较删除前后结果。

首个入口为 scripts/video_smoke.py；依赖与执行方式见 docs/video-smoke.md。多端联合推理尚未实现。

