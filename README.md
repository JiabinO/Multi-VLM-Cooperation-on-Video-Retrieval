# VLM 多端视频协作研究

研究多端联合视频理解中的通信效率与答案质量。EXP-001 关注协作消息中的冗余与证据保留；当前准备推进的 EXP-002 关注带宽预算下，按查询选择要传输的视角和时间片段。

当前状态：已在 MVVBench 的 EgoExo 与 MMPTrack 上完成 EXP-002 的 8 题流程试运行，验证候选帧提取、Qwen3-VL 打分、预算计划、QA 与离线分析链路。试运行结果和限制记录在 [EXP-002 试运行报告](results/EXP-002/pilot-run01/report.md)。当前查询驱动方法仍是启发式基线；正式扩大评测尚未完成。

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
Notion 用于想法、讨论和可读的结果解释；GitHub 保存代码、版本化配置、数据清单和精选实验结果。两处通过实验编号、运行编号和代码版本相互引用，目前不自动同步。公开仓库文档不包含私人工作区链接。

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
1. 使用场景去重后的扩展问题集（EgoExo 88 题、MMPTrack 16 题）运行候选证据打分与预算扫描。
2. 记录实际配置、代码提交号和逐题输出，保证服务器中断后可恢复并可追溯。
3. 对错误样本做人工复核，再判断查询驱动选择是否稳定优于均匀采样。

首个入口为 scripts/video_smoke.py；依赖与执行方式见 docs/video-smoke.md。多端联合推理尚未实现。

