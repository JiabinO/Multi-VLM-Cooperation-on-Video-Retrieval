# Multi-VLM Cooperation on Video Retrieval

此分支包含 MVVBench 查询驱动视频证据选择实验代码，支持 Ego-Exo4D 与 MMPTrack。数据文件、模型权重和服务器专用路径不在仓库中。

## 服务器运行

先切换到实验分支：

```sh
git fetch origin
git checkout codex/query-driven-video-selection
```

复制 `configs/exp002_budget_sweep.example.json` 为本机配置，填入标注、视频映射、视频根目录和 Qwen3-VL 模型路径。按 `docs/exp-002-query-driven-transmission.md` 中的步骤依次运行相关性打分、预算选择、QA 和分析脚本。

视频路径当前保留占位符，配置完成后再运行。预算按 JPEG 抽帧文件大小计算，是离线传输量代理指标。

## 相关文档

- [MVVBench 服务器数据准备](docs/mvvbench-server-setup.md)
- [查询驱动传输实验设计与运行步骤](docs/exp-002-query-driven-transmission.md)
- [脚本说明](scripts/README.md)
