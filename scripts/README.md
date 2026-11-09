# 实验脚本

查询驱动视频证据实验的运行顺序如下：

1. `score_video_evidence.py`：从 MVVBench 问题和映射中构造候选片段、抽取 JPEG 帧，并用配置的 VLM 计算相关性分数。
2. `select_video_evidence.py`：按预算生成全量输入、均匀时间采样、固定视角采样和查询驱动选择方案。
3. `run_video_qa.py`：对各方案选出的证据运行 QA 并记录预测和传输字节代理值。
4. `analyze_video_qa.py`：生成按策略和数据域汇总的准确率、字节量及预算利用率。

可复制运行的命令、输入输出说明及指标限制见 [实验说明](../docs/exp-002-query-driven-transmission.md)。视频和模型路径请在本地配置文件中填写。
