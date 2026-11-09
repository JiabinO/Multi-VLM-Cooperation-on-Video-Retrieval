# 运行与计算结果

本目录保存精选、可审阅的小型计算结果。EXP-002 已收录一份 8 题流程试运行记录；服务器原始产物仍保存在服务器，不纳入 Git。

- [EXP-002 试运行报告](EXP-002/pilot-run01/report.md)：8 题预算扫描的汇总结果、限制和后续计划。

## 每次运行目录
```text
results/EXP-001/<run_id>/
  run.json          执行元信息与状态
  config.json       实际配置快照（或 YAML）
  metrics.json      指标、样本数、分组与单位
  predictions.jsonl 小型逐题结果；较大时改用外部索引
  artifacts.json    大型产物的持久地址、大小、校验值
  report.md         用实验报告模板解释结果
```
文件名是约定，不要求空造输出。失败运行没有指标时，记录失败原因和已完成样本，不能填零冒充真实测量。

## run.json 的必要信息
experiment_id、run_id、status、code_commit、git_dirty、started_at、finished_at、command、config_path、dataset_manifest、model_revision、seed、environment、hardware。
时间包含时区；状态区分 running、completed、failed、interrupted。
code_commit 指向实际执行的代码版本，结果后续提交另记，不混为一谈。
指标解释分母、评分规则和缺失样本；重跑使用新 run_id。

## 结果上传
先检查文件规模、数据共享条件和凭据，精选后明确添加具体运行目录。完整视频、特征、模型、缓存和原始大日志保存在外部。
报告链接逐题证据与外部产物；外部文件校验值用于确认版本，不意味着文件已上传或备份。

