# 本地开发 → GitHub → 服务器实验

以下命令为后续工作约定；当前没有模型推理脚本，运行示例不会假装已经可用。

## 1. 本地获取代码
```sh
git clone https://github.com/JiabinO/Multi-VLM-Cooperation-on-Video-Retrieval.git
cd Multi-VLM-Cooperation-on-Video-Retrieval
git switch -c feat/exp-001-baseline
```
私有仓库需在本地和服务器分别配置 GitHub 访问权限；插件连接不会自动赋予服务器凭据。不要把令牌写入仓库或远程 URL。

修改代码、配置和实验说明，完成相应检查后：
```sh
git add src configs scripts tests docs
git commit -m "feat: add EXP-001 baseline"
git push -u origin feat/exp-001-baseline
```
只添加实际存在且需要提交的路径。通过 PR 检查后合并到 main，或明确选定实验分支；不要把未合并分支误认为服务器 main 已包含。

## 2. 服务器固定代码版本
首次使用同样的 clone 命令。已有工作目录中，先检查未提交改动；不要强行覆盖：
```sh
git status --short
git fetch origin
git switch --detach <commit_sha>
git rev-parse HEAD
```
将占位符替换为选定的完整提交 SHA；它可以来自已合并 main 或明确指定的实验分支。
使用固定版本运行，避免实验过程中拉取更新；新版本另开运行目录。
环境配置、锁定依赖和启动命令随首个实现加入；服务器数据路径通过本机配置或环境变量提供。

## 3. 启动前记录
- 实验编号、唯一运行编号、完整 commit SHA、工作区是否干净。
- 实际使用的配置副本与提示词，不能只记一个会变化的文件名。
- 模型与 tokenizer 版本、依赖版本、硬件和运行命令。
- 数据版本、样本清单与划分、随机种子、评价规则。
- 开始时间、时区，以及输出目录。

当前默认建议只用干净工作区运行；若确需带改动运行，保存差异文件并明确标注，不能仅靠 SHA 声称可复现。

## 4. 服务器整理结果并上传
运行输出先写入不入库的 artifacts/<experiment_id>/<run_id>/。
运行结束后，筛选小型记录进入 results/<experiment_id>/<run_id>/，保留失败状态和缺失样本说明。大文件留在持久存储并登记索引。

完成运行后再创建结果分支：
```sh
git switch -c results/exp-001-run-001
git add results/EXP-001/<run_id>
git commit -m "results: record EXP-001 run"
git push -u origin results/exp-001-run-001
```
将 run_id 替换为实际目录。结果提交是新的提交；元信息中的 code_commit 应继续指向执行代码版本，而不是结果提交。

## 5. 本地分析与 Notion
结果分支审阅合并后，本地干净的 main 可执行 git pull --ff-only。
Notion 报告引用实验编号、运行编号、代码 SHA 和 GitHub 结果路径，解释为什么做、看到了什么及下一步。
修改统计或评分逻辑后要保留分析版本，重新生成报告，不能悄悄替换旧数字。

