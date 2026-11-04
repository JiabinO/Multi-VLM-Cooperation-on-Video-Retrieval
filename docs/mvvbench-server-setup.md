# 在远程服务器准备 MVVBench

本项目只在仓库中保存下载与整理代码。MVVBench 标注和视频应在服务器下载到仓库外部的持久数据目录；不要把数据文件提交到 Git。

## 获取代码

首次克隆：

```bash
git clone https://github.com/JiabinO/Multi-VLM-Cooperation-on-Video-Retrieval.git
cd Multi-VLM-Cooperation-on-Video-Retrieval
```

已克隆项目时，切换到默认分支并拉取更新：

```bash
git switch main
git pull origin main
```

## 下载 MVVBench 标注包

在 Hugging Face 账号接受数据集的非商业科研条件后，在服务器登录该账号：

```bash
hf auth login
hf auth whoami
```

随后在项目根目录运行：

```bash
python scripts/download_mvvbench.py \
  --output-dir /path/to/persistent-data/mvvbench
```

输出包括问答标注、3,082 条视频映射、上游视频准备脚本和下载说明。下载器使用 `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`，也会读取已安装 `huggingface_hub` 的登录令牌。

## 准备视频

MVVBench 不分发视频。需按 `video_mapping.csv` 从 Ego-Exo4D、MMPTrack 和 CMU Panoptic Studio 下载对应源视频，并遵守各自的访问和使用条款。将原始文件按映射表 `source_rel_path` 放入各自目录后，安装上游脚本依赖 `pandas` 和 `tqdm`，再先检查映射：

```bash
python /path/to/persistent-data/mvvbench/prepare_videos.py \
  --dataset-root /path/to/persistent-data/mvvbench/dataset \
  --mapping-file /path/to/persistent-data/mvvbench/video_mapping.csv \
  --output-dir /path/to/persistent-data/mvvbench/videos \
  --dry-run
```

检查完整性后，可去掉 `--dry-run` 并增加 `--symlink`，避免复制大视频文件。每次实验记录数据源版本、样本映射清单及代码提交号。
