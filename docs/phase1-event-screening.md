# 基于 Assembly101 事件标注筛选片段与问题

状态：筛选方法已整理；W03–W06 为初筛候选。官方标注的确切帧段尚未并入，
因此本文不声称这些窗口已经与官方事件边界精确对齐，也不把用户的窗口级描述冒充官方标签。

## 目标与数据层次

实验目标是判断一个问题是否需要多视角联合证据，以及增加视角带来多少证据增益。
官方动作标签用于发现、定位和描述事件；它本身不说明某事件在某个摄像头中是否可见。
因此每个候选必须保留两个分开的标注层：

1. **数据集事件层**：序列 ID、粗/细粒度动作标签、30 fps 标注起止帧、映射后的秒数、视角同步与源剪辑偏移。
2. **视角证据层**：每个问题所需的原子事实，在各视角/实际采样帧中记为 `direct`（直接可见）、`partial`（部分可见）、`occluded`（遮挡/投影歧义）、`undersampled`（采样不足）、`out_of_view`（画面外）、`unchecked`（未审核）。窗口级事实不能自动填成每路 `direct`。

Assembly101 官方细粒度标签描述单个动词/动作与交互物体，并可能覆盖接触前、交互中和接触后的阶段；粗粒度标签把连续或相关的细粒度步骤合并为装配/拆卸一个部件的过程。[官方论文](https://assembly-101.github.io/assets/Assembly101.pdf) 官方仓库说明标注帧号以 30 fps 计算，尽管原始视频是 60 fps；细粒度 CSV 包含序列/视角、起止帧、动作、动词、物体与玩具字段，粗粒度标签包含起止帧和动作类。[标注仓库说明](https://github.com/assembly-101/assembly101-annotations)

## 候选生成与筛选规则

### 1. 用官方标签找窗口，不直接用 QA 问句搜索标签

先在 fine-grained `train.csv`、`validation.csv`、`test.csv` 中按目标序列前缀
`nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/` 筛出动作段；
再查同一序列的 coarse assembly/disassembly 标签，了解多步操作的上位事件。
初步词项可包括 remove / detach / screw / unscrew / pick up / put down / attach，
但不得仅凭词面近似就把标注和用户描述等同。重叠细粒度段应允许并存。

把标注帧转换为时间时，用 `t = frame / 30` 作为数据集标注时间基准；然后核对视频 PTS、
原始录像和已抽出五秒 clip 的时间原点/裁剪偏移。项目当前使用的
W03=100、W04=140、W05=180、W06=220 秒是已有数据准备约定，文档标注其未经独立验证，
不能仅因两者看似落入同一区间就宣称对齐。精确 join 前需要读取原始官方标签文件和 clip manifest。

### 2. 将事件拆成问题所需的原子证据

例如“螺丝是否被取下”拆为：对象是螺丝、它起初连接在组件上、之后与组件分离、发生顺序。
“拿起并装入哪个零件”拆成：接近/接触、抓取成功、物体身份、移动、装入目标、最终状态。
事件标签只提供动作类别和时间候选；对象身份、状态变化、颜色/数量等仍需看同步视频人工确认。

### 3. 逐视角编码证据，不把盲区解释成否定

每个“事件 × 原子证据 × camera × sampling profile”填一个可见性状态，并记录支持帧、
人工确认来源和未确定原因。建议分别审核 8 帧和 24 帧设置，因为当前实验已观察到
更高帧率有助于定位短时拆螺丝事件。模型答错不能反向改变可见性金标；无证据/采样不足
不等于“该动作没有发生”。

### 4. 以联合证据增益选择问题

令某题必要证据集合为 `P`。`direct` 计为已覆盖，`partial` 是否计入需按问题预先定义；
对视角集合 `S`，覆盖数为 `coverage(S) = |{p in P: 至少一路 S 对 p 是 direct}|`。
联合视角增益定义为 `gain(S) = coverage(S) - max_{c in S} coverage({c})`。
另报告覆盖比例 `coverage(S)/|P|`。至少有一路单视角无法完整支持问题，且联合增益大于零，
才作为多视角互补主候选。所有视角都同样可答的题可作冗余/负对照；任何视角都无法
凭实验实际采样提供足够证据的问题应淘汰或单列为采样率压力测试。

问题还要避免答案可由先验猜出、问题提示动作答案、答案仅靠常识，以及多道题重复同一个
简单事实。主问题采用开放式或明确要求关键事实的问法；是非题、计数题和强提示题单列诊断。

## W03–W06 初筛表

下表中的动作事件来自本项目已有 QA 配置及用户审核记录，不是从官方标签文件直接读取的结果。
“官方帧段”一栏待下载/读取官方 annotation CSV/TXT 后填写；视角描述也仍需在确定窗口边界后
回到完整 clip 核对。当前 `review.facts.json` 明确说明完整片段未逐路重看，继承的支持帧未经人工重验。

| 窗口（项目秒数约定） | 用户已确认的事件摘要 | 官方 coarse/fine 标签与帧段 | 初步角色 | 候选问题方向 | 主要筛选判断/待核查 |
| --- | --- | --- | --- | --- | --- |
| W03（100–105 s） | 左手持推土机/铲车模型，右手用螺丝刀拧底部红色螺丝；三路可见操作。 | 待官方序列标签核验 | 冗余基线 | “右手拿什么工具、对模型哪个部位做了什么？” | 目前三路答案看起来都能答，适合作为单视角冗余参照，不是互补主结果。C10115 对具体红螺丝身份是否独立可见仍有来源疑问。不要据此宣称摄像头互补。 |
| W04（140–145 s） | 拧松黑色组件螺丝，之后把螺丝取下；历史高采样实验比低采样更容易定位该短动作。 | 待官方序列标签核验 | 时间分辨率/短事件定位 | 若重做 QA，问“拧松后螺丝发生了什么变化？”并要求给出证据时刻；保留旧是非题仅作诊断。 | 用户补充 C10118 的预览草稿未覆盖后续第 4、5 帧；历史 8f 与 24f 观察显示帧率可能是关键因素。必须分别核对事件边界和各路可见性，避免把时间采样增益误判为视角增益。 |
| W05（180–185 s） | C10095 较早时手尚未接触螺丝，因透视与黑色底座重叠；C10115 后续看到拿起红螺丝；C10118 记录拿起红色零件并装入黄色设备。 | 待官方序列标签核验 | 跨阶段互补的最强候选 | 候选：“操作者从桌面取起的零件是什么，之后把它放到哪里？” | 只有在官方段落时间和三路同步后确认不同相机覆盖不同必要阶段，且联合证据覆盖优于任一单路，才保留为互补 QA。数量“7→6”问题此前不理想/已暂缓，不恢复为主问题。C10095 的投影歧义可用于标注 `occluded`，不能当成拿起黑件的证据。 |
| W06（220–225 s） | 用户给出的顺序：车身起初悬空，右手装零件，零件掉到桌面；左手把车放回桌面固定，右手继续装。 | 待官方序列标签核验 | 事件顺序与低帧率压力样本；暂不作主互补题 | 可研究“零件掉落—车身放下—继续安装”的顺序重建，需拆分必要子事实。 | 用户曾指出低采样可能看不出脱手轨迹；各视角独立可见性没有逐步核定。C10118 的“放桌面”对应后段，不与先前掉落冲突。若需要追踪掉落瞬间，先提升/比较时间采样；若实际帧仍不足，题目标成 `undersampled` 并排除主可答性指标。 |

### 当前建议排序

1. **W05 优先验证**：最可能出现跨相机阶段互补，但须由官方段落边界和逐视角证据表证实，不能预先当成结论。
2. **W04 保留为时间采样对照**：重点控制 8/24 帧采样；将“视角数”与“采样密度”作为不同实验因素。
3. **W03 做冗余负对照**：三路视角都可答时，预期多视角不带来必要信息增益。
4. **W06 暂作压力样本/候选排除项**：低采样对短暂掉落事件可见性有限，先标证据充分度再决定是否出题。

## 官方标签接入后的执行清单

- [ ] 获取并保留官方 fine-grained / coarse-annotations 的来源、下载日期、校验和与许可/引用信息。
- [ ] 在本机或服务器只读解析，不把完整官方标注打包进 Git；按 README 的数据约定保存外部数据路径。
- [ ] 精确匹配序列 ID，提取该序列全程粗粒度和 3 个 RGB 视角的细粒度动作段。
- [ ] 将 30 fps 标注帧映射到源视频 PTS，并独立验证 W03–W06 片段偏移；记录映射误差和时基。
- [ ] 依据事件段扩展候选窗口（例如标注边界前后留上下文），窗口不得截掉取前/交互/释放等回答所需阶段。
- [ ] 建立 `event × atomic_fact × camera × sampling_profile` 证据矩阵，人工审核各路及视角联合覆盖。
- [ ] 根据预注册的问题、评分要点和 `gain(S)` 选出互补主问题、冗余负对照与采样率诊断题。
- [ ] 将筛选原因、排除项、人工审核人及标签版本记入版本化 manifest；问题在查看新模型输出前冻结。

## 标注目录和外部执行提示

官方说明给出的目标 coarse 文件名是：
`assembly_nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724.txt`
和对应 `disassembly_...txt`；fine-grained 则从 CSV 的 `video` 列按相同 sequence ID 匹配。
官方主页现已公告数据于 2026 年 5 月迁移到 Hugging Face；官方旧版标注 README 仍保留 Google Drive 链接。
当前服务器下载已返回 `Cannot access gated repo` / `not in the authorized list`，所以该 HF 仓库确实要求账号授权。
本工作区未下载标注，因此**尚未拿到 W03–W06 的官方动作类和确切帧号**。

### 服务器命令行获取（推荐）

先在 Hugging Face 网页登录下载所用账号，打开 [Assembly101 数据集页面](https://huggingface.co/datasets/cvml-nus/assembly101)，
提交访问申请/同意条款；若需要作者审批，需等待账号进入授权列表。授权按个人账号生效。
获准后在服务器仓库根目录检查 CLI 当前身份，再下载（无需安装或修改 Conda 包）：

```bash
ANNOT_DIR=artifacts/phase1/official_annotations
mkdir -p "$ANNOT_DIR"
hf auth whoami
hf download cvml-nus/assembly101 \
  annotations/fine-grained-annotations/train.csv \
  annotations/fine-grained-annotations/validation.csv \
  annotations/fine-grained-annotations/test.csv \
  annotations/coarse-annotations/coarse_labels/assembly_nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724.txt \
  annotations/coarse-annotations/coarse_labels/disassembly_nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724.txt \
  --repo-type dataset --local-dir "$ANNOT_DIR"
```

细粒度 CSV 当前合计约 184 MB；这是标注表，不是 72 GB 的视频数据集。文件下载后运行：

若 `hf auth whoami` 显示的用户名不是已获准账号，在服务器运行 `hf auth login` 并登录获准账号。
错误 URL 若使用 `hf-mirror.com`，取得授权后可临时对官方域名重试：
`HF_ENDPOINT=https://huggingface.co hf download ...`。若服务器无法连接官方域名，再继续使用镜像；镜像不能绕过 HF 账号授权。

```bash
python scripts/phase1_annotations.py \
  --session-id nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724 \
  --fine-csv "$ANNOT_DIR/annotations/fine-grained-annotations/train.csv" \
            "$ANNOT_DIR/annotations/fine-grained-annotations/validation.csv" \
            "$ANNOT_DIR/annotations/fine-grained-annotations/test.csv" \
  --coarse-label "$ANNOT_DIR/annotations/coarse-annotations/coarse_labels/assembly_nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724.txt" \
                "$ANNOT_DIR/annotations/coarse-annotations/coarse_labels/disassembly_nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724.txt" \
  --output artifacts/phase1/annotation_import_run01
```

若 `hf` 命令不存在，先检查现有环境是否有 Python 客户端：
`python -c "import huggingface_hub; print(huggingface_hub.__version__)"`。不要为此直接修改服务器 Conda 环境；可改在已登录的本机下载这五个文件，再复制到 `artifacts/phase1/official_annotations/`。

从官方标注压缩包解压后，可先在仓库外部的数据目录查找目标序列：

```bash
find /path/to/assembly101-annotations \
  -type f \( -name 'train.csv' -o -name 'validation.csv' -o -name 'test.csv' \
  -o -name 'assembly_nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724.txt' \
  -o -name 'disassembly_nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724.txt' \)
```

仓库现已提供 `scripts/phase1_annotations.py`（仅 Python 标准库），把上述机械步骤自动化：
按精确序列 ID 过滤三路 fine-grained CSV，读取同序列 coarse label，保留原始 30 fps 帧号，
计算与现有 W03–W06 时间约定的候选交集，并输出源文件 SHA256、机器可读事件清单、
窗口候选表和 8/24 帧 × 三摄像头的 `unchecked` 证据审核表。候选交集明确标记为未验证，
不会自动确认 PTS/剪辑偏移或填入任何问题事实。

在服务器仓库根目录、激活现有 `vlm_redundancy` 环境后运行（无需安装新依赖）：

```bash
python scripts/phase1_annotations.py \
  --session-id nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724 \
  --fine-csv /path/to/fine-grained-annotations/train.csv \
            /path/to/fine-grained-annotations/validation.csv \
            /path/to/fine-grained-annotations/test.csv \
  --coarse-label /path/to/coarse-annotations/coarse_labels/assembly_nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724.txt \
                /path/to/coarse-annotations/coarse_labels/disassembly_nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724.txt \
  --output artifacts/phase1/annotation_import_run01
```

输出目录必须是新目录。重点文件为 `annotation_manifest.json`、
`event_window_candidates.csv` 和 `evidence_review_template.csv`。最后一个表只需人工看视频填
`required_fact_unit`、可见性、支持帧和备注；官方动作字段与时间换算不需要手工抄录。

拿到文件后应生成只含目标序列的派生清单（本项目自己的 `data/manifests/`），不提交整套
官方标注/视频；清单保留原始 frame、转换秒数、视角、标签文本及源文件校验和。

## 官方来源

- [Assembly101 官方主页](https://assembly-101.github.io/) 与 [Hugging Face 数据集](https://huggingface.co/datasets/cvml-nus/assembly101)：2026 年迁移公告及当前 gated 下载入口。
- [Assembly101 annotations README](https://github.com/assembly-101/assembly101-annotations)：标注层级、字段、序列文件命名及标注帧率。
- [Assembly101 论文](https://assembly-101.github.io/assets/Assembly101.pdf)：细粒度/粗粒度事件定义及前接触、交互、后接触的阶段。
- [Assembly101 mistake-detection annotations](https://github.com/assembly-101/assembly101-mistake-detection)：若后续要筛程序顺序错误，可额外查该任务的专用错误标签；它不能代替常规动作标注或逐视角可见性审核。
