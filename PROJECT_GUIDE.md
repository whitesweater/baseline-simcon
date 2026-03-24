# Baseline Project Guide

这份文档是 `baseline/` 仓库的内部总指南。

目标不是介绍论文，而是让后续接手的人在不依赖聊天记录的情况下，快速搞清楚：

- Git 仓库边界是什么
- `CODI/`、`Coconut/`、`CODI_rebuttal_runs/` 之间是什么关系
- 论文里的方法名和代码里的命名如何对应
- 哪些目录是正式实验入口，哪些只是历史遗留或外部参考
- 哪些结果可信，出现冲突时应该优先相信什么
- rebuttal / revision 阶段默认怎么继续工作

如果你是第一次接手这个仓库，建议阅读顺序如下：

1. 本文档 `PROJECT_GUIDE.md`
2. 根目录 `README.md`
3. `CODI/README.md`
4. `CODI/TESTING_GUIDE.md`
5. `CODI/train.py`
6. `CODI/src/model.py`
7. `CODI/scripts/`、`CODI/train_on_*`、`Coconut/run.py`

---

## 1. 仓库边界与目录关系

### 1.1 Git 根目录

真正的 Git 仓库根目录是：

```bash
/data/yhao/baseline
```

这意味着：

- `CODI/` 不是独立仓库，而是 `baseline/` 的一个子目录
- `Coconut/` 也是同一仓库中的子目录
- 所有新分支都应从 `baseline/` 根目录创建
- 所有 `git status`、`git add`、`git commit` 都默认以 `baseline/` 为边界理解

### 1.2 当前最重要的目录

#### `CODI/`

这是当前最主要的研究与实验目录。论文中与 CODI、SIM-CoT、SIRCL 相关的大部分训练脚本、评测脚本、分析脚本和历史结果都在这里。

#### `Coconut/`

这是同一仓库中的另一条 backbone 代码线，不在 `CODI/` 内部。

当前 rebuttal / revision 阶段，`Coconut/` 不是只读摆设，而是当前工作范围的一部分。如果需要做 cross-backbone 对照、补实验、对齐 SIRCL 叙述或核对论文主表，默认可以把它作为活跃范围来处理。

#### `CODI_rebuttal_runs/`

这是 revision 阶段的新实验输出根目录。它的作用是把 2026-03-25 起的新运行结果，与 `CODI/outputs`、`CODI/results` 下的历史产物隔离开。

当前默认实验根目录是：

```bash
/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325
```

#### `CODI/local_datasets/`

这是 CODI 当前运行时真正使用的本地 JSON 数据目录。当前保留并实际使用的数据包括：

- `coin_flip`
- `multiarith`
- `svamp`

这些文件来源于 `CODI/SemCoT/` 的数据拷贝，但现在应将 `local_datasets/` 视为运行入口。

#### `CODI/SemCoT/`

这是一个外部参考仓库的完整拷贝。

当前原则：

- 它不是 CODI 主逻辑的一部分
- 不应继续要求 CODI 在运行时依赖整个 `SemCoT/` 仓库
- 我们主要参考它的数据文件和部分数据处理思路
- 后续主开发提交不应把它当作活跃代码线来维护

#### 历史可信目录

如果需要回看论文已有结论，优先参考：

- `CODI/results_useful/`
- `CODI/plots/` 及其读取的数据源
- `CODI/final_use_model_codi_sim_sircl/`

---

## 2. 当前方法映射

### 2.1 论文名与代码名的对应关系

| 代码名 | 论文名 | 含义 |
| --- | --- | --- |
| `codi` | CODI | 不带 SIRCL 的 CODI backbone |
| `codi_sircl` | CODI + SIRCL | CODI 上加 SIRCL 插件 |
| `simcon` | SIM-CoT | 不带 SIRCL 的 SIM-CoT backbone |
| `simcon_sircl` | SIM-CoT + SIRCL | SIM-CoT 上加 SIRCL 插件 |
| `coconut` | Coconut | `Coconut/` 目录中的 backbone |
| `coconut cot` | Coconut CoT 变体 | 仍位于 `Coconut/` 线 |

其中最容易误解的一点是：

> 代码里的 `simcon`，就是论文里的 SIM-CoT 方法线。

不要把它当作随手取的内部简称。

### 2.2 SIRCL 的定位

SIRCL 在本项目中的定位是：

> 一个可插拔的训练期稳定器插件，通过额外的 trajectory consistency loss 约束 latent trajectory。

它可以加在多个 backbone 上，包括：

- CODI
- SIM-CoT
- Coconut

因此在论文或实验口径中，应把 SIRCL 理解为统一插件，而不是只绑定某一个 backbone。

### 2.3 常见脚本开关

脚本层最常见的 SIRCL 相关参数通常是：

```bash
--use_trajectory_consistency True
--trajectory_space_type euclidean
--trajectory_radius_threshold 2
--trajectory_loss_factor 0.2
```

一个实用的工程近似是：

| 组合 | 方法理解 |
| --- | --- |
| `use_decoder=False`, `use_trajectory_consistency=False` | `codi` |
| `use_decoder=False`, `use_trajectory_consistency=True` | `codi_sircl` |
| `use_decoder=True`, `use_trajectory_consistency=False` | `simcon` |
| `use_decoder=True`, `use_trajectory_consistency=True` | `simcon_sircl` |

这个映射是当前项目里的有效工程理解，后续看脚本、命名和结果时可以优先按它来判断。

---

## 3. 训练与评测口径

### 3.1 CODI 主训练口径

当前默认主训练口径是：

- 以 `GSM8K-Aug / icot` 为主训练集
- 将 `SVAMP / MultiArith / GSM-Hard` 作为 OOD 评测

### 3.2 `icot` 的权威入口

对于这个项目来说：

> `icot` 的权威入口就是缓存，而不是临时从 Hugging Face 重新构建。

如果看到 `train.py` 在 `icot` 分支上依赖 cache，这是正式用法，不应把它视为临时 hack。

### 3.3 单独训练结果的地位

下面两个目录中的单独训练结果，也是论文正式结果的一部分：

- `CODI/train_on_svamp_dataset/`
- `CODI/train_on_multiarith_dataset/`

不要把“正式实验”狭义理解成只包含 `icot -> OOD eval` 这条主线。

### 3.4 当前默认工作重心

当前默认工作重心不是单独做文档清理，而是：

> 围绕 rebuttal / revision 做 cross-backbone 的新实验、核对和补充说明。

这意味着：

- `CODI/` 仍是主实验目录
- `Coconut/` 也在当前工作范围内
- 处理 reviewer 问题时，默认可以准备跨 backbone 的对照或补实验

---

## 4. 证据优先级与可信来源

### 4.1 出现冲突时优先相信什么

如果文档、历史结果、脚本实现、旧目录中的指标互相冲突，默认按下面顺序判断：

1. 论文最终正文、最终表格、最终定稿口径
2. 与论文一致的可信历史产物
3. 本文档 `PROJECT_GUIDE.md` 和当前 handover 约定
4. 旧文档、零散历史输出、未核对的 `results/` 目录内容

也就是说：

> 论文最终口径优先，可信产物用于支撑；不要反过来让零散旧结果推翻论文定稿结论。

### 4.2 当前可信的历史依据

优先信这些：

- `CODI/results_useful/`
- `CODI/plots/` 中的分析脚本及其读取的数据源
- `CODI/final_use_model_codi_sim_sircl/`
- 与最终论文一致的结果表和 checkpoint

### 4.3 `CODI/results/` 如何看待

`CODI/results/` 里混有历史运行、调试试验、中间失败运行和旧配置遗留。

实务上应这样处理：

- 不要默认其中所有内容都可信
- 如果某个目录和论文表格对不上，先视为历史中间产物
- 遇到异常分数，不要优先解释它，先判断它是否是调试或失败运行

---

## 5. Rebuttal / Revision 工作原则

### 5.1 新实验输出隔离

从 2026-03-25 起，新的实验默认不再写入：

- `CODI/outputs/`
- `CODI/results/`

默认应写入：

```bash
CODI_RUN_ROOT=/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325
CODI_SAVE_DIR=${CODI_RUN_ROOT}/outputs
CODI_RESULT_DIR=${CODI_RUN_ROOT}/results
```

这样做的目的：

- 避免污染历史结果
- 明确区分 revision 阶段的新运行与旧论文产物
- 方便把代码提交与实验记录对应起来

### 5.2 Git 与提交边界

默认 Git 工作流：

1. 在 `baseline/` 根目录切分支
2. 修改前先确认 `git status`
3. 以小批次逻辑块推进
4. 先做最小验证，再提交
5. 代码、脚本、必要说明文档可以提交
6. checkpoint、logs、results、大型生成物不要提交

### 5.3 SemCoT 的处理原则

`CODI/SemCoT/` 视为外部参考目录：

- 不作为主开发内容提交
- 不作为运行时前提依赖
- 如果只是需要数据文件，优先维护 `CODI/local_datasets/`

### 5.4 外部 reviewer 输入

除仓库内文档外，外部 reviewer / rebuttal 任务清单也是正式输入之一。

当前状态：

- 这份输入尚未以固定路径沉淀在仓库中
- 后续拿到明确文件或路径后，应与本文档一起作为工作入口
- 在它落盘之前，以当前对话中已确认的 reviewer 方向为默认口径

---

## 6. 代码入口与子项目文档

### 6.1 CODI 入口

如果要进入 CODI 主线，优先看：

- `CODI/README.md`
- `CODI/TESTING_GUIDE.md`
- `CODI/train.py`
- `CODI/src/model.py`
- `CODI/scripts/`

`CODI/PROJECT_GUIDE.md` 和 `CODI/REBUTTAL_WORKSPACE.md` 现在只保留为导读页，用来把旧入口引回本根级总指南。

### 6.2 Coconut 入口

`Coconut/` 当前没有单独的 README 入口页。需要进入 Coconut 线时，优先从这些文件建立上下文：

- `Coconut/run.py`
- `Coconut/coconut.py`
- `Coconut/trajectory_consistency.py`
- `Coconut/TRAJECTORY_CONSISTENCY.md`

---

## 7. 最短接手路线

如果是新接手的人，建议直接按下面顺序开始：

### 第一步：确认仓库边界

```bash
cd /data/yhao/baseline
git status
```

先记住：Git 根目录在 `baseline/`，不在 `CODI/`。

### 第二步：读总入口

按顺序看：

1. `README.md`
2. `PROJECT_GUIDE.md`
3. `CODI/README.md`

### 第三步：确认运行环境

```bash
cd /data/yhao/baseline/CODI
source config.env
```

重点确认这些变量：

- `CODI_RUN_ROOT`
- `CODI_SAVE_DIR`
- `CODI_RESULT_DIR`
- `CODI_MULTIARITH_PATH`
- `CODI_SVAMP_PATH`
- `CODI_COIN_FLIP_PATH`

### 第四步：理解方法映射

先只记下面这张表：

| 名字 | 含义 |
| --- | --- |
| `codi` | CODI backbone |
| `codi_sircl` | CODI + SIRCL |
| `simcon` | SIM-CoT backbone |
| `simcon_sircl` | SIM-CoT + SIRCL |
| `coconut` | `Coconut/` 中的 backbone |

### 第五步：再进入脚本和模型

优先看：

- `CODI/train.py`
- `CODI/src/model.py`
- `CODI/scripts/`
- `Coconut/run.py`

到这里，基本就能在当前仓库里继续做 rebuttal 工作了。
