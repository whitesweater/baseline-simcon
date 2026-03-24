# CODI Project Guide

这份文档是 CODI 项目的总指南，目标是让任何新接手的人在不依赖聊天记录的情况下，快速搞清楚：

- 仓库边界是什么
- 论文里各个方法和代码里的名字如何对应
- 哪些目录是正式实验入口，哪些只是参考或历史遗留
- 哪些结果可信，哪些结果不要作为最终依据
- 以后应该如何继续做实验、管理输出、使用 Git

如果你是第一次接手这个项目，建议阅读顺序如下：

1. 本文档 `PROJECT_GUIDE.md`
2. `README.md`
3. `REBUTTAL_WORKSPACE.md`
4. `train.py`
5. `src/model.py`
6. `scripts/` 和 `train_on_*` 目录中的实验脚本

---

## 1. 仓库边界与目录关系

### 1.1 真正的 Git 仓库根目录

**注意：Git 仓库根目录不是 `CODI/`，而是：**

```bash
/data/yhao/baseline
```

也就是说：

- `CODI/` 是 `baseline` 仓库中的一个子目录
- `Coconut/` 也是同一个 `baseline` 仓库中的一个子目录
- 所有新分支都应该在 `baseline` 根目录创建
- 所有 `git status / git add / git commit` 都要默认从 `baseline` 根目录理解

### 1.2 本项目中最重要的几个子目录

#### `CODI/`
本项目主体。当前论文的大多数代码、脚本、分析和图表都在这里。

#### `Coconut/`
Coconut 相关代码不在 `CODI/` 内，而是在：

```bash
/data/yhao/baseline/Coconut
```

如果论文主表需要包含 `coconut / coconut cot`，要去 `Coconut/` 目录看，不要误以为它们在 `CODI/` 里。

#### `CODI/local_datasets/`
这是 **CODI 实际运行时使用的本地 JSON 数据目录**。

当前实际保留的数据有：

- `coin_flip`
- `multiarith`
- `svamp`

这些文件是从外部参考仓库 `CODI/SemCoT/` 中拷贝出来的，但今后默认应以 `local_datasets/` 为运行入口。

#### `CODI/SemCoT/`
这是一个**外部参考仓库的完整拷贝**。

当前原则：

- 它不是 CODI 主逻辑的一部分
- 我们没有使用它的训练逻辑作为主实验入口
- 我们主要参考了它的数据文件和部分数据处理思路
- 运行 CODI 时，不应继续依赖整仓 `SemCoT` 才能工作

因此：

- `SemCoT/` 被视为外部参考目录
- 不应纳入后续主开发提交
- 实际运行所需的数据已经转移到 `local_datasets/`

#### `CODI/results_useful/`
这是一个**可信的历史结果目录**。当你需要回看论文已有实验结论时，应优先参考这里。

#### `CODI/plots/`
这是论文图表与分析脚本目录。这里面的脚本，以及它们所使用的源文件，被认为是可信的分析来源。

#### `CODI/final_use_model_codi_sim_sircl/`
这是已经确认的**最终论文模型 checkpoint 目录**。如果需要对照最终论文模型，优先从这里找。

---

## 2. 论文方法与代码命名对应关系

### 2.1 当前论文主表的核心方法

当前论文主表核心看这四类：

- `codi`
- `codi_sircl`
- `simcon`
- `simcon_sircl`

除此之外，还有：

- `coconut`
- `coconut cot`

但这两类在 `Coconut/` 目录，不在 `CODI/` 目录内。

### 2.2 代码中的 `simcon` 是什么

**代码里的 `simcon`，等价于论文里的 SIM-CoT 方法线。**

不要把它当成一个随便取的内部简称。后续阅读代码和看结果时，可以直接按下面映射理解：

| 代码名 | 论文名 | 说明 |
|---|---|---|
| `codi` | CODI | 不带 SIRCL 的 CODI backbone |
| `codi_sircl` | CODI + SIRCL | 在 CODI 上加 SIRCL 插件 |
| `simcon` | SIM-CoT | 不带 SIRCL 的 SIM-CoT backbone |
| `simcon_sircl` | SIM-CoT + SIRCL | 在 SIM-CoT 上加 SIRCL 插件 |
| `coconut` | Coconut | 在 `Coconut/` 目录 |

### 2.3 SIRCL 在项目中的定位

SIRCL 的定位是：

> 一个可插拔的训练期稳定器插件，通过额外的 trajectory consistency loss 约束 latent trajectory。

它可以加在多个 backbone 上：

- CODI
- SIM-CoT
- Coconut

也就是说，后续阅读论文时，应将 SIRCL 理解为：

> “给 CODI / SIM-CoT / Coconut 都可插拔的统一稳定器”

而不是只绑定某一个 backbone。

### 2.4 SIRCL 的核心开关和参数

在脚本层，SIRCL 通常对应以下参数：

```bash
--use_trajectory_consistency True
--trajectory_space_type euclidean
--trajectory_radius_threshold 2
--trajectory_loss_factor 0.2
```

其中：

- `use_trajectory_consistency=True` 表示启用 SIRCL
- `trajectory_space_type` 表示几何空间类型，目前主线通常用 `euclidean`
- `trajectory_radius_threshold` 是半径阈值
- `trajectory_loss_factor` 是 SIRCL loss 的权重

---

## 3. 代码主干应该怎么理解

### 3.1 训练主入口

训练主入口是：

```bash
CODI/train.py
```

它负责：

- 解析训练参数
- 加载数据
- 组装 `Trainer`
- 调用 `src/model.py` 中的主模型

### 3.2 模型主实现

模型核心在：

```bash
CODI/src/model.py
```

这里实现了：

- latent loop
- teacher/student distillation
- answer CE loss
- teacher reference CE loss
- explain/decoder 路径（若启用）
- trajectory consistency / acceleration / action / geodesic / rank diversity 等额外 loss

### 3.3 一条最重要的理解线

这个项目可以先按下面这个框架理解：

1. question 先进入模型，得到 hidden state seed
2. hidden state seed 经过若干次 latent loop，生成隐式推理轨迹
3. 最后使用 latent state 去生成 answer
4. 训练时除了 answer CE，还会加入 teacher reference loss 和 distillation loss
5. 若启用 SIRCL，再额外加 trajectory consistency loss

### 3.4 `use_decoder` 的含义

在当前项目语境里，可以先这样理解：

- `use_decoder=False`：更偏 CODI 主线
- `use_decoder=True`：更偏 SIM-CoT / step-level supervision 这条线

这不是理论上唯一的定义，但作为当前项目的工程理解，这个近似是正确且足够有用的。

---

## 4. 数据集与训练/评测口径

### 4.1 主训练口径

当前项目的主训练设定是：

- 以 `GSM8K-Aug / icot` 为主训练集
- 将 `SVAMP / MultiArith / GSM-Hard` 作为 OOD 评测

### 4.2 `icot` 数据的权威入口

对于这个项目来说：

> `icot` 的权威入口就是缓存，而不是临时从 HF 重新构建。

也就是说，如果你看到 `train.py` 在 `icot` 分支上依赖 cache，这是项目的正式用法，不是临时 hack。

### 4.3 单独训练的 `svamp` / `multiarith` 结果是否正式

是的。

下面这两个目录中的单独训练结果，也属于论文正式结果的一部分：

- `CODI/train_on_svamp_dataset/`
- `CODI/train_on_multiarith_dataset/`

不要误以为只有 `icot -> OOD eval` 这条主线才算正式实验。

### 4.4 当前本地 JSON 数据入口

当前默认本地数据入口是：

- `CODI/local_datasets/coin_flip/*.json`
- `CODI/local_datasets/multiarith/*.json`
- `CODI/local_datasets/svamp/*.json`

对应代码位置：

- `CODI/train.py`
- `CODI/test_multi_dataset.py`
- `CODI/config.env`

---

## 5. 实验脚本目录应该怎么用

项目主要通过脚本运行，不是手敲一长串命令。

最重要的脚本目录有：

- `CODI/scripts/`
- `CODI/train_on_svamp_dataset/`
- `CODI/train_on_multiarith_dataset/`
- `CODI/train_on_commen_dataset/`
- `CODI/flip/`

### 5.1 `scripts/`

这里主要放：

- 主线训练脚本
- 参数消融脚本
- latent sweep 测试脚本
- batch test 脚本
- 一些图表和工具脚本

### 5.2 `train_on_svamp_dataset/`

放 SVAMP 单独训练的脚本。

### 5.3 `train_on_multiarith_dataset/`

放 MultiArith 单独训练的脚本。

### 5.4 `train_on_commen_dataset/`

放 CommonSenseQA 单独训练的脚本。

### 5.5 `flip/`

放 Coin Flip 相关脚本。

### 5.6 如何判断一个脚本属于哪条方法线

一个非常实用的经验是看两个开关：

- `use_decoder`
- `use_trajectory_consistency`

可以先按下面理解：

| 组合 | 方法理解 |
|---|---|
| `use_decoder=False`, `use_trajectory_consistency=False` | `codi` |
| `use_decoder=False`, `use_trajectory_consistency=True` | `codi_sircl` |
| `use_decoder=True`, `use_trajectory_consistency=False` | `simcon` |
| `use_decoder=True`, `use_trajectory_consistency=True` | `simcon_sircl` |

---

## 6. 哪些结果可信，哪些不要乱信

### 6.1 当前可信的历史依据

如果你要回看已有结论，优先信这些：

- `CODI/results_useful/`
- `CODI/plots/` 中的脚本及其使用的数据源
- `CODI/final_use_model_codi_sim_sircl/`
- 论文正文和最终定稿对应的结果表

### 6.2 `results/` 目录如何使用

`CODI/results/` 里有很多历史运行、调试试验、失败结果和中间产物。

结论：

- **不要默认 `results/` 里所有内容都可信**
- 如果某个目录和论文表格对不上，不要强行相信它
- 若出现冲突，以论文最终标准为准

### 6.3 一个实务原则

如果你看到某些目录分数特别奇怪，或者明显不像论文结果，不要先解释它，先把它视为：

> 可能是调试运行 / 中间失败运行 / 旧配置遗留

---

## 7. 从 2026-03-25 开始的新实验原则

### 7.1 新实验必须与历史结果隔离

从 2026-03-25 起，新的实验不要再写到旧目录：

- 不写到 `CODI/outputs/`
- 不写到 `CODI/results/`

新的默认实验根目录是：

```bash
/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325
```

对应环境变量：

```bash
CODI_RUN_ROOT=/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325
CODI_SAVE_DIR=${CODI_RUN_ROOT}/outputs
CODI_RESULT_DIR=${CODI_RUN_ROOT}/results
```

### 7.2 为什么要这样做

原因很简单：

- 避免新实验污染旧结果
- 避免 rebuttal / revision 期间把历史结论和新尝试混在一起
- 方便以后快速定位“这是不是 revision 阶段的结果”
- 方便做 Git 和实验记录对应

### 7.3 旧结果如何保留

旧结果不删除，但默认视为“历史层”。

新的 revision 阶段结果，统一进入新的 run root。

---

## 8. Git 工作原则

### 8.1 分支在哪里建

一定在下面这个仓库根目录建分支：

```bash
/data/yhao/baseline
```

不要把 `CODI/` 当成独立 Git 仓库。

### 8.2 当前约定的提交原则

只提交：

- 代码
- 脚本
- 必要说明文档

不要提交：

- checkpoints
- logs
- results
- 大型生成文件
- 临时分析图

### 8.3 `SemCoT/` 的 Git 处理原则

`CODI/SemCoT/` 视为外部参考目录：

- 不作为主开发内容提交
- 不依赖其完整仓库逻辑作为运行前提
- 如果只是需要数据文件，优先放到 `local_datasets/`

### 8.4 推荐的 Git 工作流

1. 在 `baseline` 根目录切分支
2. 在代码修改前先确认 `git status`
3. 做一个清晰的小批次改动
4. 先跑最小验证
5. 每个逻辑块单独 commit
6. 不把实验结果混进代码提交里

---

## 9. 最值得信任的 checkpoint 在哪里

如果你需要回看已经认定的最终论文模型，优先看：

```bash
/data/yhao/baseline/CODI/final_use_model_codi_sim_sircl
```

这里目前已被视为：

> 已认定的最终论文模型目录

后续如果你要做：

- 对照复现实验
- 新 reviewer 实验的基准比较
- 从论文最终模型继续测试

优先从这里出发，而不是随便从历史 `outputs/` 里找一个 checkpoint。

---

## 10. 当前项目中最重要的几个“工程事实”

下面这些不是建议，而是**当前项目事实**：

1. Git 根目录是 `/data/yhao/baseline`
2. `CODI/` 不是独立仓库
3. `Coconut/` 是同仓库中的另一个子项目
4. `simcon` 就是 SIM-CoT 这条方法线
5. SIRCL 是可插拔统一稳定器，不只服务 CODI
6. `icot` 训练强依赖 cache，这是正式入口
7. `train_on_svamp_dataset/` 和 `train_on_multiarith_dataset/` 的单独训练结果属于正式实验
8. `results_useful/`、`plots/`、`final_use_model_codi_sim_sircl/` 是可信来源
9. 从 2026-03-25 起，新实验必须与历史结果隔离
10. `SemCoT/` 只作为外部参考，不应继续作为主运行依赖

---

## 11. 新接手者的最短上手路线

如果你是新接手的人，建议直接按这个顺序开始：

### 第一步：确认仓库边界

```bash
cd /data/yhao/baseline
git status
```

先记住：Git 根目录在这里，不在 `CODI/`。

### 第二步：阅读核心文档

按顺序看：

1. `CODI/PROJECT_GUIDE.md`
2. `CODI/README.md`
3. `CODI/REBUTTAL_WORKSPACE.md`

### 第三步：确认运行环境

```bash
cd /data/yhao/baseline/CODI
source config.env
```

看下面这些变量是否符合预期：

- `CODI_RUN_ROOT`
- `CODI_SAVE_DIR`
- `CODI_RESULT_DIR`
- `CODI_MULTIARITH_PATH`
- `CODI_SVAMP_PATH`
- `CODI_COIN_FLIP_PATH`

### 第四步：理解方法映射

先只记下面这张表：

| 名字 | 含义 |
|---|---|
| `codi` | CODI backbone |
| `codi_sircl` | CODI + SIRCL |
| `simcon` | SIM-CoT backbone |
| `simcon_sircl` | SIM-CoT + SIRCL |
| `coconut` | 在 `Coconut/` 子目录 |

### 第五步：确定实验脚本入口

按任务类型去看：

- 主线与消融：`CODI/scripts/`
- SVAMP：`CODI/train_on_svamp_dataset/`
- MultiArith：`CODI/train_on_multiarith_dataset/`
- CommonSense：`CODI/train_on_commen_dataset/`
- Coin Flip：`CODI/flip/`

### 第六步：开始新实验前先确认输出目录

必须确认它写向：

```bash
/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325
```

而不是旧的 `CODI/outputs` / `CODI/results`。

---

## 12. 文档之间的分工

为了避免以后重复维护，建议按下面理解文档角色：

### `README.md`
面向一般使用者的项目说明。

### `PROJECT_GUIDE.md`
面向接手者、合作者、revision 阶段维护者的总指南。

### `REBUTTAL_WORKSPACE.md`
只记录 revision / rebuttal 阶段的输出隔离和 Git 规则。

如果三者发生冲突：

1. 先以代码和脚本为准
2. 再以 `PROJECT_GUIDE.md` 的项目口径为准
3. 最后再参考 README 中的概述性内容

---

## 13. 最后一条原则

这个项目里最容易让人晕的，不是模型公式，而是：

- 多个子目录共存
- 历史结果很多
- 不同方法线的名字在代码中比较工程化
- Git 根目录不在 `CODI/`

所以以后接手时，请始终优先确认三件事：

1. 我现在站在哪个 Git 根目录里
2. 我现在跑的是哪条方法线
3. 我现在的输出会写到哪里

只要这三件事清楚，后面的代码和实验基本都能稳稳接住。
