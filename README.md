# Baseline Workspace

这个仓库当前作为项目维护与 rebuttal / revision 阶段的工作根目录使用。

它不是一份对外展示型 README，而是给当前维护者和后续接手者使用的内部入口页。

## 先看哪里

第一次接手时，建议按这个顺序读：

1. [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)
2. [`CODI/README.md`](CODI/README.md)
3. [`CODI/TESTING_GUIDE.md`](CODI/TESTING_GUIDE.md)

如果要进入具体代码，再看：

- `CODI/train.py`
- `CODI/src/model.py`
- `Coconut/run.py`
- `CODI/train_on_gsm8k_dataset/`

## 仓库结构

当前最重要的目录有：

- `CODI/`
  - 当前主要的训练、评测、分析与历史结果目录
  - `codi / codi_sircl / simcon / simcon_sircl` 的主代码线在这里
- `Coconut/`
  - 同仓库中的另一条 backbone 代码线
  - 当前 rebuttal 视角下属于活跃范围，不是纯参考目录
- `CODI_rebuttal_runs/`
  - 2026-03-25 起的新实验输出根目录
  - 用来隔离 rebuttal 阶段的新 checkpoints、logs 和 results

## 当前默认工作原则

- Git 仓库根目录是 `baseline/`，不是 `CODI/`
- 新实验默认写到 `CODI_rebuttal_runs/rebuttal_20260325`
- 冲突时优先以论文最终口径和可信历史产物为准
- 当前默认工作重心是 cross-backbone 的 rebuttal 新实验
- 当前多 backbone 新训练与 `math500 / aime` 扩展评测，优先走 `CODI/train_on_gsm8k_dataset/`

推荐环境变量：

```bash
CODI_RUN_ROOT=/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325
CODI_SAVE_DIR=${CODI_RUN_ROOT}/outputs
CODI_RESULT_DIR=${CODI_RUN_ROOT}/results
```

## 文档导航

- [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)
  - 仓库级总指南
  - 说明目录边界、方法映射、可信结果、实验口径、Git 规则和 rebuttal 默认策略
- [`CODI/README.md`](CODI/README.md)
  - CODI 子项目说明
  - 适合在明确要进入 CODI 训练与评测脚本后继续阅读
- `CODI/train_on_gsm8k_dataset/`
  - 当前 cross-backbone rebuttal 实验的专用脚本目录
  - 包含 `prepare_assets.sh`、4 个直接训练入口和 `llama1b` 的扩展评测入口
- [`CODI/PROJECT_GUIDE.md`](CODI/PROJECT_GUIDE.md)
  - 兼容旧入口的导读页
- [`CODI/REBUTTAL_WORKSPACE.md`](CODI/REBUTTAL_WORKSPACE.md)
  - 兼容旧入口的导读页

## 快速上手

```bash
cd /data/yhao/baseline
git status

cd /data/yhao/baseline/CODI
source config.env
```

确认下面这些变量已经指向新的 rebuttal 输出目录：

- `CODI_RUN_ROOT`
- `CODI_SAVE_DIR`
- `CODI_RESULT_DIR`

如果你现在要把任务分发到多台 `4 x H800 80GB` 机器，优先直接执行：

```bash
bash CODI/train_on_gsm8k_dataset/train_llama1b.sh
```

其余 `llama3b`、`llama8b`、`qwen3` 和 `llama1b` 额外评测入口，也都统一放在 `CODI/train_on_gsm8k_dataset/`。

如果后续 reviewer / rebuttal 任务清单有单独文件，它也应与本仓库文档一起作为正式输入。
