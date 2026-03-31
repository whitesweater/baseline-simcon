# Baseline 新人接手与迁移指南

这份文档给第一次接手 `baseline/` 仓库的人使用。

目标不是复制一份新的项目百科，而是把仓库里已经存在的文档、脚本和当前工作约定串起来，让新人在尽量少依赖口头说明的情况下，做到三件事：

1. 理解这个仓库当前到底在做什么
2. 知道哪些文档和脚本才是 source of truth
3. 能把当前最小可运行的 CODI 主线迁移到另一台机器并验证成功

如果你只记住一句话，请记住：

> 这个仓库的 Git 根目录是 `baseline/`；当前最重要的工作主线在 `CODI/`；通用最小迁移入口是 `scripts/migrate_baseline_minimal.sh`；HPC2 长期协作迁移入口是 `scripts/migrate_baseline_hpc2_longterm.sh`；当前最重要的训练入口在 `CODI/train_on_gsm8k_dataset/`。

---

## 1. 先读什么，不要重复造轮子

这份文档本身不是新的 source of truth。它的作用是帮助你高效使用仓库里已经存在的文档。

### 1.1 文档导航

| 你现在的问题 | 优先看哪里 | 为什么 |
| --- | --- | --- |
| 这个仓库整体是什么、边界在哪、哪些结果可信 | `PROJECT_GUIDE.md` | 这是全仓库级总指南 |
| CODI 子项目具体怎么理解 | `CODI/README.md` | 这是 CODI 主线说明 |
| CODI 现在的测试方式和结果结构 | `CODI/TESTING_GUIDE.md` | 这是测试与结果组织入口 |
| 当前本地数据集到底从哪里来 | `CODI/local_datasets/README.md` | 这里明确说明了 `local_datasets/` 与 `SemCoT/` 的关系 |
| 当前迁移怎么做 | 本文档 + `scripts/migrate_baseline_hpc2_longterm.sh` / `scripts/migrate_baseline_minimal.sh` | HPC2 长期协作优先前者；通用最小迁移可用后者 |
| 迁移后怎么验证 | `scripts/verify_baseline_minimal.sh` | 这是标准验收入口 |
| 当前 rebuttal 多 backbone 训练怎么启动 | `CODI/train_on_gsm8k_dataset/` | 这是当前主线脚本目录 |

### 1.2 建议阅读顺序

如果你是完全第一次接手，建议按下面顺序：

1. `README.md`
2. `PROJECT_GUIDE.md`
3. 本文档 `NEWCOMER_HANDOVER.md`
4. `CODI/README.md`
5. `CODI/TESTING_GUIDE.md`
6. `CODI/train_on_gsm8k_dataset/prepare_assets.sh`
7. `CODI/train_on_gsm8k_dataset/train_llama1b.sh`

这个顺序的目的很简单：

- 先理解仓库边界
- 再理解当前活跃主线
- 最后再进入具体训练脚本

---

## 2. 先建立正确的全局脑图

### 2.1 Git 根目录在哪里

真正的 Git 仓库根目录是：

```bash
/data/yhao/baseline
```

因此：

- `CODI/` 不是独立仓库
- `Coconut/` 也不是独立仓库
- `git status`、`git add`、`git commit` 都应在 `baseline/` 这个边界下理解

### 2.2 当前最重要的目录

#### `CODI/`

这是当前最重要的研究与实验目录。

当前最活跃的主线是：

```bash
CODI/train_on_gsm8k_dataset/
```

也就是这轮 rebuttal / revision 的多 backbone、新 benchmark、新 stage 输出隔离，都主要围绕这个目录展开。

#### `Coconut/`

这是仓库里的另一条 backbone 代码线。

重要事实有两个：

- 从项目理解上，它仍然属于活跃范围，不是完全废弃的历史目录
- 通用最小迁移脚本默认不同步 `Coconut/`，但当前 HPC2 长期协作迁移会同步 `Coconut/` 的代码与小数据，并排除大 checkpoint、wandb 和日志

也就是说：

> 你要先区分“项目整体工作范围”“通用最小迁移保证范围”和“HPC2 长期协作同步范围”。

项目整体仍然会参考 Coconut；通用最小迁移只保证 CODI 主线；HPC2 长期协作则额外带上 Coconut 代码与小数据。

#### `CODI_rebuttal_runs/`

这是从 2026-03-25 开始的新实验输出根目录。

当前默认 rebuttal 根目录是：

```bash
/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325
```

当前 multi-backbone GSM8K stage 的子目录是：

```bash
/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1
```

这个目录非常重要，因为它承担了：

- stage 专属模型目录
- stage 专属 cache
- stage 专属 manifests
- stage 专属 outputs / results / logs

#### `CODI/local_datasets/`

这是当前 CODI 运行时真正使用的本地数据入口。

当前保留并实际使用的本地数据包括：

- `coin_flip`
- `multiarith`
- `svamp`

不要把 `CODI/SemCoT/` 当作当前运行时前提依赖。`SemCoT/` 现在更像外部参考仓库的拷贝，`local_datasets/` 才是当前实际入口。

#### `.github/skills/` 与 `imported_skills/`

这些目录不是核心科研代码。

它们主要是：

- HPC / VPN / 远程访问的操作性技能说明
- 迁移与访问流程的辅助脚本

什么时候需要看它们：

- 你在做 HPC 访问、VPN、远程容器接入

什么时候不需要看它们：

- 你只是要理解 CODI 主训练逻辑
- 你只是要按标准 SOP 迁移仓库

---

## 3. 当前方法映射与项目口径

这部分请和 `PROJECT_GUIDE.md` 一起看。

### 3.1 代码名和论文名的映射

| 代码名 | 论文/方法理解 |
| --- | --- |
| `codi` | CODI backbone |
| `codi_sircl` | CODI + SIRCL |
| `simcon` | SIM-CoT backbone |
| `simcon_sircl` | SIM-CoT + SIRCL |
| `coconut` | Coconut backbone |

最容易混淆的一点：

> 代码里的 `simcon`，就是论文里的 SIM-CoT 主线。

### 3.2 SIRCL 是什么

当前工程口径里，SIRCL 是一个可插拔稳定器插件，不只属于某一个 backbone。

它可以加在：

- CODI
- SIM-CoT
- Coconut

因此理解实验时，不要把它看成“某个模型自己的特殊分支”，而应理解为统一插件。

### 3.3 当前最重要的实验入口

如果你现在要继续当前主线实验，优先使用：

- `CODI/train_on_gsm8k_dataset/prepare_assets.sh`
- `CODI/train_on_gsm8k_dataset/train_llama1b.sh`
- `CODI/train_on_gsm8k_dataset/train_llama3b.sh`
- `CODI/train_on_gsm8k_dataset/train_llama8b.sh`
- `CODI/train_on_gsm8k_dataset/train_qwen3.sh`
- `CODI/train_on_gsm8k_dataset/train_qwen3_codi.sh`
- `CODI/train_on_gsm8k_dataset/eval_llama1b_math500_aime.sh`

主线的四个 `train_*.sh` 当前默认每次只训练一条 SIM-CoT 方法线：

- 默认：`simcon`
- 如果显式传 `--sircl` 或 `--variant simcon_sircl`：`simcon_sircl`

额外的 `train_qwen3_codi.sh` 提供可选的 `Qwen3-4B` CODI 入口：

- 默认：`codi`
- 如果显式传 `--sircl` 或 `--variant codi_sircl`：`codi_sircl`

---

## 4. 当前运行布局，你需要记住哪些路径

这部分以 `CODI/config.env` 和 `CODI/train_on_gsm8k_dataset/env.sh` 为准。

### 4.1 当前关键环境变量

当前配置里最关键的路径是：

```bash
CODI_RUN_ROOT=/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325
CODI_SAVE_DIR=${CODI_RUN_ROOT}/outputs
CODI_RESULT_DIR=${CODI_RUN_ROOT}/results
CODI_VENV_PATH=/data/yhao/baseline/.venv/bin/activate
```

这意味着：

- 当前 `.venv` 默认就在仓库根目录下
- 当前所有新实验产物默认应写入 `CODI_rebuttal_runs/rebuttal_20260325`
- 不应再把新的运行产物默认写回历史 `CODI/outputs` 或 `CODI/results`

### 4.2 当前 stage 目录结构

`CODI/train_on_gsm8k_dataset/env.sh` 会进一步把当前 active stage 组织成：

```bash
${CODI_RUN_ROOT}/multimodel_gsm8k_math500_aime_v1/
```

其下再分出：

- `models/`
- `outputs/`
- `results/`
- `logs/`
- `manifests/`
- `cache/`
- `hf_home/`
- `hf_datasets_cache/`
- `modelscope_cache/`

这是当前主线非常关键的工程约定：

> 新 stage 的模型、cache、结果都和历史目录隔离，尽量不要混回旧目录树。

### 4.3 当前 4 个基础模型

当前主线围绕 4 个基础模型：

- `Llama-3.2-1B-Instruct`
- `Llama-3.2-3B-Instruct`
- `Meta-Llama-3.1-8B-Instruct`
- `Qwen3-4B`

源机器上，它们可能来自外部绝对路径或软链。

迁移时不应简单照搬软链。

当前通用最小迁移脚本会把模型真实目录同步到新机器 stage 目录；
当前 HPC2 长期协作迁移则改为让 HPC2 自己把模型下载到共享缓存目录，并在 stage 目录下建立软链。

这也是为什么当前不同迁移脚本会对“工作区同步”和“模型准备”采取不同策略。

---

## 5. 什么是当前“最小可运行迁移”

当前最小迁移目标不是“把 3.6T 全部搬走”，而是：

> 在不改当前仓库结构和大部分路径假设的前提下，把当前 CODI 主线迁到新机器，保证 `CODI/train_on_gsm8k_dataset/*` 可以继续工作。

### 5.1 迁移时保留什么

当前通用最小迁移会保留：

- 根目录源码
- `.git`
- 当前未提交改动
- `CODI/config.env`
- `CODI/local_datasets/`
- `CODI/train_on_gsm8k_dataset/`
- 当前 stage 所需 cache / manifests
- 当前 4 个基础模型

### 5.2 迁移时不保什么

当前通用最小迁移默认不保：

- `.venv`
- checkpoint
- wandb
- 历史 outputs / results
- `Coconut/`
- `baid/`

请特别记住：

> 这是“最小可运行 CODI 主线迁移”，不是“完整镜像整个 baseline 仓库的所有研究资产”。

如果你后面要继续 Coconut 线，必须额外同步 `Coconut/`。

### 5.3 HPC2 长期协作和通用最小迁移有什么不同

如果目标是：

```bash
/hpc2hdd/home/yhao481/jhupload/proj/baseline
```

并且你希望之后在本机和 HPC2 之间长期来回切换开发，那么当前推荐的不是纯 rsync，而是：

- Git bootstrap 仓库
- 再用 rsync 覆盖当前 dirty / untracked 代码状态
- 仅在 HPC2 副本里最小本地化 `CODI/config.env`
- 用 HPC2 的 `uv` 建项目内 `.venv`
- 让 HPC2 自己下载基础模型到共享缓存

这条路径的入口是：

```bash
scripts/migrate_baseline_hpc2_longterm.sh
```

### 5.4 为什么当前推荐保留 `/data/yhao/baseline`

当前很多路径约定依赖这个绝对路径，例如：

- `CODI_VENV_PATH`
- 若干脚本里的默认路径假设
- 当前环境变量体系

因此最省心的做法不是改代码，而是在新机器继续提供：

```bash
/data/yhao/baseline
```

通常做法是：

- 真实目录放在别处
- 再把 `/data/yhao/baseline` 软链过去

---

## 6. 新人真正要执行的迁移 SOP

这一节是标准操作流程。

如果你的目标机是 HPC2，并且要按当前长期协作方案落地，请优先看 6.3A。
如果你的目标是普通 Linux 机器上的通用最小迁移，再看 6.3B。

### 6.1 迁移前准备

你需要一台满足下面条件的机器来执行迁移：

- 能访问源机器上的 `/data/yhao/baseline`
- 能通过 `ssh` 访问目标机器
- 有 `rsync`
- 有 `python3.11` 或你知道目标机上可用的 Python 版本

重要提醒：

> 迁移脚本本身不负责帮你建立 VPN。

如果目标机器是 HPC，而当前机器并不能连上 HPC，请换到一台已经能正常 SSH 到 HPC 的机器上执行迁移脚本。迁移脚本只假设网络已经通，不负责打通网络。

### 6.2 目标机先准备真实目录和兼容软链

推荐在目标机先准备：

```bash
ssh user@new-host "mkdir -p /your/real/path/baseline && mkdir -p /data/yhao && ln -sfn /your/real/path/baseline /data/yhao/baseline"
```

### 6.3A HPC2 长期协作迁移入口

从本地工作站执行：

```bash
cd /data/yhao/baseline

bash scripts/migrate_baseline_hpc2_longterm.sh
```

这条脚本固定使用：

- 远端入口：`hpc2-vpn`
- 目标目录：`/hpc2hdd/home/yhao481/jhupload/proj/baseline`
- 缓存目录：`/hpc2hdd/home/yhao481/jhupload/cache`

它会自动完成：

1. 通过远端 VPN 跳板验证 HPC2 接入
2. 在 HPC2 固定目录做 Git bootstrap
3. 用 rsync 覆盖当前 dirty / untracked 工作区
4. 在 HPC2 副本中本地化 `CODI/config.env`
5. 用 `uv` 建 Python 3.11 项目内 `.venv`
6. 在后台启动其余大模型下载
7. 前台完成 `llama1b + datasets` 验证和 `train.py --help` smoke check

### 6.3B 直接使用通用最小迁移脚本

从源机器执行：

```bash
cd /data/yhao/baseline

bash scripts/migrate_baseline_minimal.sh \
  --dst-host user@new-host \
  --dst-real /your/real/path/baseline \
  --dst-link /data/yhao/baseline \
  --python-bin python3.11
```

如果目标是 HPC，并且你已经有现成 SSH config，可写成：

```bash
cd /data/yhao/baseline

bash scripts/migrate_baseline_minimal.sh \
  --dst-host hpc2 \
  --dst-real /hpc2hdd/home/yhao481/jhupload/proj/baseline \
  --dst-link /data/yhao/baseline \
  --python-bin python3.11 \
  --ssh-config /root/.codex/skills/hpc-login-ssh/references/ssh_config_hpc
```

### 6.4 这个通用最小迁移脚本实际做了什么

`scripts/migrate_baseline_minimal.sh` 会自动：

1. 准备目标目录与兼容软链
2. 用 `rsync` 同步工作区
3. 排除 `.venv`、历史大结果目录和当前 stage 的 `models/`
4. 单独同步 4 个基础模型真实目录
5. 在目标机重建 `.venv`
6. 调用 `scripts/verify_baseline_minimal.sh` 做验收

### 6.5 常用参数

#### 只预演，不真正传输

```bash
bash scripts/migrate_baseline_minimal.sh \
  --dst-host user@new-host \
  --dst-real /your/real/path/baseline \
  --dry-run
```

#### 先只同步，不在目标机建环境

```bash
bash scripts/migrate_baseline_minimal.sh \
  --dst-host user@new-host \
  --dst-real /your/real/path/baseline \
  --no-bootstrap-venv \
  --no-verify
```

#### 只同步工作区

```bash
bash scripts/migrate_baseline_minimal.sh \
  --dst-host user@new-host \
  --dst-real /your/real/path/baseline \
  --workspace-only
```

#### 只同步模型

```bash
bash scripts/migrate_baseline_minimal.sh \
  --dst-host user@new-host \
  --dst-real /your/real/path/baseline \
  --models-only
```

### 6.6 如果你需要完整仓库而不是通用最小迁移

当前通用最小迁移脚本默认不传：

- `Coconut/`
- `baid/`
- 大量历史 outputs / results

如果你的目标是“完整接手整个 baseline 的所有历史实验资产”，你不能只依赖当前最小脚本。你需要额外设计分层同步策略。

如果你的目标只是：

> 把当前 CODI 主线继续跑起来

那当前脚本就是推荐入口。

如果你的目标是 HPC2 长期协作，请优先改用 `scripts/migrate_baseline_hpc2_longterm.sh`。

---

## 7. 迁移后如何验收

标准验收脚本是：

```bash
scripts/verify_baseline_minimal.sh
```

### 7.1 你应该期待它验证什么

它会检查：

- `/data/yhao/baseline` 是否正确指向目标真实目录
- `CODI/config.env` 是否存在
- `CODI/local_datasets/multiarith/train_42.json` 是否存在
- 当前 stage 所需 `icot` cache 是否存在
- stage 模型目录里的 `config.json` 是否存在
- `.venv` 是否可用
- `torch`、`transformers`、`datasets`、`modelscope` 是否能导入
- `prepare_assets.sh --models llama1b --force-datasets` 是否能走通
- `python train.py --help` 是否能通过 smoke check

### 7.2 手动复跑验收

如果你想单独在目标机上再验一次：

```bash
ssh user@new-host "
  cd /data/yhao/baseline &&
  bash scripts/verify_baseline_minimal.sh \
    --repo-root /data/yhao/baseline \
    --expected-real /your/real/path/baseline \
    --python-bin python3.11 \
    --bootstrap-venv
"
```

---

## 8. 迁移完成后，第一天应该怎么进入项目

### 8.1 最短上手路径

```bash
cd /data/yhao/baseline
git status

cd CODI
source config.env
source train_on_gsm8k_dataset/env.sh
source ../.venv/bin/activate
```

### 8.2 然后先做一个轻量准备动作

```bash
cd /data/yhao/baseline
source .venv/bin/activate
bash CODI/train_on_gsm8k_dataset/prepare_assets.sh --models llama1b --force-datasets
```

这一步会：

- 检查 stage 目录是否齐
- 检查 `icot` cache 是否齐
- 检查模型目录是否齐
- 必要时补 dataset warm-up

### 8.3 当前最常见训练入口

```bash
bash CODI/train_on_gsm8k_dataset/train_llama1b.sh
bash CODI/train_on_gsm8k_dataset/train_llama3b.sh
bash CODI/train_on_gsm8k_dataset/train_llama8b.sh
bash CODI/train_on_gsm8k_dataset/train_qwen3.sh
bash CODI/train_on_gsm8k_dataset/train_qwen3_codi.sh
```

默认训练方法线是：

```bash
simcon
```

如果要启用 SIRCL：

```bash
bash CODI/train_on_gsm8k_dataset/train_llama1b.sh --sircl
```

### 8.4 当前主线的资源准备逻辑

`prepare_assets.sh` 的资源优先级是：

1. 先复用本机已有模型路径
2. 再尝试 ModelScope
3. 再尝试 HF Mirror
4. 最后才是 Hugging Face 原站加代理

这意味着：

> 只要迁移时已经把 stage 模型目录同步好，目标机通常不应该再重新下载大模型。

---

## 9. 哪些东西可信，哪些东西要谨慎

这一部分请和 `PROJECT_GUIDE.md` 一起看。

### 9.1 当前优先相信什么

如果不同文档、脚本、历史结果互相冲突，优先级按下面顺序理解：

1. 论文最终口径与最终表格
2. 与论文一致的可信历史产物
3. 仓库根目录 `PROJECT_GUIDE.md`
4. 当前主线脚本与当前 stage 约定
5. 零散旧结果、未核对目录、调试产物

### 9.2 当前应优先参考的历史目录

如果你在核对历史结论，优先参考：

- `CODI/results_useful/`
- `CODI/plots/`
- `CODI/final_use_model_codi_sim_sircl/`

### 9.3 当前应谨慎处理的目录

需要谨慎对待：

- `CODI/results/`
- 历史 `outputs/`
- 零散日志与调试产物

原则不是“它们都错”，而是：

> 如果一个旧目录的内容和当前论文口径或可信产物冲突，先把它视为中间产物，而不是自动视为真相。

---

## 10. 当前接手人最容易踩的坑

### 10.1 把 `CODI/` 当成 Git 根目录

这是最常见错误之一。

请始终记住：

```bash
/data/yhao/baseline
```

才是 Git 根目录。

### 10.2 误以为新实验还应该写回旧的 `CODI/outputs` 和 `CODI/results`

不是。

当前新实验默认写入：

```bash
/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325
```

### 10.3 误以为 `SemCoT/` 是当前运行时必须依赖

不是。

当前运行时优先依赖：

```bash
CODI/local_datasets/
```

### 10.4 误以为最小迁移等于完整迁移

不是。

当前最小迁移只保证：

- 当前 CODI 主线
- 当前 4 个基础模型
- 当前 stage cache / manifests / 本地数据

不保证完整历史资产、Coconut 线和所有旧实验结果一起到位。

### 10.5 误以为迁移脚本会自动帮你解决 VPN / HPC 连通性

不会。

迁移脚本只负责：

- 传文件
- 建软链
- 建环境
- 验证可运行性

它不负责建立网络连通性。

---

## 11. 一张给新人的第一周清单

如果你是新接手者，建议在第一周完成下面这些动作。

### Day 1

1. 阅读 `README.md`
2. 阅读 `PROJECT_GUIDE.md`
3. 阅读本文档
4. 确认 Git 根目录和当前 active stage 路径

### Day 2

1. 阅读 `CODI/README.md`
2. 阅读 `CODI/TESTING_GUIDE.md`
3. 阅读 `CODI/train_on_gsm8k_dataset/prepare_assets.sh`
4. 阅读 `CODI/train_on_gsm8k_dataset/train_llama1b.sh`

### Day 3

1. 在目标机完成最小迁移
2. 运行 `scripts/verify_baseline_minimal.sh`
3. 运行一次 `prepare_assets.sh --models llama1b --force-datasets`

### Day 4-5

1. 读 `CODI/train.py`
2. 读 `CODI/src/model.py`
3. 理清 `simcon` / `simcon_sircl` 对应关系
4. 理清 stage 输出目录和 checkpoint 扫描逻辑

### Day 6-7

1. 试跑一个最小实验或 smoke check
2. 读一遍 `PROJECT_GUIDE.md` 中的“可信来源”和“rebuttal 规则”
3. 确认自己能独立回答下面 5 个问题

你应该能回答：

1. 为什么 Git 根目录是 `baseline/` 而不是 `CODI/`
2. 为什么当前新实验默认写到 `CODI_rebuttal_runs/`
3. 为什么通用最小迁移不默认带 `Coconut/`
4. 为什么 HPC2 长期协作改成 “Git 管代码，HPC2 自己下载模型”
5. 为什么当前 `simcon` 对应的是论文里的 SIM-CoT

---

## 12. 最后一句交接建议

如果你后面只做一件事来避免接手混乱，请做这件事：

> 每次开始工作前，先在仓库根目录 `git status`，然后回到 `PROJECT_GUIDE.md` 和当前 active script 检查自己是不是还站在当前主线上。

这比记住很多零散历史背景更重要。
