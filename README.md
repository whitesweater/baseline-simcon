# Baseline Workspace

SIM-CoT（Supervised Implicit Chain-of-Thought）与 SIRCL（trajectory stability plugin）的研究仓库。当前处于 **rebuttal / revision 阶段**，主要工作是 cross-backbone 实验（LLaMA-1B/3B/8B、Qwen3-4B）和扩展评测（GSM8K、Math500、AIME）。

---

## 1. 仓库结构

Git 根目录是 `baseline/`，不是 `CODI/`。

| 目录 | 角色 | 状态 |
|------|------|------|
| `CODI/` | 主研究代码（训练、评测、分析、历史结果） | 活跃 |
| `Coconut/` | 另一条 backbone（GPT-2/LLaMA via FSDP） | 活跃（rebuttal 范围） |
| `CODI_rebuttal_runs/` | 2026-03-25 起的新实验输出根目录 | 活跃 |
| `CODI/local_datasets/` | 运行时本地 JSON 数据（coin_flip, multiarith, svamp） | 当前入口 |
| `CODI/SemCoT/` | 外部参考仓库拷贝 | 只读参考，非运行时依赖 |
| `CODI/results_useful/` | 论文可信历史结果 | 只读参考 |
| `CODI/final_use_model_codi_sim_sircl/` | 论文最终 checkpoint | 只读参考 |
| `docs/` | HPC/VPN 访问等辅助文档 | 按需查阅 |
| `scripts/` | 迁移/部署工具 | 按需使用 |

---

## 2. 方法映射

| 代码名 | 论文名 | 开关组合 |
|--------|--------|----------|
| `codi` | CODI | `use_decoder=False`, `use_trajectory_consistency=False` |
| `codi_sircl` | CODI + SIRCL | `use_decoder=False`, `use_trajectory_consistency=True` |
| `simcon` | **SIM-CoT** | `use_decoder=True`, `use_trajectory_consistency=False` |
| `simcon_sircl` | **SIM-CoT + SIRCL** | `use_decoder=True`, `use_trajectory_consistency=True` |
| `coconut` | Coconut | `Coconut/` 目录中的 backbone |

> 代码里的 `simcon` 就是论文里的 SIM-CoT。不要把它当随意内部简称。

**SIRCL** 是可插拔的 trajectory consistency loss 稳定器，不绑定特定 backbone。可加在 CODI、SIM-CoT、Coconut 上。

---

## 3. 快速开始

### 3.1 环境配置

```bash
cd /data/yhao/baseline
source .venv/bin/activate
cd CODI && source config.env
```

`CODI/config.env` 是每台机器的本地配置（已 gitignore），关键变量：

| 变量 | 说明 |
|------|------|
| `CODI_RUN_ROOT` | 当前 rebuttal 实验根目录 |
| `CODI_SAVE_DIR` | checkpoint/日志输出 |
| `CODI_RESULT_DIR` | 评测结果输出 |
| `CODI_LLAMA1B_PATH` | LLaMA-1B 模型路径 |
| `CODI_LLAMA3B_PATH` | LLaMA-3B 模型路径 |
| `CODI_GSM8K_AUG_PATH` | GSM8k-Aug 数据集路径 |
| `CODI_CACHE_DIR` | 预处理数据缓存 |
| `HF_ENDPOINT` | HuggingFace 镜像（中国用 `https://hf-mirror.com`） |

当前默认 rebuttal 输出目录：

```bash
CODI_RUN_ROOT=/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325
CODI_SAVE_DIR=${CODI_RUN_ROOT}/outputs
CODI_RESULT_DIR=${CODI_RUN_ROOT}/results
```

### 3.2 训练

当前 rebuttal 主线入口在 `CODI/train_on_gsm8k_dataset/`：

```bash
# 准备模型和数据集
bash CODI/train_on_gsm8k_dataset/prepare_assets.sh --models llama1b --force-datasets

# SIM-CoT 训练（默认 simcon，加 --sircl 切换 simcon_sircl）
bash CODI/train_on_gsm8k_dataset/train_llama1b.sh
bash CODI/train_on_gsm8k_dataset/train_llama3b.sh
bash CODI/train_on_gsm8k_dataset/train_llama8b.sh
bash CODI/train_on_gsm8k_dataset/train_qwen3.sh

# Qwen3-4B CODI 入口（默认 codi，加 --sircl 切换 codi_sircl）
bash CODI/train_on_gsm8k_dataset/train_qwen3_codi.sh
```

关键训练参数：`--num_latent`（隐式 token 数）、`--use_decoder`（启用 SIM-CoT）、`--distill_loss_factor`（推荐 20）、`--explain_loss_factor`（推荐 1.0）。

Coconut 两阶段训练：

```bash
cd Coconut
torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_coconut.yaml       # Stage 1
torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_simcot.yaml        # Stage 2（需设 load_model_path）
```

### 3.3 评测

```bash
# 单数据集
python CODI/test.py --model_name <model> --ckpt_dir <path> --data_name gsm8k

# 多数据集（加载一次模型测所有数据集）
python CODI/test_multi_dataset.py ...

# 批量测试
bash CODI/scripts/batch_test_multi.sh

# 扩展评测（Math500, AIME）
bash CODI/train_on_gsm8k_dataset/eval_llama1b_math500_aime.sh
```

---

## 4. 工作规则

### 4.1 Rebuttal 输出隔离

新实验**不写**回 `CODI/outputs/` 或 `CODI/results/`，统一写入：

```
CODI_rebuttal_runs/rebuttal_20260325/
├── multimodel_gsm8k_math500_aime_v1/
│   ├── models/      # stage 专属模型
│   ├── outputs/     # checkpoint
│   ├── results/     # 评测结果
│   ├── logs/        # 训练日志
│   ├── cache/       # 预处理缓存
│   └── manifests/   # 资产记录
```

### 4.2 证据优先级

当文档、历史结果、脚本互相冲突时：

1. 论文最终正文和最终表格
2. `CODI/results_useful/`、`CODI/final_use_model_codi_sim_sircl/` 等可信产物
3. 本文档 `README.md`
4. 当前活跃脚本
5. 零散旧结果和调试产物

`CODI/results/` 混有调试试验和失败运行，不要默认视为可信。

### 4.3 Git 规则

- 在 `baseline/` 根目录切分支、提交
- 代码和必要文档可以提交
- checkpoint、logs、results、大型生成物不要提交
- `CODI/SemCoT/` 不作为主开发内容提交

### 4.4 `icot` 数据入口

`icot` 的权威入口是缓存（cache），不是从 Hugging Face 重新构建。`train.py` 在 `icot` 分支上依赖 cache 是正式用法。

---

## 5. 迁移

### 5.1 HPC2 长期协作迁移（推荐）

目标：`/hpc2hdd/home/yhao481/jhupload/proj/baseline`

```bash
cd /data/yhao/baseline
bash scripts/migrate_baseline_hpc2_longterm.sh
```

自动完成：Git bootstrap → rsync 覆盖 dirty/untracked → 本地化 config.env → uv 建 .venv → 后台下载模型。

### 5.2 通用最小迁移

只保证 CODI 主线可运行，不包含 Coconut 和历史结果。

```bash
bash scripts/migrate_baseline_minimal.sh \
  --dst-host user@new-host \
  --dst-real /your/real/path/baseline \
  --dst-link /data/yhao/baseline \
  --python-bin python3.11
```

支持 `--dry-run`（预演）、`--workspace-only`（只同步代码）、`--models-only`（只同步模型）。

### 5.3 验收

```bash
bash scripts/verify_baseline_minimal.sh --repo-root /data/yhao/baseline
```

检查项：目录软链、config.env、local_datasets、icot cache、stage 模型、.venv、torch/transformers 可导入、prepare_assets.sh 及 train.py --help smoke check。

### 5.4 迁移后第一步

```bash
cd /data/yhao/baseline && source .venv/bin/activate
bash CODI/train_on_gsm8k_dataset/prepare_assets.sh --models llama1b --force-datasets
```

---

## 6. 架构与关键文件

### 核心代码

| 文件 | 说明 |
|------|------|
| `CODI/src/model.py` | CODI 模型：CausalLM + LoRA + latent loop + 投影层 + 多 loss |
| `CODI/train.py` | CustomTrainer：多 loss 计算、dynamic scheduling via step_ratio |
| `CODI/test.py` / `test_multi_dataset.py` | 评测（latent iteration inference） |
| `CODI/src/tokenizer_utils.py` | `load_tokenizer_with_fallback()` 保证 special token 一致 |
| `Coconut/coconut.py` | Coconut 多 pass forward 逻辑 |
| `Coconut/run.py` | FSDP 分布式训练 |

### Loss 模块（`CODI/src/`）

| 模块 | 说明 |
|------|------|
| `trajectory_consistency.py` | Fréchet mean 约束（Euclidean/Hyperbolic） |
| `trajectory_acceleration.py` | 二阶平滑 |
| `trajectory_action.py` | 路径能量（最小作用量） |
| `trajectory_geodesic.py` | 测地线偏差 |
| `rank_diversity.py` | Rank collapse 防护 |

### Special Tokens

`pad_id`、`bot_id`（begin-of-thought）、`eot_id`（end-of-thought）。训练和测试时 token ID 必须一致，始终用 `load_tokenizer_with_fallback()`。

### LoRA 配置

默认：rank 128, alpha 16, dropout 0.05。目标模块按模型族不同：
- LLaMA/Qwen: q/k/v/o/gate/up/down_proj
- Phi: q/k/v_proj, dense, fc1, fc2
- GPT-2: c_attn, c_proj, c_fc

---

## 7. 子项目文档

| 文档 | 内容 |
|------|------|
| [`CODI/README.md`](CODI/README.md) | CODI 技术细节：架构图、参数表、目录结构、实验脚本 |
| [`CODI/TESTING_GUIDE.md`](CODI/TESTING_GUIDE.md) | 多模型多数据集测试协议、结果目录结构 |
| [`Coconut/TRAJECTORY_CONSISTENCY.md`](Coconut/TRAJECTORY_CONSISTENCY.md) | Coconut + trajectory consistency 实现指南 |
| [`CODI/local_datasets/README.md`](CODI/local_datasets/README.md) | 本地数据集版本管理说明 |
| [`docs/HPC_REMOTE_VPN_ACCESS.md`](docs/HPC_REMOTE_VPN_ACCESS.md) | HPC2/HPC4 远程 VPN 接入拓扑 |

---

## 8. 常见问题

| 问题 | 解决 |
|------|------|
| OOM | 减 `per_device_train_batch_size`、增 `gradient_accumulation_steps`、减 `max_token_num` |
| 评估准确率为 0 | 检查 `ckpt_dir` 路径、`inf_latent_iterations` 与 `num_latent` 匹配、`remove_eos` 一致 |
| Token 对齐 | train/test 的 special token ID 必须一致 |
| Loss 平衡 | `distill_loss_factor=20`、`explain_loss_factor=1.0` |
| Checkpoint 加载 | Coconut Stage 2 用 `load_model_path`；CODI 评测用 `--ckpt_dir` |

---

## 9. 环境要求

- Python ≥3.10, <3.13
- 使用 `uv` + `pyproject.toml` 管理依赖
- 虚拟环境在 `.venv/`
- 核心：torch 2.5.1, transformers 4.46.2, peft 0.13.0+, accelerate 1.7.0+
- 硬件默认：1 台机器 × 4 张 H800 80GB
- 基础模型：Llama-3.2-1B-Instruct、Llama-3.2-3B-Instruct、Meta-Llama-3.1-8B-Instruct、Qwen3-4B

### 网络/镜像（HPC 环境）

```bash
export http_proxy=http://127.0.0.1:3128
export https_proxy=http://127.0.0.1:3128
pip config set global.index-url http://harbor.internal.com:8081/repository/pypi-hkust/simple
pip config set install.trusted-host harbor.internal.com
```
