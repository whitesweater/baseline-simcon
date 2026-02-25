# Scripts 目录说明

本目录包含 CODI 项目的训练、测试和工具脚本。

## 脚本分类概览

| 类别 | 数量 | 说明 |
|------|------|------|
| ✅ 本地适配（可用） | 29 | 已适配 `config.env`，可直接使用 |
| ❌ 原始未适配（无法直接使用） | 13 | Clone 时带来的原始脚本，路径指向其他系统 |
| 🔧 Python 工具 | 4 | 绘图、数据展示等辅助工具 |

---

## ✅ 本地适配脚本（可直接使用）

这些脚本已通过 `config.env` 进行环境适配。

### 训练脚本 (Training)

| 脚本名 | 功能 | 关键参数 |
|--------|------|----------|
| `codi_bash.sh` | **CODI 基线训练** | 4 GPU，10 epochs，无 trajectory loss |
| `codi_sircl_005factor.sh` | CODI + SIRCL (factor=0.05) | 含 trajectory consistency |
| `euclidean_001factor.sh` | Euclidean 空间 (factor=0.01) | 消融实验 |
| `euclidean_005factor.sh` | Euclidean 空间 (factor=0.05) | 消融实验 |
| `euclidean_0075factor.sh` | Euclidean 空间 (factor=0.075) | 消融实验 |
| `euclidean_015factor.sh` | Euclidean 空间 (factor=0.15) | 消融实验 |
| `euclidean_4.sh` | Euclidean + 4 latent tokens | latent 数量实验 |
| `euclidean_16.sh` | Euclidean + 16 latent tokens | latent 数量实验 |
| `euclidean_32.sh` | Euclidean + 32 latent tokens | latent 数量实验 |
| `euclidean_bigthrehold.sh` | Euclidean + 大阈值 | 阈值消融 |
| `euclidean_smallthrehold.sh` | Euclidean + 小阈值 | 阈值消融 |
| `hyperbolic.sh` | **Hyperbolic 空间训练** | Poincaré 距离 |
| `geodesic.sh` | Geodesic Deviation Loss | 测地线偏差 |
| `acceleration.sh` | Acceleration Loss | 二阶平滑 |
| `least_action.sh` | Least Action Loss | 路径能量 |
| `train_llama3b_gsm8k-aug-decoder-2.sh` | LLaMA 3B 训练 | 大模型 |
| `train_llama8b_gsm8k-aug-decoder-2.sh` | LLaMA 8B 训练 | 大模型 |
| `train_and_test_both.sh` | MultiArith + SVAMP 并行训练 | 多数据集 |
| `run_commen_simcon_sircl_ablation.sh` | SimCon SIRCL 消融实验 | 消融 |

### 测试脚本 (Testing)

| 脚本名 | 功能 | 说明 |
|--------|------|------|
| `test_llama1b.sh` | LLaMA 1B 单模型测试 | 基础测试 |
| `test_llama1b-hyperbolic-radius.sh` | Hyperbolic 半径统计 | 调试用 |
| `testcopy.sh` | 通用测试模板 | 支持多数据集 |
| `batch_test.sh` | **批量测试** | 多模型、多次运行 |
| `batch_test_multi.sh` | **多模型多数据集批量测试** | 推荐使用 |
| `run_all_models.sh` | 遍历所有模型测试 | 自动化 |
| `test_latent_sweep.sh` | Latent 数量遍历测试 (1-18) | 主脚本 |
| `test_latent_sweep_group1.sh` | Group 1: codi, codi_sircl | 并行 |
| `test_latent_sweep_group2.sh` | Group 2: simcon, sircl | 并行 |

### 工具脚本 (Tools)

| 脚本名 | 功能 |
|--------|------|
| `probe_latent_token.sh` | Latent token 探测分析 |

---

## ❌ 原始脚本（未适配，无法直接使用）

这些脚本是从原始仓库 clone 来的，包含硬编码的外部路径，**在当前环境无法运行**。

| 脚本名 | 原始路径 | 说明 |
|--------|----------|------|
| `train_gpt2_commonsense.sh` | `/ephemeral/gpt2_commonsense` | GPT-2 Commonsense 训练 |
| `train_gpt2_gsm8k-aug.sh` | `/scratch/prj/.../gpt_codi_lora` | GPT-2 GSM8K 训练 |
| `train_gpt2_gsm8k-aug-nl.sh` | `/scratch/prj/.../gpt_codi_lora_nl` | GPT-2 无 LoRA 训练 |
| `train_gpt2_gsm8k-aug-decoder-2.sh` | `/mnt/shared-storage-user/...` | GPT-2 Decoder 训练 |
| `train_llama1b_gsm8k-aug.sh` | `/ephemeral/codi_llama1b_full` | LLaMA 1B 基础训练 |
| `train_llama1b_gsm8k-aug-nl.sh` | `~/codi_ckpt/codi_nl_llama` | LLaMA 1B 无 LoRA |
| `train_llama1b_gsm8k-aug-decoder.sh` | `/data/user/yhao481/proj/...` | LLaMA 1B Decoder (旧路径) |
| `train_llama_commonsense.sh` | `~/codi_ckpt/llama_commonsense` | LLaMA Commonsense |
| `test_gpt2.sh` | 无 SAVE_DIR 定义 | GPT-2 测试 (不完整) |
| `test_gpt2_decoder-copy.sh` | `/mnt/shared-storage-user/...` | GPT-2 Decoder 测试 |
| `test_llama3b-copy.sh` | `/mnt/shared-storage-user/...` | LLaMA 3B 测试 |
| `test_llama8b-copy.sh` | `/mnt/shared-storage-user/...` | LLaMA 8B 测试 |
| `test_latent_sweep_all.sh` | 无 config.env | 并发测试入口 (不完整) |

---

## 🔧 Python 工具脚本

| 脚本名 | 功能 | 依赖 |
|--------|------|------|
| `plot_colors.py` | 统一颜色配置 | matplotlib |
| `plot_gsm8k_comparison.py` | GSM8K 模型对比柱状图 | matplotlib, numpy |
| `plot_individual_figures.py` | 可视化图片生成 | sklearn, matplotlib |
| `show_dataset_samples.py` | 数据集样例展示 | 无 |

---

## 快速使用指南

### 1. 训练 CODI 基线模型
```bash
cd /data/yhao/baseline/CODI
./scripts/codi_bash.sh
```

### 2. 训练带 SIRCL 的模型
```bash
./scripts/codi_sircl_005factor.sh
```

### 3. 批量测试多个模型
```bash
# 测试所有模型，每个数据集运行 1 次
./scripts/batch_test_multi.sh

# 只测试 gsm8k 数据集
./scripts/batch_test_multi.sh -d gsm8k

# 预览命令（不实际运行）
./scripts/batch_test_multi.sh --dry-run
```

### 4. Latent 数量遍历实验
```bash
# 测试 latent tokens 从 1 到 18
./scripts/test_latent_sweep.sh
```

### 5. 查看数据集样例
```bash
python scripts/show_dataset_samples.py --datasets "gsm8k svamp"
```

---

## 脚本依赖

所有本地适配脚本依赖 `config.env` 文件：
```bash
# config.env 主要变量
CODI_SAVE_DIR        # 模型保存目录
CODI_RESULT_DIR      # 测试结果目录
CODI_LLAMA1B_PATH    # LLaMA 1B 模型路径
CODI_LLAMA3B_PATH    # LLaMA 3B 模型路径
CODI_LLAMA8B_PATH    # LLaMA 8B 模型路径
```

---

## 实验模型对应关系

| 模型类型 | 训练脚本 | 输出目录 |
|----------|----------|----------|
| CODI 基线 | `codi_bash.sh` | `codi-base/` |
| CODI + SIRCL | `codi_sircl_005factor.sh` | `codi-euclidean/` |
| SimCon | 参见 `train_on_*` 目录 | `simcon/` |
| SimCon + SIRCL | `run_commen_simcon_sircl_ablation.sh` | `simcon_sircl/` |
| Euclidean | `euclidean_*.sh` | `decoder-trajectory-euclidean-*/` |
| Hyperbolic | `hyperbolic.sh` | `gsm8k_llama1b_latent_decoder-trajectory-hyperbolic/` |



# CODI Scripts 目录说明

本目录包含训练、测试和可视化的各类脚本。以下按功能分类，并标注实际使用状态。

---

## 📊 使用状态说明

| 标记 | 含义 |
|------|------|
| ✅ **本地使用** | 已本地化配置（使用 `config.env`），可直接运行 |
| ⚠️ **部分本地化** | 混合使用本地配置和硬编码路径 |
| ❌ **原始仓库** | 保留原始仓库路径（`/mnt/`, `/scratch/`, `/ephemeral/`, `~/`），需修改后使用 |

---

## 1. 训练脚本 (Training)

### ✅ 本地使用 - Trajectory Consistency 实验

| 脚本 | 功能 | 关键参数 |
|------|------|----------|
| `codi_bash.sh` | CODI 基线训练（无 trajectory loss） | `num_latent=6`, 无 `use_trajectory_consistency` |
| `codi_sircl_005factor.sh` | CODI + SIRCL（factor=0.05） | `trajectory_loss_factor=0.05` |
| `euclidean_001factor.sh` | Euclidean trajectory，factor=0.01 | `trajectory_space_type=euclidean`, `factor=0.01` |
| `euclidean_005factor.sh` | Euclidean trajectory，factor=0.05 | `factor=0.05` |
| `euclidean_0075factor.sh` | Euclidean trajectory，factor=0.075 | `factor=0.075` |
| `euclidean_015factor.sh` | Euclidean trajectory，factor=0.15 | `factor=0.15` |
| `euclidean_bigthrehold.sh` | Euclidean，大阈值 | `radius_threshold=8.0` |
| `euclidean_smallthrehold.sh` | Euclidean，小阈值 | `radius_threshold=1.0` |
| `hyperbolic.sh` | Hyperbolic trajectory consistency | `trajectory_space_type=hyperbolic` |

### ✅ 本地使用 - Latent Token 数量实验

| 脚本 | 功能 | 关键参数 |
|------|------|----------|
| `euclidean_4.sh` | 4 个 latent tokens | `num_latent=3` (生成4个) |
| `euclidean_16.sh` | 16 个 latent tokens | `num_latent=15` |
| `euclidean_32.sh` | 32 个 latent tokens | `num_latent=31` |

### ✅ 本地使用 - 其他 Trajectory Loss 变体

| 脚本 | 功能 |
|------|------|
| `acceleration.sh` | 二阶加速度平滑约束 |
| `geodesic.sh` | 测地线偏离约束 |
| `least_action.sh` | 最小作用量约束 |

### ✅ 本地使用 - 多数据集/大模型训练

| 脚本 | 功能 |
|------|------|
| `train_and_test_both.sh` | 同时在 MultiArith 和 SVAMP 上训练 |
| `train_llama3b_gsm8k-aug-decoder-2.sh` | LLaMA 3B 训练 |
| `train_llama8b_gsm8k-aug-decoder-2.sh` | LLaMA 8B 训练 |

### ❌ 原始仓库 - 未本地化

| 脚本 | 问题 | 原始路径 |
|------|------|----------|
| `train_gpt2_gsm8k-aug.sh` | 使用外部集群路径 | `/scratch/prj/inf_multimodal_qa/...` |
| `train_gpt2_gsm8k-aug-nl.sh` | 使用外部集群路径 | `/scratch/prj/inf_multimodal_qa/...` |
| `train_gpt2_gsm8k-aug-decoder-2.sh` | 使用共享存储路径 | `/mnt/shared-storage-user/...` |
| `train_gpt2_commonsense.sh` | 使用临时目录 | `/ephemeral/gpt2_commonsense` |
| `train_llama1b_gsm8k-aug.sh` | 使用临时目录 | `/ephemeral/codi_llama1b_full` |
| `train_llama1b_gsm8k-aug-nl.sh` | 使用 home 目录 | `~/codi_ckpt/codi_nl_llama` |
| `train_llama1b_gsm8k-aug-decoder.sh` | 使用其他机器路径 | `/data/user/yhao481/...` |
| `train_llama_commonsense.sh` | 使用 home 目录 | `~/codi_ckpt/llama_commonsense` |

---

## 2. 测试脚本 (Testing)

### ✅ 本地使用 - 批量测试

| 脚本 | 功能 | 使用方式 |
|------|------|----------|
| `batch_test.sh` | 单模型多数据集批量测试 | `./batch_test.sh -m euclidean -d "gsm8k svamp"` |
| `batch_test_multi.sh` | 多模型多数据集批量测试 | `./batch_test_multi.sh -m "codi simcon" -d gsm8k` |
| `run_all_models.sh` | 遍历所有模型运行测试 | `./run_all_models.sh --parallel` |
| `run_commen_simcon_sircl_ablation.sh` | Commonsense QA 消融实验 | 专用于 coin_flip/commonsense 数据集 |

### ✅ 本地使用 - Latent Sweep 测试

| 脚本 | 功能 |
|------|------|
| `test_latent_sweep.sh` | 单 GPU 遍历不同 latent 迭代次数 |
| `test_latent_sweep_group1.sh` | 第1组模型（codi, codi_sircl） |
| `test_latent_sweep_group2.sh` | 第2组模型（simcon, sircl） |
| `test_latent_sweep_all.sh` | 并发运行两组测试 |

### ⚠️ 部分本地化

| 脚本 | 问题 |
|------|------|
| `testcopy.sh` | 基本本地化，有默认 ckpt 路径 fallback |
| `test_llama1b.sh` | 混合本地化，注释中有硬编码路径 |
| `test_llama1b-hyperbolic-radius.sh` | 使用本地配置 |
| `test_gpt2.sh` | 使用 `~/transfer/` 路径 |
| `probe_latent_token.sh` | 有硬编码的 model 和 ckpt 路径 |

### ❌ 原始仓库 - 未本地化

| 脚本 | 问题 |
|------|------|
| `test_gpt2_decoder-copy.sh` | 使用 `/mnt/shared-storage-user/` 路径 |
| `test_llama3b-copy.sh` | 使用 `/mnt/shared-storage-user/` 路径 |
| `test_llama8b-copy.sh` | 使用 `/mnt/shared-storage-user/` 路径 |

---

## 3. 可视化与分析脚本 (Python)

| 脚本 | 功能 | 用法 |
|------|------|------|
| `plot_colors.py` | 定义统一配色方案（53行） | 作为 module 导入 |
| `plot_gsm8k_comparison.py` | 绘制 GSM8K 模型对比图（99行） | `python scripts/plot_gsm8k_comparison.py` |
| `plot_individual_figures.py` | 批量生成独立图表（985行） | 完整的可视化 pipeline |
| `show_dataset_samples.py` | 展示数据集样例（255行） | `python scripts/show_dataset_samples.py --datasets "gsm8k svamp"` |

---

## 4. 快速参考

### 常用训练命令

```bash
# 基线 CODI
./scripts/codi_bash.sh

# CODI + Euclidean Trajectory (推荐配置)
./scripts/euclidean_005factor.sh

# CODI + Hyperbolic Trajectory
./scripts/hyperbolic.sh

# 更多 latent tokens
./scripts/euclidean_16.sh   # 16 tokens
./scripts/euclidean_32.sh   # 32 tokens
```

### 常用测试命令

```bash
# 批量测试所有模型
./scripts/batch_test_multi.sh

# 测试特定模型和数据集
./scripts/batch_test.sh -m euclidean -d "gsm8k svamp multi-arith"

# Latent sweep（不同迭代次数）
./scripts/test_latent_sweep_all.sh --start 1 --end 10
```

---

## 5. 配置依赖

所有 ✅ 本地使用的脚本都依赖 `config.env` 文件，需要设置以下变量：

```bash
# config.env 示例
CODI_LLAMA1B_PATH="/path/to/Llama-3.2-1B-Instruct"
CODI_SAVE_DIR="/data/yhao/baseline/CODI/outputs"
CODI_RESULT_DIR="/data/yhao/baseline/CODI/results"
```

---

## 6. 目录结构速览

```
scripts/
├── 训练脚本
│   ├── codi_bash.sh                    # ✅ CODI 基线
│   ├── codi_sircl_005factor.sh         # ✅ CODI + SIRCL
│   ├── euclidean_*.sh                  # ✅ Euclidean 变体 (7个)
│   ├── hyperbolic.sh                   # ✅ Hyperbolic
│   ├── acceleration.sh                 # ✅ 加速度约束
│   ├── geodesic.sh                     # ✅ 测地线约束
│   ├── least_action.sh                 # ✅ 最小作用量
│   ├── train_llama{3b,8b}_*.sh         # ✅ 大模型训练
│   └── train_{gpt2,llama1b}_*.sh       # ❌ 原始仓库
│
├── 测试脚本
│   ├── batch_test*.sh                  # ✅ 批量测试
│   ├── test_latent_sweep*.sh           # ✅ Latent sweep
│   ├── run_*.sh                        # ✅ 运行辅助
│   ├── testcopy.sh                     # ⚠️ 部分本地化
│   └── test_{gpt2,llama{3b,8b}}-copy.sh # ❌ 原始仓库
│
└── Python 脚本
    ├── plot_*.py                       # 可视化
    └── show_dataset_samples.py         # 数据集展示
```

---

## 7. 注意事项

1. **运行前检查**：确保 `config.env` 存在且配置正确
2. **GPU 配置**：训练脚本默认使用 4 GPU（`nproc_per_node=4`），按需修改
3. **端口冲突**：不同脚本使用不同的 `master_port`，并行运行时注意
4. **原始脚本**：标记为 ❌ 的脚本需要修改路径后才能在本地使用
