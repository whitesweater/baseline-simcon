# CODI: Continuous Thought Distillation

CODI（Continuous thought DIstillation）是一个通过 LoRA 适配训练因果语言模型（Causal LM）生成隐式思维链（Implicit Chain-of-Thought）潜在向量的研究项目。该框架支持教师蒸馏、轨迹一致性约束（Trajectory Consistency），以及可选的辅助解码器路径。

## 接手建议

如果你是第一次接手这个项目，或者正在做 rebuttal / revision，建议优先阅读：

- `../PROJECT_GUIDE.md`：仓库级总指南，包含 repo 边界、CODI/Coconut 关系、方法映射、可信结果、实验原则、Git 规则
- `PROJECT_GUIDE.md`：CODI 子项目导读页，用于把旧入口引回根级总指南
- `REBUTTAL_WORKSPACE.md`：CODI 侧 rebuttal 导读页，说明新的 workspace 规则已经合并到根级总指南
- `TESTING_GUIDE.md`：CODI 的专项测试说明

## 目录

- [项目架构](#项目架构)
- [环境安装](#环境安装)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
- [目录结构详解](#目录结构详解)
- [核心组件](#核心组件)
- [配置参数](#配置参数)
- [实验脚本](#实验脚本)
- [结果分析](#结果分析)

---

## 项目架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         CODI Framework                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   Teacher    │───▶│   Latent     │───▶│   Student        │   │
│  │   Model      │    │   Loop       │    │   Decoder        │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
│         │                   │                     │              │
│         ▼                   ▼                     ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   Reference  │    │  Trajectory  │    │   Answer CE      │   │
│  │   CE Loss    │    │  Consistency │    │   Loss           │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 核心机制

1. **Latent Loop（潜在循环）**: 将问题编码为隐藏状态种子，迭代 `num_latent` 步生成潜在思维链
2. **知识蒸馏**: 将教师模型的隐藏状态蒸馏到学生模型的答案位置
3. **轨迹约束**: 支持多种几何空间（Euclidean/Hyperbolic）的轨迹一致性损失
4. **LoRA 适配**: 针对不同模型家族（LLaMA/Mistral/Falcon/Qwen/Phi/GPT-2）使用特定的 LoRA 目标模块

---

## 环境安装

### 前置要求

- Python 3.8+
- CUDA 11.8+ (推荐)
- PyTorch 2.0+

### 安装步骤

```bash
# 1. 克隆项目
git clone <repository-url>
cd CODI

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
# .venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp config.env.example config.env  # 如果有示例文件
# 编辑 config.env 设置路径
```

### 主要依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| `torch` | 2.0+ | 深度学习框架 |
| `transformers` | 4.30+ | Hugging Face 模型 |
| `peft` | 0.15+ | LoRA 适配 |
| `datasets` | 3.6+ | 数据集加载 |
| `accelerate` | 1.7+ | 分布式训练 |
| `safetensors` | 0.5+ | 模型权重保存 |

---

## 快速开始

### 1. 配置环境变量

编辑 `config.env` 文件：

```bash
# rebuttal / revision 输出目录
export CODI_RUN_ROOT="/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325"
export CODI_SAVE_DIR="${CODI_RUN_ROOT}/outputs"
export CODI_RESULT_DIR="${CODI_RUN_ROOT}/results"

# 模型路径
export CODI_LLAMA1B_PATH="/path/to/Llama-3.2-1B-Instruct"
export CODI_GPT2_PATH="/path/to/gpt2"

# 数据集路径
export CODI_GSM8K_AUG_PATH="/path/to/GSM8k-Aug"
```

### 2. 训练模型

```bash
# 激活环境
source /path/to/.venv/bin/activate
source config.env

# 基础训练
python train.py \
    --model_name_or_path "${CODI_LLAMA1B_PATH}" \
    --data_name icot \
    --output_dir "${CODI_SAVE_DIR}/my_experiment" \
    --num_latent 6 \
    --per_device_train_batch_size 32 \
    --num_train_epochs 10 \
    --learning_rate 8e-4 \
    --use_lora True \
    --lora_r 128 \
    --lora_alpha 32 \
    --lora_init \
    --bf16

# 带轨迹一致性约束的训练
python train.py \
    --model_name_or_path "${CODI_LLAMA1B_PATH}" \
    --data_name icot \
    --output_dir "${CODI_SAVE_DIR}/trajectory_exp" \
    --num_latent 6 \
    --use_trajectory_consistency True \
    --trajectory_space_type euclidean \
    --trajectory_radius_threshold 2.0 \
    --trajectory_loss_factor 0.1 \
    --use_decoder True
```

### 3. 评估模型

```bash
# 单数据集评估
python test.py \
    --model_name_or_path "${CODI_LLAMA1B_PATH}" \
    --ckpt_dir "/path/to/checkpoint" \
    --data_name gsm8k \
    --batch_size 128 \
    --inf_latent_iterations 6

# 多数据集评估
python test_multi_dataset.py \
    --model_name_or_path "${CODI_LLAMA1B_PATH}" \
    --ckpt_dir "/path/to/checkpoint" \
    --datasets "gsm8k svamp multi-arith" \
    --num_runs 3
```

---

## 使用说明

### 训练流程

1. **数据准备**: 数据集以 JSON 格式存储，包含 `question`、`cot`/`steps`、`answer` 字段
2. **模型初始化**: 加载预训练模型并应用 LoRA 适配
3. **前向传播**: 
   - 编码问题 → 生成初始隐藏状态种子
   - 迭代执行 latent loop（投影 + LM 前向）
   - 拼接答案 token 并计算损失
4. **损失计算**:
   - `ce_loss`: 学生模型在生成答案上的交叉熵损失
   - `distill_loss`: 层级蒸馏损失（学生与教师隐藏状态的距离）
   - `ref_ce_loss`: 教师模型在参考输入上的交叉熵损失
   - `trajectory_loss`: 轨迹一致性损失（可选）

### 评估流程

1. **加载 Checkpoint**: 从 `ckpt_dir` 加载 LoRA 权重
2. **推理生成**:
   - 使用采样/贪婪解码
   - 重复 latent 迭代 `inf_latent_iterations` 次
3. **答案提取**: 通过正则表达式提取数值/选项答案
4. **指标计算**: 计算准确率并保存到结果目录

---

## 目录结构详解

```
CODI/
├── 📄 train.py                    # 主训练脚本
├── 📄 test.py                     # 单数据集评估脚本
├── 📄 test_multi_dataset.py       # 多数据集批量评估脚本
├── 📄 test_baseline.py            # 基线模型（无 LoRA）评估
├── 📄 probe_latent_token.py       # Latent token 探测与分析
├── 📄 config.env                  # 环境配置文件
├── 📄 requirements.txt            # Python 依赖
├── 📄 TESTING_GUIDE.md            # 测试指南
│
├── 📁 src/                        # 核心源代码
│   ├── 📄 model.py                # CODI 模型定义
│   ├── 📄 trajectory_consistency.py   # 轨迹一致性损失（Fréchet mean）
│   ├── 📄 trajectory_acceleration.py  # 二阶平滑（加速度）损失
│   ├── 📄 trajectory_action.py        # 最小作用量（路径能量）损失
│   ├── 📄 trajectory_geodesic.py      # 测地线偏差损失
│   └── 📄 trajectory_hyperbolic.py    # 双曲空间几何实现
│
├── 📁 scripts/                    # 实验脚本集合
│   ├── 📄 train_llama1b_gsm8k-aug.sh      # LLaMA 1B 训练
│   ├── 📄 train_gpt2_gsm8k-aug.sh         # GPT-2 训练
│   ├── 📄 test_llama1b.sh                 # LLaMA 1B 评估
│   ├── 📄 euclidean_*.sh                  # Euclidean 轨迹实验
│   ├── 📄 hyperbolic.sh                   # Hyperbolic 轨迹实验
│   ├── 📄 geodesic.sh                     # 测地线实验
│   ├── 📄 batch_test*.sh                  # 批量测试脚本
│   └── 📄 plot_*.py                       # 结果绘图脚本
│
├── 📁 flip/                       # Coin Flip 数据集实验脚本
│   ├── 📄 filp_codi_base.sh
│   ├── 📄 filp_codi_sircl.sh
│   ├── 📄 filp_simcon_base.sh
│   └── 📄 filp_simcon_sircl*.sh
│
├── 📁 train_on_commen_dataset/    # CommonSenseQA 训练脚本
├── 📁 train_on_multiarith_dataset/ # MultiArith 训练脚本
├── 📁 train_on_svamp_dataset/     # SVAMP 训练脚本
│
├── 📁 local_datasets/             # CODI 实际运行使用的本地 JSON 数据
│   ├── 📁 coin_flip/
│   ├── 📁 multiarith/
│   └── 📁 svamp/
│
├── 📁 SemCoT/                     # 外部参考仓库（主要保留作数据来源/处理参考）
│
├── 📁 plots/                      # 论文图表生成
│   ├── 📄 color_config.py         # 统一颜色配置
│   ├── 📄 plot_ablation.py        # 消融实验图表
│   ├── 📄 plot_gsm8k_comparison.py
│   └── 📄 plot_latent_sweep.py
│
├── 📁 outputs/                    # 训练输出目录
│   ├── 📁 trained/                # 保存的模型 checkpoint
│   ├── 📁 logs/                   # TensorBoard 日志
│   └── 📄 decoded_latent*.txt     # 解码的 latent 日志
│
├── 📁 results/                    # 评估结果目录
│   ├── 📁 models/                 # 按模型组织的结果
│   │   └── 📁 {model_name}/
│   │       └── 📁 {dataset}/
│   │           └── 📁 run_{i}/
│   │               ├── 📄 predictions.json
│   │               └── 📄 metrics.json
│   ├── 📁 datasets/               # 按数据集组织的结果
│   └── 📁 summary/                # 汇总报告
│       ├── 📄 all_results.csv
│       └── 📄 comparison_matrix.csv
│
├── 📁 final_use_model_codi_sim_sircl/  # 最终使用的模型 checkpoint
│   ├── 📁 codi/
│   ├── 📁 codi_sircl/
│   ├── 📁 sim/
│   └── 📁 simcon_sircl/
│
├── 📄 analyze_results.py          # 结果分析与可视化
├── 📄 analyze_latent_visualization.py  # Latent 可视化分析
├── 📄 rebuild_summary.py          # 重建结果汇总
├── 📄 debug_model_output.py       # 模型输出调试
└── 📄 plooot.py                   # 快速绘图工具
```

---

## 核心组件

### 1. CODI 模型 (`src/model.py`)

主模型类，继承自 `torch.nn.Module`：

```python
class CODI(torch.nn.Module):
    """
    核心功能：
    - 封装 Hugging Face CausalLM
    - 扩展词表（pad/BOT/EOT token）
    - 可选的第二解码器路径
    - 投影 MLP（use_prj=True 时启用）
    - LoRA 适配
    """
```

**关键参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `model_name_or_path` | str | 基础模型路径 |
| `full_precision` | bool | True=全精度，False=4bit 量化 |
| `use_decoder` | bool | 是否使用辅助解码器 |
| `use_prj` | bool | 是否使用投影层 |
| `num_latent` | int | Latent 迭代次数 |

**LoRA 目标模块**:

| 模型家族 | 目标模块 |
|----------|----------|
| LLaMA/Mistral/Falcon/Qwen | q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj |
| Phi | q_proj, k_proj, v_proj, dense, fc1, fc2 |
| GPT-2 | c_attn, c_proj, c_fc |

### 2. 轨迹约束损失 (`src/trajectory_*.py`)

#### 2.1 轨迹一致性 (`trajectory_consistency.py`)

基于 Fréchet mean 的质心约束：

```python
# Euclidean 空间
center = X.mean(dim=0)  # 算术平均
dist = ||X - center||_2   # L2 距离

# Hyperbolic (Poincaré ball) 空间
center = karcher_mean(X)  # 迭代 Karcher mean
dist = poincare_distance(X, center)  # 双曲距离
```

#### 2.2 加速度平滑 (`trajectory_acceleration.py`)

约束轨迹的二阶导数（加速度）：

```python
# 速度: v_k = z_{k+1} - z_k
# 加速度: a_k = v_{k+1} - v_k = z_{k+2} - 2*z_{k+1} + z_k
loss = mean(max(0, ||a_k|| - max_acceleration))
```

#### 2.3 最小作用量 (`trajectory_action.py`)

最小化路径能量：

```python
energy = λ_energy * E[||v||²] + λ_length * E[||z - center||²]
```

#### 2.4 测地线偏差 (`trajectory_geodesic.py`)

约束轨迹贴近起点到终点的测地线（仅双曲空间）：

```python
geodesic = compute_geodesic(z_start, z_end, T)
loss = mean(hyperbolic_distance(z, geodesic))
```

### 3. 训练器 (`train.py`)

自定义 `CustomTrainer` 继承自 `transformers.Trainer`：

```python
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch):
        # 计算 step_ratio 用于动态调度
        # 前向传播获取所有损失
        # 记录细粒度日志
```

**损失组合**:
```
total_loss = ce_loss 
           + distill_loss_factor * distill_loss 
           + ref_loss_factor * ref_ce_loss 
           + trajectory_loss_factor * trajectory_loss
           + acceleration_loss_factor * acceleration_loss
           + action_loss_factor * action_loss
           + geodesic_loss_factor * geodesic_loss
```

---

## 配置参数

### ModelArguments

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name_or_path` | mistralai/Mistral-7B-Instruct-v0.2 | 基础模型 |
| `lora_r` | 128 | LoRA rank |
| `lora_dropout` | 0.05 | LoRA dropout |
| `lora_alpha` | 16 | LoRA alpha |
| `full_precision` | True | 全精度训练 |
| `use_decoder` | False | 启用辅助解码器 |
| `ckpt_dir` | None | Checkpoint 目录 |

### TrainingArguments

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_latent` | 5 | Latent 迭代次数 |
| `use_prj` | False | 启用投影层 |
| `prj_dim` | 2048 | 投影层隐藏维度 |
| `distill_loss_factor` | 1.0 | 蒸馏损失权重 |
| `ref_loss_factor` | 1.0 | 参考损失权重 |
| `use_trajectory_consistency` | False | 启用轨迹一致性 |
| `trajectory_space_type` | "euclidean" | 几何空间类型 |
| `trajectory_radius_threshold` | 2.0 | 轨迹半径阈值 |
| `trajectory_loss_factor` | 0.1 | 轨迹损失权重 |
| `inf_latent_iterations` | 1 | 推理时 latent 迭代次数 |

### DataArguments

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_name` | None | 数据集名称 |
| `batch_size` | 1 | 推理 batch size |
| `debug_data` | False | 调试模式（使用小数据集） |

---

## 实验脚本

### 训练脚本示例

```bash
# scripts/train_llama1b_gsm8k-aug.sh
python train.py \
    --output_dir "${SAVE_DIR}" \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --data_name icot \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 8e-4 \
    --num_latent 6 \
    --use_prj True \
    --distill_loss_factor 20
```

### 分布式训练

```bash
torchrun --nnodes 1 --nproc_per_node 2 train.py \
    --model_name_or_path "${CODI_LLAMA1B_PATH}" \
    --data_name coin_flip \
    --use_trajectory_consistency True \
    --trajectory_space_type euclidean
```

### 超参数扫描实验

```bash
# Euclidean 轨迹 - 不同 loss factor
scripts/euclidean_001factor.sh   # factor=0.01
scripts/euclidean_005factor.sh   # factor=0.05
scripts/euclidean_015factor.sh   # factor=0.15

# 不同 radius threshold
scripts/euclidean_smallthrehold.sh  # threshold=1.0
scripts/euclidean_bigthrehold.sh    # threshold=8.0
```

---

## 结果分析

### 生成汇总报告

```bash
# 从 metrics.json 重建汇总
python rebuild_summary.py

# 分析所有结果
python analyze_results.py

# 生成可视化图表
python analyze_results.py --plot
```

### 结果目录结构

```
results/
├── models/
│   ├── euclidean/
│   │   ├── gsm8k/
│   │   │   ├── run_0/
│   │   │   │   ├── predictions.json  # 详细预测
│   │   │   │   ├── metrics.json      # 准确率指标
│   │   │   │   └── trajectory_stats.json
│   │   │   └── run_1/
│   │   └── model_summary.csv
│   └── geodesic/
├── datasets/
│   └── gsm8k/
│       └── all_models.csv    # 所有模型在该数据集的对比
└── summary/
    ├── all_results.csv       # 完整结果记录
    └── comparison_matrix.csv # 模型×数据集矩阵
```

### Latent 可视化分析

```bash
python analyze_latent_visualization.py \
    --results_dir ./results \
    --dataset gsm8k \
    --methods tsne pca
```

生成的图表：
- t-SNE/PCA 降维可视化
- 余弦相似度热力图
- 轨迹聚类分析

---

## 支持的数据集

| 数据集 | 类型 | 答案格式 | HuggingFace ID |
|--------|------|----------|----------------|
| GSM8K | 数学推理 | 数值 | zen-E/GSM8k-Aug |
| GSM-Hard | 数学推理 | 数值 | juyoung-trl/gsm-hard |
| MultiArith | 数学推理 | 数值 | ChilleD/MultiArith |
| SVAMP | 数学推理 | 数值 | ChilleD/SVAMP |
| CommonSenseQA | 常识推理 | 选项 | zen-E/CommonsenseQA-GPT4omini |
| StrategyQA | 策略推理 | 布尔 | ChilleD/StrategyQA |
| AQuA-RAT | 代数推理 | 选项 | deepmind/aqua_rat |
| Coin Flip | 逻辑推理 | 布尔 | 本地 JSON |

---

## 注意事项

1. **路径配置**: 代码中包含绝对路径，部署到新环境时需要修改 `config.env` 或创建符号链接

2. **Tokenizer 一致性**: 训练和评估时 pad/EOS/BOT/EOT token 必须保持一致

3. **`remove_eos` 参数**: 影响 BOT/EOT 前缀方式和答案组装，训练/测试需保持一致

4. **内存管理**: 
   - 使用 `max_token_num` 限制最长序列以避免 OOM
   - 大模型建议使用梯度累积 (`gradient_accumulation_steps`)

5. **Checkpoint 加载**: 评估时 adapter 权重通过 `ckpt_dir` 加载，然后执行 `tie_weights()`

---

## 常见问题

### Q: 训练时出现 OOM

A: 尝试以下方案：
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 减小 `max_token_num`
- 使用 `full_precision=False` 启用 4bit 量化

### Q: 评估准确率为 0

A: 检查以下项：
- `ckpt_dir` 路径是否正确
- `inf_latent_iterations` 是否与训练时的 `num_latent` 匹配
- `remove_eos` 设置是否与训练一致

### Q: 如何添加新的模型家族

A: 在 `src/model.py` 中更新 LoRA 目标模块选择逻辑：
```python
if any(name in model_name.lower() for name in ["your_model"]):
    target_modules = ["your", "target", "modules"]
```

---

## 引用

如果您使用了本项目的代码，请引用相关论文。

---

## 许可证

请参阅 [LICENSE](LICENSE) 文件。
