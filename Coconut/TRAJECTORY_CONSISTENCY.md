# Coconut with Trajectory Consistency Loss

本文档说明如何在 Coconut 项目中使用 Trajectory Consistency Loss 进行训练。

## 概述

Trajectory Consistency Loss 是一种正则化方法，通过约束 latent tokens 在其几何中心（Fréchet mean）的一定半径内，来鼓励更连贯的隐式推理轨迹。

**核心思想**：
- 计算所有 latent token embeddings 的几何中心（欧几里得空间中的算术平均）
- 计算每个 latent token 到中心的 L2 距离
- 对超过 `radius_threshold` 的点施加惩罚

**Loss 公式**：
```
trajectory_loss = mean(max(0, distance - radius_threshold))
```

## 数据集准备

### 1. 下载并处理 GSM8k 数据集

```bash
cd Coconut
bash preprocessing/gsm_icot.bash
```

这将在 `./data/` 目录下生成：
- `gsm_train.json` - 训练集
- `gsm_valid.json` - 验证集
- `gsm_test.json` - 测试集

### 2. 数据格式

JSON 文件格式：
```json
[
  {
    "question": "问题文本",
    "steps": ["步骤1", "步骤2", "..."],
    "answer": "最终答案"
  },
  ...
]
```

### 3. 准备预训练模型

**GPT-2**:
```bash
mkdir -p pretrained
# 下载 GPT-2 到 ./pretrained/gpt2
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; m=AutoModelForCausalLM.from_pretrained('gpt2'); t=AutoTokenizer.from_pretrained('gpt2'); m.save_pretrained('./pretrained/gpt2'); t.save_pretrained('./pretrained/gpt2')"
```

**LLaMA**（可选）:
```bash
# 设置 HuggingFace 镜像（如果在国内）
export HF_ENDPOINT=https://hf-mirror.com

# 下载 LLaMA 模型
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir ./pretrained/llama-3.1-8b
```

## 训练流程

### Stage 0: 训练 Qwen3 CoT-SFT Baseline（可选但推荐）

如果你希望先用更强的 backbone 做纯 CoT-SFT baseline，再继续做 Coconut 或 trajectory consistency 相关实验，可以直接使用新增的 Qwen3 配置：

```bash
cd Coconut
bash scripts/train_cot_qwen3.sh 4
```

对应配置文件：

- `args/gsm_cot_qwen3.yaml`
- `args/gsm_cot_qwen3_eval.yaml`

默认本地模型路径：

- `/data/yhao/rank/models/Qwen3-4B`

默认产出 checkpoint：

- `./ckpts/gsm-qwen3-cot-sft/`

### Stage 1: 训练 Coconut Baseline

首先需要训练一个 Coconut baseline 模型：

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml
```

或者使用已有的 checkpoint。

### Stage 2: 带 Trajectory Consistency 的训练

在 Coconut baseline 的基础上，添加 trajectory consistency loss 继续训练：

```bash
# 使用训练脚本
bash scripts/train_trajectory.sh 8  # 8 GPUs

# 或直接运行
torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_coconut_trajectory.yaml
```

## 配置参数说明

在 `args/gsm_coconut_trajectory.yaml` 中的关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_trajectory_consistency` | `True` | 是否启用 trajectory consistency loss |
| `trajectory_radius_threshold` | `2.0` | 允许的最大半径，超过此半径的点会受到惩罚 |
| `trajectory_loss_factor` | `0.1` | Trajectory loss 的权重系数 |

### 参数调优建议

**`trajectory_radius_threshold`**:
- 值过小：约束过强，可能限制模型表达能力
- 值过大：约束过弱，效果不明显
- 推荐范围：1.0 - 10.0，根据 hidden dimension 和数据集调整

**`trajectory_loss_factor`**:
- 值过大：可能导致训练不稳定
- 值过小：正则化效果不明显
- 推荐：从 0.01 开始，逐步增加到 0.1

## 评测

```bash
# 使用评测脚本
bash scripts/eval_trajectory.sh 8  # 8 GPUs

# 或直接运行
torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_coconut_trajectory_eval.yaml
```

## 文件结构

```
Coconut/
├── trajectory_consistency.py      # Trajectory Consistency Loss 模块
├── coconut.py                     # 修改后的模型（集成 trajectory loss）
├── run.py                         # 修改后的训练脚本（日志支持）
├── args/
│   ├── gsm_cot_qwen3.yaml                # Qwen3-4B CoT-SFT 训练配置
│   ├── gsm_cot_qwen3_eval.yaml           # Qwen3-4B CoT-SFT 评测配置
│   ├── gsm_coconut_trajectory.yaml       # 训练配置
│   └── gsm_coconut_trajectory_eval.yaml  # 评测配置
├── scripts/
│   ├── train_cot_qwen3.sh         # Qwen3-4B CoT-SFT 训练脚本
│   ├── eval_cot_qwen3.sh          # Qwen3-4B CoT-SFT 评测脚本
│   ├── train_trajectory.sh        # 训练脚本
│   └── eval_trajectory.sh         # 评测脚本
└── data/
    ├── gsm_train.json             # 训练数据
    └── gsm_test.json              # 测试数据
```

## 与 CODI 实现的对比

| 特性 | CODI | Coconut |
|------|------|---------|
| 几何空间 | Euclidean + Hyperbolic | Euclidean only |
| 输入维度 | [T, B, D] | [T, B, D] |
| 收集时机 | 迭代生成 latent 时 | 多 pass forward 时 |
| 额外 loss | Acceleration, Action, Geodesic | 无（可扩展） |

## 常见问题

**Q: 训练时 trajectory_loss 一直为 0？**
A: 检查 `max_n_latents` 是否 >= 2，需要至少 2 个 latent token 才能计算有意义的 trajectory loss。

**Q: 训练不稳定？**
A: 尝试降低 `trajectory_loss_factor`（如 0.01）或增大 `trajectory_radius_threshold`。

**Q: 如何监控 trajectory loss？**
A: 训练时会在进度条和 wandb 中显示 `traj_loss`，也可以在 `outputs.trajectory_loss` 获取。

## 引用

如果使用本代码，请引用：
- Coconut 原始论文
- SIM-CoT / CODI 相关工作
