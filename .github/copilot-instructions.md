# SIM-CoT: Supervised Implicit Chain-of-Thought

## Project Overview
研究代码库，实现 **SIM-CoT**（通过辅助 decoder 为隐式 CoT 添加 step-level 监督）。两个主要 baseline：
- **Coconut/**：原始 Coconut + SIM-CoT 扩展（GPT-2、LLaMA）
- **CODI/**：CODI baseline + decoder-based SIM-CoT（LLaMA 1B/3B/8B）

文档入口：`README.md`（仓库总指南）→ `CODI/README.md` → `CODI/TESTING_GUIDE.md`

## 多机配置（重要）
CODI 通过 `config.env` 支持多机部署，每台机器维护自己的配置：
```bash
cd CODI
cp config.env.example config.env  # 创建本地配置
# 编辑 config.env，填入本机路径
```

关键环境变量：
| 变量 | 说明 |
|------|------|
| `CODI_SAVE_DIR` | checkpoint/日志输出目录 |
| `CODI_LLAMA1B_PATH` | LLaMA-1B 模型路径 |
| `CODI_LLAMA3B_PATH` | LLaMA-3B 模型路径 |
| `CODI_GSM8K_AUG_PATH` | GSM8k-Aug 数据集路径 |
| `CODI_CACHE_DIR` | 预处理数据缓存目录 |
| `CODI_CKPT_DIR` | 评测时的 checkpoint 路径 |
| `HF_ENDPOINT` | HuggingFace 镜像（中国用 `https://hf-mirror.com`）|

**注意**：`config.env` 已加入 `.gitignore`，不会被 git 追踪。

## 网络/镜像配置（HPC 环境）
```bash
# 代理
export http_proxy=http://127.0.0.1:3128
export https_proxy=http://127.0.0.1:3128

# PyPI 镜像
pip config set global.index-url http://harbor.internal.com:8081/repository/pypi-hkust/simple
pip config set install.trusted-host harbor.internal.com
```

## 核心工作流

### CODI 训练
```bash
cd CODI
bash scripts/euclidean-train_llama1b_gsm8k-aug-decoder-trajectory.sh
```
关键参数：`--num_latent`（隐式 token 数）、`--use_decoder True`（启用 SIM-CoT）、`--distill_loss_factor`、`--explain_loss_factor`

### CODI 评测
```bash
bash scripts/test_llama1b-copy.sh
# 或设置 CODI_CKPT_DIR 后运行
```

### Coconut 两阶段训练
1. **Stage 1**：训练 Coconut baseline
   ```bash
   cd Coconut && torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_coconut.yaml
   ```
2. **Stage 2**：加载 Stage 1 checkpoint，继续 SIM-CoT 训练
   ```bash
   torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_simcot.yaml
   ```
   需在 YAML 中设置 `load_model_path` 指向 Stage 1 的 checkpoint。

## 架构要点

### Latent Token 机制
- CODI：`bot_id`（begin-of-thought）、`eot_id`（end-of-thought）标记隐式推理段
- Coconut：`<|latent|>`、`<|start-latent|>`、`<|end-latent|>` 特殊 token
- 投影层 `prj_in`/`prj_out` 处理 main model 与 decoder 之间的维度映射

### 多 Loss 训练（CODI）
- `ce_loss`：最终答案的交叉熵
- `distill_loss`：对齐 teacher 的隐藏状态（默认 SmoothL1）
- `explain_loss`：辅助 decoder 预测推理步骤
- `trajectory_loss`（可选）：Fréchet mean 约束 latent 分布

## 关键文件
- [CODI/src/model.py](CODI/src/model.py)：CODI 模型、投影层、多 loss 训练
- [CODI/train.py](CODI/train.py)：CustomTrainer 多 loss 日志
- [CODI/test.py](CODI/test.py)：GSM8K/SVAMP 等评测
- [Coconut/coconut.py](Coconut/coconut.py)：Coconut 多 pass forward 逻辑
- [Coconut/run.py](Coconut/run.py)：FSDP 分布式训练

## 常见问题
- **OOM**：用 `--max_token_num` 过滤过长样本
- **Token 对齐**：确保 train/test 的 special token ID 一致
- **Loss 平衡**：CODI 推荐 `distill_loss_factor=20`，`explain_loss_factor=1.0`
- **Checkpoint 加载**：Coconut Stage 2 用 `load_model_path`；CODI 评测用 `--ckpt_dir`
