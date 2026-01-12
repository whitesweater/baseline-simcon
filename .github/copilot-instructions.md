# SIM-CoT: Supervised Implicit Chain-of-Thought

## Project Overview

This is a research codebase implementing **SIM-CoT** (Supervised Implicit Chain-of-Thought), a method that adds step-level supervision to implicit CoT training via an auxiliary decoder. The project has two main baseline implementations:

- **Coconut/**: Original Coconut baseline with SIM-CoT extensions (GPT-2, LLaMA models)
- **CODI/**: CODI baseline with decoder-based SIM-CoT (LLaMA 1B/3B/8B)

## Environment Setup

### Proxy Configuration (Required)
**All network operations in this project require proxy configuration:**

```bash
# Set environment variables for proxy
export http_proxy=http://127.0.0.1:3128
export https_proxy=http://127.0.0.1:3128
export HTTP_PROXY=http://127.0.0.1:3128
export HTTPS_PROXY=http://127.0.0.1:3128
```

**Apply proxy for specific operations:**
- **HuggingFace model downloads**: Proxy is automatically used via environment variables
- **pip/conda installs**: `pip install --proxy http://127.0.0.1:3128 <package>`
- **git operations**: `git config --global http.proxy http://127.0.0.1:3128`
- **wget/curl**: `wget -e use_proxy=yes -e http_proxy=http://127.0.0.1:3128 <url>`

Add proxy exports to your `~/.bashrc` or `~/.zshrc` for persistent configuration.

### Network Restrictions & Mirrors
- This HPC environment cannot access the public Internet directly. Always use the proxy above or internal mirrors.
- Mirror portal (web): https://nexus.hpc.hkust-gz.edu.cn/
- Ensure the domain `harbor.internal.com` is resolvable. If not, add DNS mapping via hosts on login nodes.

#### PyPI via Internal Mirror
- Global config inside a virtualenv:
  ```bash
  pip config set global.index-url http://harbor.internal.com:8081/repository/pypi-hkust/simple
  pip config set install.trusted-host harbor.internal.com
  ```
- Temporary per-install:
  ```bash
  pip install <pkg> --index-url http://harbor.internal.com:8081/repository/pypi-hkust/simple --trusted-host harbor.internal.com
  ```
- List available versions:
  ```bash
  pip index versions <pkg> --index-url http://harbor.internal.com:8081/repository/pypi-hkust/simple --trusted-host harbor.internal.com
  ```

#### Conda via Internal Mirror
- Global config:
  ```bash
  conda config --remove channels defaults
  conda config --add channels http://harbor.internal.com:8081/repository/conda-hkust/main
  conda config --add channels http://harbor.internal.com:8081/repository/conda-hkust/free
  conda config --add channels http://harbor.internal.com:8081/repository/conda-hkust/msys2
  conda config --add channels http://harbor.internal.com:8081/repository/conda-hkust/pro
  conda config --add channels http://harbor.internal.com:8081/repository/conda-hkust/r
  ```
- Search available versions:
  ```bash
  conda search <pkg>
  ```

#### uv usage with Mirrors
- uv reads indexes from `pyproject.toml`; this repo includes internal PyPI mirrors under `[[tool.uv.index]]`.
- You can also install via uv’s pip frontend with mirror flags:
  ```bash
  uv pip install -r requirements.txt \
    --index-url http://harbor.internal.com:8081/repository/pypi-hkust/simple \
    --trusted-host harbor.internal.com
  ```
- If uv tries to download a managed CPython and fails, disable managed Python and use your local interpreter:
  ```bash
  export UV_NO_MANAGED_PYTHON=1
  export UV_PYTHON="$(which python)"
  uv sync --python "$UV_PYTHON"
  ```

## Architecture Patterns

### Latent Token Mechanism
- Models use **special latent tokens** (marked by `latent_token_id`) to represent implicit reasoning steps
- Coconut fills latent embeddings iteratively using previous hidden states in multi-pass forward loops
  - Each forward pass processes one latent token, using KV cache from previous passes
  - `gen_forward_cnt` tracks total forward passes (equals `num_latent_tokens + 1`)
- CODI uses projection modules (`prj_in`/`prj_out`) to transform latent embeddings before auxiliary decoder processing
- Special tokens in CODI: `bot_id` (begin-of-thought), `eot_id` (end-of-thought), `pad_token_id`
- Coconut special tokens: `<|latent|>`, `<|start-latent|>`, `<|end-latent|>` added to tokenizer

### Two-Stage Training (Coconut)
1. **Stage 1**: Train Coconut baseline to expand vocabulary with latent tokens
   ```bash
   cd Coconut
   torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_coconut.yaml
   ```
   - Config: `mode: coconut_baseline`, `training_method: only_base_causallm`
   - Progressively increases latent token count via `c_thought`, `epochs_per_stage`, `max_latent_stage`
   
2. **Stage 2**: Continue training with SIM-CoT auxiliary decoder
   ```bash
   torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_simcot.yaml
   ```
   - Select a checkpoint from Stage 1 that has expanded to predefined implicit tokens
   - Configure `load_model_path` in YAML to point to this checkpoint
   - Config: `mode: coconutgpt_same_word_embedding`, `training_method: full`

### CODI Architecture Details
- **Main model** (`self.codi`): Base LLM with LoRA adapters for efficient training
- **Decoder** (`self.decoder`): Auxiliary model to decode latent tokens into readable reasoning steps
- **Projection layers**: `pj_in` and `pj_out` handle dimension mismatches between main/decoder models
- **Loss components**:
  - `ce_loss`: Cross-entropy loss on final answers
  - `distill_loss`: Knowledge distillation from reference CoT steps (default: SmoothL1, configurable via `distill_loss_type`)
  - `explain_loss`: Auxiliary decoder's step prediction loss
  - `ref_ce_loss`: Teacher model loss for reference
  - `trajectory_loss` (optional): Fréchet mean-based trajectory consistency loss

## Critical Developer Workflows

### Running Training (Coconut)
```bash
cd Coconut
torchrun --nnodes 1 --nproc_per_node 8 run.py args/<config>.yaml
```
- YAML configs in `args/` control all hyperparameters
- Key config fields: `c_thought` (latent count), `epochs_per_stage`, `max_latent_stage`, `mode`
- Set `coconutgpt: True` and `mode: coconutgpt_same_word_embedding` for SIM-CoT variant

### Running Training (CODI)
```bash
cd CODI
bash scripts/train_llama3b_gsm8k-aug-decoder-2.sh
```
- Scripts in `scripts/` contain full training configurations
- Key args: `--num_latent` (number of implicit tokens), `--use_decoder True` (enable SIM-CoT)
- Adjust `--distill_loss_factor` and `--explain_loss_factor` to balance loss components
- Models saved to `--output_dir` with experiment name `--expt_name`

### Evaluation
**Coconut**:
```bash
torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_simcot_eval.yaml
```

**CODI**:
```bash
bash CODI/scripts/test_llama3b-copy.sh
```
- Set `--ckpt_dir` to trained checkpoint path
- Use `--greedy True` for deterministic evaluation
- `--inf_latent_iterations` controls iterative latent refinement during inference

### Data Preprocessing
Coconut expects JSON format with structure:
```json
{"question": "...", "steps": ["step1", "step2", ...], "answer": "..."}
```
- Convert iCoT text format: `cd Coconut && python preprocessing/gsm_icot.py <split>`
- Text format: `question || step1 step2 ... ## answer`
- Dataset loading in [Coconut/dataset.py](Coconut/dataset.py) tokenizes each component separately
- CODI uses HuggingFace datasets format, loaded via `load_dataset()` from `data_name` argument

## Project-Specific Conventions

### Custom Trainer (CODI)
- `CustomTrainer` in [train.py](CODI/train.py) extends HuggingFace `Trainer` to log multiple loss components
- Passes `step_ratio` (current_step/total_steps) to model for dynamic loss weighting
- Loss logging at `--logging_steps` intervals includes all components separately

### Multi-Pass Forward (Coconut)
- [coconut.py](Coconut/coconut.py) implements iterative forward passes (one per latent token)
- KV cache reused across passes for efficiency
- `gen_forward_cnt` tracks total forward passes (important for debugging)

### Distributed Training
- **Coconut**: Uses PyTorch DDP/FSDP, requires `torchrun` launcher
- **CODI**: Single-GPU or data-parallel via HuggingFace Trainer
- Coconut wraps LlamaDecoderLayer/GPT2Block for FSDP auto-wrapping

### LoRA Integration
- CODI uses PEFT library LoRA by default (`--use_lora True`)
- Target modules auto-selected based on model type (LLaMA: q/k/v/o_proj + FFN, GPT-2: c_attn/c_proj/c_fc)
- `lora_init` flag controls zero/gaussian init vs loading from checkpoint

### Configuration Management
- **Coconut**: YAML files with `Config` wrapper class ([Coconut/utils.py](Coconut/utils.py))
  - Supports checkpoint resumption via `resume` field and auto-detection of interrupted runs
  - `load_model_path` can load base models into Coconut structure or continue from previous Coconut checkpoints
- **CODI**: HuggingFace dataclass-based args (`ModelArguments`, `DataArguments`, `TrainingArguments`)
- Both use deterministic seeding via `set_seed()`

### Trajectory Consistency (CODI Optional Feature)
- Enable with `--use_trajectory_consistency True` to add Fréchet mean-based regularization
- Constrains latent embeddings to stay within `trajectory_radius_threshold` of geometric center
- Use `trajectory_space_type: euclidean` (recommended) or `hyperbolic` (experimental, unstable)
- Controlled via `--trajectory_loss_factor` (default: 0.1)
- See [CODI/TRAJECTORY_CONSISTENCY_README.md](CODI/TRAJECTORY_CONSISTENCY_README.md) for details

## Key Files and Their Roles

- [Coconut/coconut.py](Coconut/coconut.py): Core Coconut model with latent token iteration logic
- [Coconut/run.py](Coconut/run.py): Training/eval orchestration, FSDP setup, checkpoint management
- [Coconut/dataset.py](Coconut/dataset.py): Data loading with custom collators for latent tokens
- [CODI/src/model.py](CODI/src/model.py): CODI model with decoder, projection, and multi-loss training
- [CODI/train.py](CODI/train.py): Training script with CustomTrainer for multi-loss logging
- [CODI/test.py](CODI/test.py): Evaluation script for math reasoning benchmarks (GSM8K, SVAMP, etc.)

## Common Pitfalls

- **Checkpoint Loading**: Coconut requires `load_model_path` for Stage 2; CODI uses `--ckpt_dir` for inference
- **Attention Masks**: Set `fix_attn_mask: True` in CODI to correctly handle reference attention masks
- **Token Alignment**: Ensure special token IDs (latent/bot/eot) match between training and inference
- **OOM Prevention**: Use `--max_token_num` in CODI to filter overly long sequences
- **Loss Balancing**: CODI's multiple loss factors require tuning; start with `distill_loss_factor=20`, `explain_loss_factor=1.0`
