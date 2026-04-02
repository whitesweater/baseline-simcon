---
name: hpc-sbatch
description: HPC1/HPC2/HPC3 SLURM 任务提交规范，包括按集群工作目录组织项目、优先使用 uv 管理环境、在 HPC3 离线条件下运行作业，以及常用 sbatch/squeue 命令。
---

# HPC1 / HPC2 / HPC3 任务提交规范（SLURM / sbatch）

## 1) 三台机器关键差异

| 项目 | HPC1 | HPC2 | HPC3 |
|---|---|---|---|
| 登录别名 | `hpc1` | `hpc2` | `hpc3` |
| 项目根目录 | `/hpc2hdd/home/yhao481/jhupload/proj` | `/hpc2hdd/home/yhao481/jhupload/proj` | `/data/user/yhao481/proj` |
| 缓存根目录 | `/hpc2hdd/home/yhao481/jhupload/cache` | `/hpc2hdd/home/yhao481/jhupload/cache` | `/data/user/yhao481/cache` |
| `sbatch` 路径 | 以 `which sbatch` / `sinfo` 实测为准 | 常见为 `/opt/slurm/bin/sbatch`（若 PATH 无命令） | `/usr/bin/sbatch` |
| 网络 | 通常可联网 | 通常可联网 | 离线/受限网络（外网通常不可达） |
| 环境习惯 | 优先项目内 `.venv` | 优先项目内 `.venv` | 优先项目内 `.venv` |

> 建议：优先在 HPC1 / HPC2 上准备依赖和缓存，再迁移到 HPC3 使用。HPC3 默认按离线机器对待。

---

## 2) 提交前先查分区

不要把分区名写死成常识。每次先在目标集群上确认：

```bash
# HPC1 / HPC2 常见需要先补 PATH
export PATH=/opt/slurm/bin:$PATH

which sbatch
sinfo
```

选择作业脚本里的 `#SBATCH -p <partition>` 时，以当前 `sinfo` 输出为准。

> 经验规则：HPC3 往往要求显式申请 GPU 资源，且默认按离线环境准备运行。

---

## 3) 脚本模板

### 3.1 HPC1 / HPC2 CPU 模板（uv 优先）

```bash
#!/bin/bash
#SBATCH -J <job_name>
#SBATCH -p <partition>
#SBATCH -n 16
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /hpc2hdd/home/yhao481/jhupload/proj/<project>

mkdir -p ./logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="./logs/job_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOG_FILE" 2>&1

export UV_CACHE_DIR=/hpc2hdd/home/yhao481/jhupload/cache/uv
test -f .venv/bin/activate
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python main.py
```

### 3.2 HPC1 / HPC2 GPU 模板（uv 优先）

```bash
#!/bin/bash
#SBATCH -J <job_name>
#SBATCH -p <partition>
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /hpc2hdd/home/yhao481/jhupload/proj/<project>

mkdir -p ./logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="./logs/job_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOG_FILE" 2>&1

module load cuda/12.2 2>/dev/null || module load cuda 2>/dev/null || true
export UV_CACHE_DIR=/hpc2hdd/home/yhao481/jhupload/cache/uv
test -f .venv/bin/activate
source .venv/bin/activate
export PYTHONUNBUFFERED=1

nvidia-smi || true
python main.py
```

### 3.3 HPC3 GPU 模板（推荐）

```bash
#!/bin/bash
#SBATCH -J <job_name>
#SBATCH -p <partition>
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/yhao481/proj/<project>

mkdir -p ./logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="./logs/job_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOG_FILE" 2>&1

module load cuda/12.2 2>/dev/null || module load cuda 2>/dev/null || true
export UV_CACHE_DIR=/data/user/yhao481/cache/uv
test -f .venv/bin/activate
source .venv/bin/activate
export PYTHONUNBUFFERED=1

export HF_HOME=/data/user/yhao481/cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub
export MODELSCOPE_CACHE=/data/user/yhao481/cache/modelscope
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

nvidia-smi || true
python main.py
```

---

## 4) 关键规范

### SBATCH 头部参数

| 参数 | 说明 | 示例 |
|---|---|---|
| `-J` | 任务名称 | `#SBATCH -J train_v1` |
| `-p` | 分区 | `#SBATCH -p <partition>` |
| `-n` | CPU 核心数 | `#SBATCH -n 8` |
| `--gres=gpu:N` | GPU 数量 | `#SBATCH --gres=gpu:2` |
| `-D` | 工作目录（绝对路径） | `#SBATCH -D /data/user/yhao481/proj/xxx` |
| `-o` / `-e` | 标准输出/错误 | 推荐设为 `/dev/null`，用 `exec` 重定向到自定义日志 |

### UV 环境激活（默认）

SLURM 环境下默认使用项目内 `.venv`：

```bash
test -f .venv/bin/activate
source .venv/bin/activate
```

如项目仍使用 Conda，再退回绝对路径激活，避免依赖 `conda activate` 的 shell 初始化。

### 工作目录与缓存

`-D` 就是作业运行时 `$PWD`，`./logs` 等相对路径都以此为基准。

建议同时设置：

```bash
# HPC1 / HPC2
export UV_CACHE_DIR=/hpc2hdd/home/yhao481/jhupload/cache/uv

# HPC3
# export UV_CACHE_DIR=/data/user/yhao481/cache/uv
```

---

## 5) 常用命令

```bash
# 提交
sbatch job.sh

# 查询队列
squeue -u yhao481

# 任务详情
scontrol show job <job_id>

# 取消任务
scancel <job_id>

# 历史
sacct -u yhao481 --format=JobID,JobName,State,Elapsed,Start,End
```

### HPC2 命令找不到时

```bash
export PATH=/opt/slurm/bin:$PATH

/opt/slurm/bin/sbatch job.sh
/opt/slurm/bin/sinfo
```
