---
name: hpc3-slurm
description: "Use when: connecting to HKUST-GZ HPC3, fixing SSH host key mismatch, clearing known_hosts fingerprints, reconnecting to hpc3login.hpc.hkust-gz.edu.cn, choosing acd_u or acd_ue queues, sizing GPU or CPU requests, using sbatch or srun, and managing Slurm jobs on the offline HPC3 cluster."
argument-hint: "例如：清除指纹并重连后，在 HPC3 提交 1 卡训练任务"
---

# HPC3 SSH + Slurm Workflow

面向 HKUST-GZ HPC3 的可复用工作流，覆盖从 SSH 指纹修复、重新连接，到队列选择、脚本提交、交互作业和作业管理。

官方参考：
- HPC3 作业与队列：<https://docs.hpc.hkust-gz.edu.cn/docs/hpc3/slurm/queue>
- HPC3 校内软件镜像仓：<https://docs.hpc.hkust-gz.edu.cn/docs/hpc3/on-campus-repository>

## When To Use

- 用户说要连 HPC3，但遇到 `REMOTE HOST IDENTIFICATION HAS CHANGED`
- 需要“清除指纹并重新连接”到 `hpc3login.hpc.hkust-gz.edu.cn`
- 需要在 HPC3 上写 `sbatch` 或 `srun` 命令
- 需要在 `acd_u`、`acd_ue`、`emergency_acd` 之间选队列
- 需要查看作业、诊断 PENDING 原因、取消或恢复任务
- 需要在 HPC3 离线环境下安装 Python 包或配置镜像仓

## Cluster Facts

| Item | Value |
|---|---|
| Login host | `hpc3login.hpc.hkust-gz.edu.cn` |
| User | `yhao481` |
| Project root | `/data/user/yhao481/proj` |
| Cache root | `/data/user/yhao481/cache` |
| Internet | 默认按离线/受限外网处理 |

## Queue Cheat Sheet

用 `sinfo -l` 作为实时真值；下面是文档中的默认规则。

| Queue | Type | Priority | Quota | Default wall-time |
|---|---|---|---|---|
| `acd_u` | GPU shared | low | CPU 128 cores, GPU 16 cards | 7 days |
| `acd_ue` | GPU exclusive | mid | CPU 128 cores, GPU 16 cards | 7 days |
| `emergency_acd` | GPU emergency | high | CPU 128 cores, GPU 16 cards | 7 days |

快速选择规则：

- 成本优先、普通训练：`acd_u`
- 想减少共享干扰，整节点独占：`acd_ue`
- 紧急抢高优先级：`emergency_acd`

提交前必须先在 HPC3 登录节点执行：

```bash
sinfo -l
```

## Workflow

### 1. 修复 SSH 指纹并重新连接

当 SSH 报错 `REMOTE HOST IDENTIFICATION HAS CHANGED` 时，不要直接忽略。先删除本机旧指纹，再重新连接。

```bash
ssh-keygen -f ~/.ssh/known_hosts -R hpc3login.hpc.hkust-gz.edu.cn
ssh yhao481@hpc3login.hpc.hkust-gz.edu.cn
```

如果你想先验证网络层是否通：

```bash
getent hosts hpc3login.hpc.hkust-gz.edu.cn
timeout 5 bash -lc 'echo > /dev/tcp/hpc3login.hpc.hkust-gz.edu.cn/22' && echo ok
```

连接成功后立刻确认：

```bash
hostname
whoami
pwd
cd /data/user/yhao481/proj
```

完成标准：

- 能成功 SSH 到 `hpc3login`
- `whoami` 输出 `yhao481`
- 当前目录切到 `/data/user/yhao481/proj` 或其子目录

### 2. 先确认队列和节点状态

不要把旧经验写死成固定分区。先看当前队列和节点：

```bash
sinfo -l
sinfo -N -p acd_u,acd_ue,emergency_acd -O "NodeList:18,Partition:15,StateLong:15,Gres:20,GresUsed:20,CPUsState:20"
```

如果要判断能否立刻开跑，再补：

```bash
squeue -u $USER
squeue -t PENDING -o "%.10i %.8u %.18P %.6D %.6C %.15b %.25R %.20V"
```

判断规则：

- 单机作业能否立即启动，关键看单节点剩余 GPU 数
- `Reason=Resources` 或 `Reason=Priority` 才算真实资源排队
- `draining` 节点不要计入可用容量

### 3. 选择提交方式

#### 普通批处理

推荐用脚本模式，方便复用和记录。

```bash
sbatch my_job.sh
```

最小 GPU 模板：

```bash
#!/bin/bash
#SBATCH -p acd_u
#SBATCH -J my_train
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH -D /data/user/yhao481/proj/<project>

set -euo pipefail
mkdir -p logs

source .venv/bin/activate
export PYTHONUNBUFFERED=1

python your_script.py
```

对应命令行模式：

```bash
sbatch -p acd_u -o output_%j.txt -e err_%j.txt -n 8 --gres=gpu:1 job_script.sh
```

#### 多节点并行作业

用于分布式训练。先确认 `--nodes`、`--ntasks-per-node` 和 `--gres=gpu:<N>` 是否一致。

```bash
#!/bin/bash
#SBATCH -p acd_u
#SBATCH --job-name=speed_test
#SBATCH -o /data/user/yhao481/proj/<project>/logs/%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

module load anaconda3
module load cuda/12.4

export NCCL_SOCKET_IFNAME=vlan0.2135
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600

python your_script.py
```

#### 数组作业

```bash
#!/bin/bash
#SBATCH -p acd_u
#SBATCH -o output_%A_%a.txt
#SBATCH -e error_%A_%a.txt
#SBATCH -n 1
#SBATCH --array=1-10

PARAM=$SLURM_ARRAY_TASK_ID
python your_script.py "$PARAM"
```

#### 交互式作业

```bash
srun -p acd_u -n 4 --mem=8G --gres=gpu:1 --time=01:00:00 --pty bash
```

### 4. 在 HPC3 离线环境准备依赖

HPC3 默认按受限网络处理。优先复用已有 `.venv`、模型缓存和数据缓存；缺包时走校内镜像仓。

PyPI 虚拟环境全局配置：

```bash
pip config set global.index-url http://harbor.internal.com:8081/repository/pypi-hkust/simple
pip config set install.trusted-host harbor.internal.com
```

临时安装：

```bash
pip install <pkg> --index-url http://harbor.internal.com:8081/repository/pypi-hkust/simple --trusted-host harbor.internal.com
```

查看可安装版本：

```bash
pip index versions <pkg> --index-url http://harbor.internal.com:8081/repository/pypi-hkust/simple --trusted-host harbor.internal.com
```

作业脚本里建议显式设置缓存：

```bash
export HF_HOME=/data/user/yhao481/cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 5. 查看与管理作业

```bash
# 我的等待/运行/挂起作业
squeue -u yhao481

# 历史作业
sacct -u yhao481 --array

# 查看详情和 PENDING 原因
scontrol show job <jobid>

# 查看数组子任务
scontrol show job <jobid>_<task_id>

# 挂起 / 恢复 / 取消
scontrol suspend <jobid>
scontrol resume <jobid>
scancel <jobid>
```

PENDING 诊断优先顺序：

1. 看 `scontrol show job <jobid>` 里的 `Reason=`
2. 看目标队列是否有空闲节点和空闲 GPU
3. 看自己是否超配额或请求过大

## Resource Sizing Rules

- 1 张 GPU 训练：先用 `acd_u`，`-n 8` 起步
- 2 到 4 张 GPU：先用 `acd_u`，并把 CPU 提到 8 到 16 核
- 整节点独占、通信敏感任务：改用 `acd_ue`
- 只做快速调试：优先 `srun` 交互，而不是直接提交长作业

默认检查项：

- `#SBATCH -D` 必须是绝对路径
- `logs/` 目录要能创建
- `.venv/bin/activate` 必须存在，或改成明确的环境激活方式
- GPU 作业必须显式写 `--gres=gpu:<N>`
- 多节点作业必须检查 NCCL 变量和网卡配置

## Completion Checklist

在回答用户“已经可以在 HPC3 上提交任务了吗”之前，至少确认：

1. SSH 指纹问题已清理，且可以重新登录
2. 已用 `sinfo -l` 确认可用队列
3. 作业脚本含有正确的 `#SBATCH -p`、`-n`、`--gres`、`-D`
4. HPC3 所需缓存和离线环境变量已明确
5. 用户知道怎么用 `squeue`、`scontrol show job`、`scancel` 管理任务

## Suggested Responses

当用户问“帮我在 HPC3 提交一个 1 卡任务”时，按这个顺序处理：

1. 先确认 SSH 是否正常；若有 host key 冲突，先清理指纹
2. 登录后执行 `sinfo -l`，确认队列
3. 选择 `acd_u` 或 `acd_ue`
4. 生成 `sbatch` 脚本，填入绝对路径和日志目录
5. 如果涉及安装包，优先用校内镜像仓，不假设外网可用
6. 提交后返回 `jobid`，并给出 `squeue` 和 `scontrol show job` 的跟踪命令