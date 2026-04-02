---
name: hpc-migration
description: 本地工作站与 HPC1/HPC2/HPC3/HPC4 之间的项目迁移规范，优先使用 rsync 分层同步源码、基础模型和共享缓存，并在目标机重建项目内 .venv；当仓库内已存在迁移脚本时优先复用脚本。
---

# HPC 集群迁移操作规范

## 适用范围

本技能用于以下场景：

- 本地工作站 -> HPC1 / HPC2 / HPC3 / HPC4
- HPC1 / HPC2 / HPC4 之间的项目、模型、缓存分发
- HPC3 这种离线目标机的预分发迁移
- 项目内 `.venv` 的本地重建或离线恢复

默认优先级：

- 先同步源码和必要运行资产
- 再同步基础模型和共享缓存
- 最后在目标机重建项目内 `.venv`

## 集群目录约定

| 集群 | 登录主机 | 用户 | 项目根目录 | 缓存根目录 | 备注 |
|------|----------|------|------------|------------|------|
| HPC1 | `hpc1login.hpc.hkust-gz.edu.cn` | `yhao481` | `/hpc2hdd/home/yhao481/jhupload/proj` | `/hpc2hdd/home/yhao481/jhupload/cache` | 和 HPC2 目录习惯一致 |
| HPC2 | `hpc2login.hpc.hkust-gz.edu.cn` | `yhao481` | `/hpc2hdd/home/yhao481/jhupload/proj` | `/hpc2hdd/home/yhao481/jhupload/cache` | 常用下载源或中转源 |
| HPC3 | `hpc3login.hpc.hkust-gz.edu.cn` | `yhao481` | `/data/user/yhao481/proj` | `/data/user/yhao481/cache` | 默认按离线目标机处理 |
| HPC4 | `hpc4login.hpc.hkust-gz.edu.cn` | `user224` | `/data/user/user224/proj` | `/data/user/user224/cache` | 登录节点偏 CPU / 调度 |

## 当前 baseline 项目的推荐迁移路径

如果当前任务是迁移 `/data/yhao/baseline`，并且仓库内存在：

```bash
/data/yhao/baseline/scripts/migrate_baseline_minimal.sh
```

优先直接复用这个脚本，而不是临时手写一长串 rsync。

### 目标目录建议

- HPC1 / HPC2: `/hpc2hdd/home/yhao481/jhupload/proj/baseline`
- HPC3: `/data/user/yhao481/proj/baseline`
- HPC4: `/data/user/user224/proj/baseline`

### 推荐调用方式

在本地工作站执行：

```bash
cd /data/yhao/baseline

# 例：迁到 HPC2
bash scripts/migrate_baseline_minimal.sh \
  --dst-host hpc2 \
  --dst-real /hpc2hdd/home/yhao481/jhupload/proj/baseline \
  --ssh-config /root/.codex/skills/hpc-login-ssh/references/ssh_config_hpc
```

如果目标机暂时不方便装环境，可以先只同步：

```bash
bash scripts/migrate_baseline_minimal.sh \
  --dst-host hpc2 \
  --dst-real /hpc2hdd/home/yhao481/jhupload/proj/baseline \
  --ssh-config /root/.codex/skills/hpc-login-ssh/references/ssh_config_hpc \
  --no-bootstrap-venv \
  --no-verify
```

### 这条路径默认完成的事情

- 同步仓库源码和当前未提交改动
- 排除 `.venv`、历史 checkpoint、wandb、大结果目录
- 单独同步基础模型目录，避免把旧机器软链原样搬过去
- 在目标机保留 `/data/yhao/baseline` 兼容软链
- 可选重建目标机 `.venv` 并做 smoke check

## 通用 rsync 原则

- 默认使用 `rsync`
- 大目录优先走分层同步，不要直接整仓库无脑 `scp -r`
- 默认不要在迁移脚本里加 `--delete`，除非用户明确要求目标端与源端完全镜像
- HPC3 这类离线机器优先迁移模型、缓存和已验证环境，不要现场在线下载

### 机器到机器迁移

在目标机上拉取通常更稳，尤其是拉到 HPC3 这类离线目标机时。

```bash
ssh hpc3
mkdir -p /data/user/yhao481/proj/<project>
rsync -avh \
  yhao481@hpc2login.hpc.hkust-gz.edu.cn:/hpc2hdd/home/yhao481/jhupload/proj/<project>/ \
  /data/user/yhao481/proj/<project>/
```

### 缓存同步

```bash
mkdir -p /data/user/yhao481/cache
rsync -avh \
  yhao481@hpc2login.hpc.hkust-gz.edu.cn:/hpc2hdd/home/yhao481/jhupload/cache/data/ \
  /data/user/yhao481/cache/data/

rsync -avh \
  yhao481@hpc2login.hpc.hkust-gz.edu.cn:/hpc2hdd/home/yhao481/jhupload/cache/Models/ \
  /data/user/yhao481/cache/Models/
```

少量文件时可退回 `scp`：

```bash
scp -p yhao481@hpc2login.hpc.hkust-gz.edu.cn:<source_file> <target_path>
scp -rp yhao481@hpc2login.hpc.hkust-gz.edu.cn:<source_path> <target_path>
```

## 环境恢复原则

- 目标机优先重建项目内 `.venv`
- 如果项目明确是 `uv` 路线，优先 `uv sync`
- 如果仓库已经提供 `pyproject.toml` 且项目迁移脚本会自动重建 `.venv`，优先走仓库脚本
- 只有 legacy 项目才继续走 conda 打包迁移

### uv 路线

```bash
cd <project>
export UV_CACHE_DIR=<cache_root>/uv
uv sync
source .venv/bin/activate
python -V
```

在 HPC3 这类离线机器上建议同时带上 uv 缓存：

```bash
rsync -avh <source_cache_root>/uv/ <target_cache_root>/uv/
export UV_CACHE_DIR=<target_cache_root>/uv
uv sync --offline
```

### 离线缓存变量

```bash
export HF_HOME=/data/user/yhao481/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export MODELSCOPE_CACHE=/data/user/yhao481/cache/modelscope
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### legacy conda 路线

仅当项目明确是 Conda 路线时再用：

```bash
conda activate <env_name>
conda pack -n <env_name> -o <env_name>.tar.gz
```

## 注意事项

- HPC3 默认按离线机器处理。
- 当前 baseline 项目优先使用仓库内 `scripts/migrate_baseline_minimal.sh`。
- 模型和缓存尽量单独同步，不要让目标机在线重下。
- 迁移前先在源端做 smoke check，迁移后在目标端立即验证 `hostname`、`whoami`、`pwd`、环境导入和项目入口。
