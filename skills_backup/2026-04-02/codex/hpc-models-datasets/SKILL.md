---
name: hpc-models-datasets
description: HPC 通用模型与数据集下载/分发指引，包括 ModelScope/HF Mirror 下载策略、跨服务器分发、完整性校验和离线运行配置。
---

# HPC1 / HPC2 / HPC3 / HPC4 模型与数据集下载/分发指引

更新时间：2026-03-14

## 1. 适用范围

本指引用于 HPC 环境中的通用模型与数据集管理，覆盖：

- 下载（有外网权限的节点）
- 分发（HPC1 / HPC2 / HPC3 / HPC4）
- 完整性校验
- 离线运行配置

> 本文不绑定任何单一项目。

## 2. 目录约定（固定）

统一使用以下缓存根目录：

- HPC1：`/hpc2hdd/home/yhao481/jhupload/cache`
- HPC2：`/hpc2hdd/home/yhao481/jhupload/cache`
- HPC3：`/data/user/yhao481/cache`
- HPC4：`/data/user/user224/cache`

建议子目录结构：

- 模型：`<cache_root>/Models/<model_name>/`
- 数据集：`<cache_root>/data/<dataset_name>/`
- Hugging Face：`<cache_root>/huggingface/`
- ModelScope：`<cache_root>/modelscope/`
- UV：`<cache_root>/uv/`

常用环境变量映射：

```bash
export HF_HOME=<cache_root>/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export MODELSCOPE_CACHE=<cache_root>/modelscope
```

## 3. 下载原则

### 3.1 下载节点选择

- 优先在能联网且磁盘空间充足的一台机器先下载，通常选 HPC2，其次 HPC4。
- HPC1 与 HPC2 使用相同目录约定，也可作为同类下载源。
- HPC3 无法访问公共互联网，不要把 HPC3 当作首选下载源。
- 其他机器通过 `rsync` / `scp` 分发，避免重复外网下载。

### 3.2 网络与镜像

下载前先取消代理：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

下载优先级：

1. 优先使用 ModelScope
2. 若 ModelScope 无该资源，再使用 Hugging Face 镜像

当需要走 Hugging Face 镜像时，再设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 3.3 通用下载模板

先选缓存根目录：

```bash
# HPC1 / HPC2
export CACHE_ROOT=/hpc2hdd/home/yhao481/jhupload/cache

# HPC4
# export CACHE_ROOT=/data/user/user224/cache
```

优先：ModelScope（模型/数据集）

```bash
# 模型（ModelScope）
modelscope download --model <org_or_user>/<model_repo> \
  --local_dir ${CACHE_ROOT}/Models/<model_name>

# 数据集（ModelScope）
modelscope download --dataset <org_or_user>/<dataset_repo> \
  --local_dir ${CACHE_ROOT}/data/<dataset_name>
```

兜底：HF Mirror（仅当 ModelScope 不提供时）

```bash
# 数据集（HF Mirror）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset <org_or_user>/<dataset_repo> \
  --local-dir ${CACHE_ROOT}/data/<dataset_name>

# 模型（HF Mirror）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download <org_or_user>/<model_repo> \
  --local-dir ${CACHE_ROOT}/Models/<model_name>
```

## 4. 分发到其他服务器

以下示例默认在已下载源机器执行。

### 4.1 分发数据集

```bash
# -> HPC1
rsync -avh --delete <source_cache_root>/data/<dataset_name>/ \
  yhao481@hpc1login.hpc.hkust-gz.edu.cn:/hpc2hdd/home/yhao481/jhupload/cache/data/<dataset_name>/

# -> HPC2
rsync -avh --delete <source_cache_root>/data/<dataset_name>/ \
  yhao481@hpc2login.hpc.hkust-gz.edu.cn:/hpc2hdd/home/yhao481/jhupload/cache/data/<dataset_name>/

# -> HPC3
rsync -avh --delete <source_cache_root>/data/<dataset_name>/ \
  yhao481@hpc3login.hpc.hkust-gz.edu.cn:/data/user/yhao481/cache/data/<dataset_name>/

# -> HPC4
rsync -avh --delete <source_cache_root>/data/<dataset_name>/ \
  user224@hpc4login.hpc.hkust-gz.edu.cn:/data/user/user224/cache/data/<dataset_name>/
```

### 4.2 分发模型

```bash
# -> HPC1
rsync -avh --delete <source_cache_root>/Models/<model_name>/ \
  yhao481@hpc1login.hpc.hkust-gz.edu.cn:/hpc2hdd/home/yhao481/jhupload/cache/Models/<model_name>/

# -> HPC2
rsync -avh --delete <source_cache_root>/Models/<model_name>/ \
  yhao481@hpc2login.hpc.hkust-gz.edu.cn:/hpc2hdd/home/yhao481/jhupload/cache/Models/<model_name>/

# -> HPC3
rsync -avh --delete <source_cache_root>/Models/<model_name>/ \
  yhao481@hpc3login.hpc.hkust-gz.edu.cn:/data/user/yhao481/cache/Models/<model_name>/

# -> HPC4
rsync -avh --delete <source_cache_root>/Models/<model_name>/ \
  user224@hpc4login.hpc.hkust-gz.edu.cn:/data/user/user224/cache/Models/<model_name>/
```

## 5. 完整性校验（通用）

### 5.1 数据集文件校验

```bash
test -f <cache_root>/data/<dataset_name>/test.jsonl && echo "test split ok"
test -f <cache_root>/data/<dataset_name>/train.jsonl && echo "train split ok"
```

### 5.2 目录摘要校验

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('<cache_root>/data/<dataset_name>')
print('exists:', p.exists())
if p.exists():
    files = sorted([x.name for x in p.iterdir()])
    print('count:', len(files))
    print('sample:', files[:20])
PY
```

## 6. 离线运行建议

在离线机器或不希望触发重复下载的脚本里建议固定：

```bash
export HF_HOME=<cache_root>/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export MODELSCOPE_CACHE=<cache_root>/modelscope
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

必要时显式指定本地路径，避免代码触发在线拉取。

## 7. 常见问题

- 下载不完整：先检查磁盘空间、网络稳定性，再重试下载。
- 目标机器缺目录：先 `mkdir -p` 再 `rsync`。
- HPC3 上意外触发在线下载：检查离线环境变量和本地缓存路径是否已设置。
- 远端权限问题：确认目标路径所属用户与配额。

## 8. 推荐操作顺序

1. 在 HPC2 或 HPC4 这类可联网机器下载到对应的共享 `cache`。
2. 本地完成完整性校验。
3. 使用 `rsync --delete` 分发到其他机器。
4. 在每台目标机器做文件存在性校验。
5. 在 HPC3 上默认启用离线环境变量。
