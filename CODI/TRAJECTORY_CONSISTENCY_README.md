# 轨迹一致性损失 (Trajectory Consistency Loss)

## 概述

基于Fréchet均值的轨迹一致性约束，通过限制所有推理步骤（latent tokens）到几何中心的距离，防止推理偏离主题，类似物理学中的向心力约束。

## 核心思想

将推理轨迹视为一个点云，要求所有推理步骤围绕一个"主题中心"分布：

```
Loss = (1/K) * Σ max(0, d(z_k, center) - radius_threshold)
```

其中：
- `z_k` 是第k个latent embedding
- `center` 是Fréchet mean（几何中心）
- `radius_threshold` 是允许的最大半径
- `d(·,·)` 是距离函数（欧式或双曲）

## 支持的空间类型

### 1. 欧式空间 (Euclidean)

- **中心计算**: 简单的算术平均
- **距离度量**: L2范数
- **适用场景**: 一般推理任务，线性结构

```python
center = mean(latent_tokens)
distance = ||z_k - center||₂
```

### 2. 双曲空间 (Hyperbolic)

- **中心计算**: Karcher mean（通过梯度下降迭代）
- **距离度量**: Poincaré球模型距离
- **适用场景**: 层次化/树形推理结构
- **注意**: 当前双曲空间实现存在数值稳定性问题，**推荐使用欧式空间**

```python
center = argmin_c Σ d_hyperbolic(z_k, c)²
distance = d_hyperbolic(z_k, center)
```

## 使用方法

### 训练参数

```bash
python train.py \
    --use_trajectory_consistency True \
    --trajectory_space_type euclidean \  # 或 hyperbolic
    --trajectory_radius_threshold 2.0 \
    --trajectory_loss_factor 0.1 \
    --trajectory_curvature -1.0  # 仅双曲空间需要
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_trajectory_consistency` | bool | False | 是否启用轨迹一致性损失 |
| `trajectory_space_type` | str | "euclidean" | 空间类型: "euclidean" (推荐) 或 "hyperbolic" (实验性) |
| `trajectory_radius_threshold` | float | 2.0 | 允许的最大距离半径 |
| `trajectory_loss_factor` | float | 0.1 | 损失权重系数 |
| `trajectory_curvature` | float | -1.0 | 双曲空间曲率常数 |

## 示例脚本

### 欧式空间训练

```bash
bash scripts/train_with_trajectory_euclidean.sh
```

### 双曲空间训练

```bash
bash scripts/train_with_trajectory_hyperbolic.sh
```

## 测试验证

运行测试脚本验证实现：

```bash
python test_trajectory_consistency.py
```

测试包括：
- ✓ 欧式空间Frechet均值计算
- ✓ 距离约束损失计算
- ✓ 梯度反向传播
- ✓ 与CODI训练流程集成

**注意**: 双曲空间测试已禁用，因为当前实现存在数值稳定性问题。推荐使用欧式空间。

## 技术细节

### 欧式空间

**Fréchet均值**:
```python
center = (1/K) * Σ z_k
```

**距离**:
```python
d(x, y) = ||x - y||₂
```

### 双曲空间

**Fréchet均值** (迭代算法):
```python
初始化: c₀ = project_to_hyperbolic(mean(z_k))
迭代: c_{t+1} = exp_c_t(-η * Σ log_c_t(z_k))
```

**Poincaré距离**:
```python
d(x, y) = arcsinh(sqrt(c * ||x-y||² / ((1-c||x||²)(1-c||y||²))))
```

**指数映射** (exp_p):
```python
exp_p(v) = p ⊕ tanh(√c||v||/2) * v/||v||
```

**对数映射** (log_p):
```python
log_p(x) = (2/√c) * arctanh(√c||(-p)⊕x||) * ((-p)⊕x)/||(−p)⊕x||
```

## 与CODI集成

在`src/model.py`的`CODI.forward()`中：

```python
# 1. 初始化收集器
latent_embeddings_for_consistency = []

# 2. 在latent迭代中收集
for i in range(num_latent):
    outputs = self.codi(inputs_embeds=latent_embd, ...)
    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
    
    if self.use_trajectory_consistency:
        latent_embeddings_for_consistency.append(latent_embd.squeeze(1))

# 3. 计算损失
trajectory_loss = self.trajectory_consistency_loss(latent_embeddings_for_consistency)
total_loss += trajectory_loss_factor * trajectory_loss
```

## 监控与调试

训练时会记录以下指标：

```
loss=X.XX, ce_loss=X.XX, distill_loss=X.XX, trajectory_loss=X.XX
```

可在TensorBoard中查看：
- `trajectory_loss`: 轨迹一致性损失值
- 如损失持续为0，考虑减小`trajectory_radius_threshold`
- 如损失过大，考虑增大阈值或减小权重因子

## 超参数调优建议

### 欧式空间
- `radius_threshold`: 1.0 - 3.0（根据hidden_dim调整）
- `loss_factor`: 0.05 - 0.2

### 双曲空间（实验性，不推荐）
- `radius_threshold`: 0.5 - 2.0（双曲距离增长较快）
- `loss_factor`: 0.05 - 0.15
- `curvature`: -1.0（标准配置）
- **注意**: 存在数值稳定性问题，建议使用欧式空间

## 参考文献

1. "Intrinsic Statistics on Riemannian Manifolds: Basic Tools for Geometric Measurements"
2. "Hyperbolic Image Embeddings" - NIPS 2018
3. "A Discriminative Feature Learning Approach for Deep Face Recognition" - CVPR 2016
4. "Fast Computation of Wasserstein Barycenters" - ICML 2014

## 文件结构

```
src/
  trajectory_consistency.py       # 核心实现
  model.py                         # CODI集成
train.py                          # 训练脚本
test_trajectory_consistency.py   # 测试脚本
scripts/
  train_with_trajectory_euclidean.sh   # 欧式空间训练
  train_with_trajectory_hyperbolic.sh  # 双曲空间训练
```
