# Rebuttal 分析工具与结果索引

更新时间：2026-03-28

本文档将 rebuttal 期间新增的 5 个分析脚本、对应的产出结果、以及它们各自回答的审稿问题整理在一起。

---

## 总览：脚本 ↔ 审稿问题 映射

| # | 脚本 | 回答的审稿问题 | 核心产出目录 |
|---|------|--------------|-------------|
| 1 | `analyze_latent_collapse.py` | 防 collapse 的直接证据 | `results/reviewer_boundary_simcon_t16/latent_collapse/` |
| 2 | `analyze_centroid_reference.py` | centroid = 语义锚点的解释与证据 | `results/reviewer_boundary_simcon_t16/centroid_reference/` |
| 3 | `analyze_sircl_failure_modes.py` | 轨迹已错→centroid 也错的适用边界 | `results/reviewer_boundary_simcon_t16/failure_modes/` |
| 4 | `plot_scaling_stability.py` | 训练时 scaling：补齐 no-SIRCL matched baseline | `results/rebuttal_scaling_{simcon,codi}_20260328/` |
| 5 | `analyze_trajectory_by_correctness.py` | 只分析 all-correct 子集与失败样本分析 | `results/reviewer_boundary_simcon_t16/trajectory_correctness/` |

---

## 1. `analyze_latent_collapse.py` — 防 collapse 的直接证据

### 回答的审稿问题

> 你们声称同时解决 drift 与 collapse，但正则项只惩罚离 centroid 太远的状态；有什么直接证据证明它防止 collapse，而不是单纯让轨迹更紧、更收缩？

### 脚本功能

专注于 "是否发生 latent collapse" 的诊断。从 `run_*` 目录读取 `latents.json` / `trajectory_stats.json` / `metrics.json`，计算：

- **Effective rank**：latent matrix 奇异值分布熵 → `exp(H)`
- **RandomSim / anisotropy**：随机两 token 的平均 cosine similarity（越高越挤进窄锥）
- **Trajectory diversity**：同一样本内 latent token 的平均两两 L2 距离 和 cosine distance
- **Radius quantiles**：`r_t` 的 P50 / P90 / P99

### 运行方式

```bash
source /data/yhao/baseline/CODI/config.env
source "$CODI_VENV_PATH"

python /data/yhao/baseline/CODI/plots/analyze_latent_collapse.py \
  /data/yhao/baseline/CODI/results/latent_sweep_gsm8k/latent_16/models/simcon/gsm8k/run_0 \
  /data/yhao/baseline/CODI/results/latent_sweep_gsm8k/latent_16/models/simcon_sircl/gsm8k/run_0 \
  --labels simcon simcon_sircl \
  --output-dir /data/yhao/baseline/CODI/plots/results/reviewer_boundary_simcon_t16/latent_collapse \
  --max-samples 300 \
  --bootstrap-iters 200
```

### 核心产出文件

| 文件 | 说明 |
|------|------|
| `results/reviewer_boundary_simcon_t16/latent_collapse/latent_collapse_summary.csv` | 各指标汇总表 |
| `results/reviewer_boundary_simcon_t16/latent_collapse/latent_collapse_summary.json` | 同上 JSON 格式 |
| `results/reviewer_boundary_simcon_t16/latent_collapse/gsm8k_collapse_summary.png` | 可视化对比图 |

额外跑出的扩展结果（不同 backbone / 数据集）：

| 目录 | 说明 |
|------|------|
| `results/latent_collapse_simcon_t16_20260328/` | simcon 家族 T=16 独立运行 |
| `results/latent_collapse_codi_t16_20260328/` | codi 家族 T=16 独立运行 |
| `results/latent_collapse_gsmhard_fallback_20260328/` | GSM-Hard fallback 对照 |

### 关键结论

| 指标 | simcon | simcon_sircl |
|------|--------|-------------|
| effective rank | 6.46 | 5.23 |
| randomsim | 0.4928 | 0.5340 |
| diversity L2 | 14.20 | 8.18 |
| radius P90 | 16.92 | 9.32 |

+SIRCL 确实做了更强的几何收缩，但这种收缩伴随 +7.96 个百分点的准确率提升。更合理的表述是：baseline 的失败模式更像"无约束漂移"，SIRCL 施加的是"有益的结构化收缩"，而非有害的 collapse。

### 对应分析报告

- `results/LATENT_COLLAPSE_REPORT_20260328.md`

---

## 2. `analyze_centroid_reference.py` — centroid = 语义锚点的解释与证据

### 回答的审稿问题

> 为什么 centroid 应被解释为"核心问题上下文 / 语义锚点"，而不是轨迹几何平均？能否提供 probing 或 intervention 证据？

### 脚本功能

验证 centroid 是否是一个有意义的全局参考点。通过对比 own centroid / shuffled centroid / wrong centroid，测量：

- Token 到各类 centroid 的距离
- `z1` 与 centroid 的余弦相似度
- 长度鲁棒性
- 支持 `--auto-model-probe` 和 `--run-intervention`（本次未启用，离线版本）

### 运行方式

```bash
source /data/yhao/baseline/CODI/train_on_gsm8k_dataset/env.sh

python /data/yhao/baseline/CODI/plots/analyze_centroid_reference.py \
  --dataset gsm8k \
  --run simcon=/data/yhao/baseline/CODI/results/latent_sweep_gsm8k/latent_16/models/simcon/gsm8k/run_0 \
  --run simcon_sircl=/data/yhao/baseline/CODI/results/latent_sweep_gsm8k/latent_16/models/simcon_sircl/gsm8k/run_0 \
  --output-dir /data/yhao/baseline/CODI/plots/results/reviewer_boundary_simcon_t16/centroid_reference
```

### 核心产出文件

| 文件 | 说明 |
|------|------|
| `results/reviewer_boundary_simcon_t16/centroid_reference/CENTROID_REFERENCE_SUMMARY.md` | 文字总结 |
| `results/reviewer_boundary_simcon_t16/centroid_reference/probe_summary.csv` | probing 统计 |
| `results/reviewer_boundary_simcon_t16/centroid_reference/offline_intervention_summary.csv` | 干预对比 |
| `results/reviewer_boundary_simcon_t16/centroid_reference/centroid_probing_summary.png` | probing 可视化 |
| `results/reviewer_boundary_simcon_t16/centroid_reference/centroid_offline_intervention.png` | 干预可视化 |
| `results/reviewer_boundary_simcon_t16/centroid_reference/sample_level_metrics.csv` | 样本级数据 |

额外的独立正式跑：

| 目录 | 说明 |
|------|------|
| `results/centroid_reference_gsm8k_mature_20260328/` | GSM8K 完整版 centroid 分析（含 `INTERPRETATION_AND_SCRIPT_GUIDE.md`） |

### 关键结论

| 条件 | own centroid 距离 | shuffled centroid 距离 | wrong centroid 距离 | shuffled/own 倍率 |
|------|-------------------|----------------------|--------------------|--------------------|
| simcon | 9.98 | 21.67 | 21.88 | 2.17x |
| simcon_sircl | 5.59 | 14.91 | 15.59 | 2.67x |

Centroid 不是随便替换都一样的虚假参考点——own centroid 明显比错配 centroid 更接近真实 token 轨迹，且 +SIRCL 后区分度更大。但其有效性依赖 trajectory 本身的质量。

---

## 3. `analyze_sircl_failure_modes.py` — 轨迹已错→centroid 也错的适用边界

### 回答的审稿问题

> 如果 trajectory 本来就错，centroid 也会错；是否意味着 SIRCL 只在原始 implicit reasoning 已经比较好时才有效？

### 脚本功能

**最直接回答审稿问题的脚本**。输入 baseline 与 +SIRCL 的一对 `run_*` 目录，做样本级配对分析：

- 按 baseline 正确/错误分桶，统计 +SIRCL 的 transition（recovered / regressed / both correct / both wrong）
- 每组的几何统计（`r_t` 分布、token diversity、path length）
- 失败案例 / 恢复案例导出

### 运行方式

```bash
python /data/yhao/baseline/CODI/plots/analyze_sircl_failure_modes.py \
  --baseline-run /data/yhao/baseline/CODI/results/latent_sweep_gsm8k/latent_16/models/simcon/gsm8k/run_0 \
  --sircl-run /data/yhao/baseline/CODI/results/latent_sweep_gsm8k/latent_16/models/simcon_sircl/gsm8k/run_0 \
  --output-dir /data/yhao/baseline/CODI/plots/results/reviewer_boundary_simcon_t16/failure_modes \
  --max-samples-per-group 100
```

### 核心产出文件

| 文件 | 说明 |
|------|------|
| `results/reviewer_boundary_simcon_t16/failure_modes/summary.md` | 文字总结 |
| `results/reviewer_boundary_simcon_t16/failure_modes/summary.json` | 机器可读汇总 |
| `results/reviewer_boundary_simcon_t16/failure_modes/transition_summary.csv` | 四类 transition 统计 |
| `results/reviewer_boundary_simcon_t16/failure_modes/gain_by_baseline_bucket.csv` | 按 baseline 正确/错误分桶的收益 |
| `results/reviewer_boundary_simcon_t16/failure_modes/geometry_group_summary.csv` | 各组几何统计 |
| `results/reviewer_boundary_simcon_t16/failure_modes/failure_examples.csv` | 典型失败/恢复样例 |
| `results/reviewer_boundary_simcon_t16/failure_modes/radius_profiles.csv` | 半径分布 |
| `results/reviewer_boundary_simcon_t16/failure_modes/sample_level.csv` | 样本级完整数据 |
| `results/reviewer_boundary_simcon_t16/failure_modes/accuracy_gain_by_baseline_bucket.png` | 分桶收益图 |
| `results/reviewer_boundary_simcon_t16/failure_modes/baseline_rt_by_correctness.png` | baseline 正确/错误的 r_t 对比 |
| `results/reviewer_boundary_simcon_t16/failure_modes/baseline_wrong_rt_split.png` | baseline-wrong 内 recovered vs still-wrong |
| `results/reviewer_boundary_simcon_t16/failure_modes/baseline_group_diversity.png` | 各组多样性对比 |

### 关键结论

**Transition 统计**（simcon T=16, GSM8K）：

| Transition | 样本数 |
|-----------|--------|
| Both correct | 541 |
| Recovered by +SIRCL | 192 |
| Both wrong | 499 |
| Regressed with +SIRCL | 87 |

**净收益来自 baseline-wrong 样本**：baseline-wrong 分桶上 +SIRCL 带来 +27.79 个百分点恢复率。

**方法边界的几何刻画**：

| 组 | radius_mean | token diversity |
|----|-------------|-----------------|
| baseline-correct | 10.991 | 15.923 |
| baseline-wrong | 9.110 | 13.480 |
| → recovered | 9.825 | 14.448 |
| → still wrong | 8.659 | 12.771 |

结论：当 baseline 虽然错，但轨迹仍有一定几何展开时，SIRCL 更可能把它拉回去；当 baseline 已过早塌缩，centroid 会变得噪声化，帮助受限。

---

## 4. `plot_scaling_stability.py` — 训练时 scaling：补齐 no-SIRCL matched baseline

### 回答的审稿问题

> 为什么训练期 scaling（T=6/16/32）的实验没有 no-SIRCL baseline？如果"稳定 scaling"是 strongest claim，这个对照是必要的。

### 脚本功能

将 `latent_sweep_gsm8k` 里不同 T 的结果按 T / condition（no-SIRCL vs +SIRCL）/ family（Sim-CoT / CODI）聚合，同时报告：

- accuracy
- effective rank ratio
- randomsim
- adjacent cosine
- radius mean

支持 `--preset latent_sweep_simcon` 和 `--preset latent_sweep_codi` 两个预设。

### 运行方式

```bash
source /data/yhao/baseline/CODI/config.env
source "$CODI_VENV_PATH"

# Sim-CoT 家族
python /data/yhao/baseline/CODI/plots/plot_scaling_stability.py \
  --preset latent_sweep_simcon \
  --output-dir /data/yhao/baseline/CODI/plots/results/rebuttal_scaling_simcon_20260328

# CODI 家族
python /data/yhao/baseline/CODI/plots/plot_scaling_stability.py \
  --preset latent_sweep_codi \
  --output-dir /data/yhao/baseline/CODI/plots/results/rebuttal_scaling_codi_20260328
```

### 核心产出文件

**Sim-CoT 家族**：

| 文件 | 说明 |
|------|------|
| `results/rebuttal_scaling_simcon_20260328/sim-cot_summary.md` | 文字总结 |
| `results/rebuttal_scaling_simcon_20260328/delta_summary.csv` | +SIRCL 与 no-SIRCL 的 delta 表 |
| `results/rebuttal_scaling_simcon_20260328/grouped_summary.csv` | 按 T 分组的聚合数据 |
| `results/rebuttal_scaling_simcon_20260328/sim-cot_scaling_summary.png` | 可视化 |

**CODI 家族**：

| 文件 | 说明 |
|------|------|
| `results/rebuttal_scaling_codi_20260328/codi_summary.md` | 文字总结 |
| `results/rebuttal_scaling_codi_20260328/delta_summary.csv` | delta 表 |
| `results/rebuttal_scaling_codi_20260328/grouped_summary.csv` | 按 T 分组聚合 |
| `results/rebuttal_scaling_codi_20260328/codi_scaling_summary.png` | 可视化 |

### 关键结论

| 家族 | T>=10 平均提升 | 提升范围 |
|------|---------------|---------|
| Sim-CoT | +7.50 pp | +6.14 ~ +8.26 pp |
| CODI | +2.58 pp | +2.12 ~ +3.11 pp |

SIRCL 的收益在长链 latent token 设置下是稳定趋势，不是偶然点。Sim-CoT 上效果更强，CODI 上也存在但幅度更小。

### 对应分析报告

- `results/SCALING_STABILITY_ANALYSIS_20260328.md`

---

## 5. `analyze_trajectory_by_correctness.py` — 只分析 all-correct 子集与失败样本分析

### 回答的审稿问题

> 几何分析为何只看 all-correct？能否分析失败例或混合难度子集，证明 SIRCL 在"关键处"改变轨迹？

### 脚本功能

按 `correct / wrong / all_correct / all_wrong / mixed / sircl_flips` 将轨迹几何拆开看，输出：

- Per-step `r_t` 曲线（均值 ± bootstrap CI）
- PCA trajectory projection
- Token cosine similarity heatmap

### 运行方式

```bash
python /data/yhao/baseline/CODI/plots/analyze_trajectory_by_correctness.py \
  /data/yhao/baseline/CODI/results/latent_sweep_gsm8k/latent_16/models/simcon/gsm8k/run_0 \
  /data/yhao/baseline/CODI/results/latent_sweep_gsm8k/latent_16/models/simcon_sircl/gsm8k/run_0 \
  --labels simcon simcon_sircl \
  --output-dir /data/yhao/baseline/CODI/plots/results/reviewer_boundary_simcon_t16/trajectory_correctness \
  --max-samples 300 \
  --bootstrap 200 \
  --projection pca \
  --max-trajectories 60
```

### 核心产出文件

| 文件 | 说明 |
|------|------|
| `results/reviewer_boundary_simcon_t16/trajectory_correctness/trajectory_correctness_summary.csv` | 各组几何统计汇总 |
| `results/reviewer_boundary_simcon_t16/trajectory_correctness/trajectory_correctness_summary.json` | 同上 JSON |
| `results/reviewer_boundary_simcon_t16/trajectory_correctness/rt_curve_compare_simcon_vs_simcon_sircl.png` | r_t 曲线对比 |
| `results/reviewer_boundary_simcon_t16/trajectory_correctness/rt_curve_simcon.png` | simcon 按正确性分组的 r_t |
| `results/reviewer_boundary_simcon_t16/trajectory_correctness/rt_curve_simcon_sircl.png` | simcon_sircl 分组 r_t |
| `results/reviewer_boundary_simcon_t16/trajectory_correctness/trajectory_pca_groups_simcon.png` | PCA 轨迹分组 |
| `results/reviewer_boundary_simcon_t16/trajectory_correctness/trajectory_pca_groups_simcon_sircl.png` | PCA 轨迹分组（+SIRCL） |
| `results/reviewer_boundary_simcon_t16/trajectory_correctness/sim_heatmap_simcon.png` | cosine similarity heatmap |
| `results/reviewer_boundary_simcon_t16/trajectory_correctness/sim_heatmap_simcon_sircl.png` | heatmap（+SIRCL） |

### 关键结论

**Baseline simcon**（按正确性分组的 `r_t`）：

| 组 | r_t mean |
|----|----------|
| correct | 11.11 |
| wrong | 9.09 |
| all_wrong | 8.69 |
| sircl_flips | 9.97 |

正确样本保留更大几何展开；完全错误样本更容易早期收缩；sircl_flips 处于两者之间（"还有得救"的那一侧）。

**+SIRCL 后**：所有组都被压到 `r_t ≈ 5.3~5.7`，错误样本不再表现出特别大的几何差异。几何规范化不等于全部答对，但显著缓解了 baseline 的 "塌缩型失败"。

---

## 结果目录结构总览

```
CODI/plots/
├── analyze_latent_collapse.py          ← 脚本 1：collapse 诊断
├── analyze_centroid_reference.py       ← 脚本 2：centroid 有效性
├── analyze_sircl_failure_modes.py      ← 脚本 3：failure modes / 适用边界
├── plot_scaling_stability.py           ← 脚本 4：scaling stability
├── analyze_trajectory_by_correctness.py ← 脚本 5：按正确性分组分析
│
├── REBUTTAL_ANALYSIS_TOOLS.md          ← 本文档
├── REBUTTAL_TRAJECTORY_ANALYSIS_20260328.md  ← 综合分析报告
│
└── results/
    ├── LATENT_COLLAPSE_REPORT_20260328.md     ← collapse 分析报告
    ├── SCALING_STABILITY_ANALYSIS_20260328.md  ← scaling 分析报告
    ├── REBUTTAL_TRAJECTORY_ANALYSIS_20260328.md ← 综合报告副本
    │
    │  ── 核心结果（审稿问题 1-3-5 合并目录）──
    ├── reviewer_boundary_simcon_t16/
    │   ├── latent_collapse/            ← 脚本 1 产出
    │   ├── centroid_reference/         ← 脚本 2 产出
    │   ├── failure_modes/              ← 脚本 3 产出
    │   └── trajectory_correctness/     ← 脚本 5 产出
    │
    │  ── 核心结果（审稿问题 4）──
    ├── rebuttal_scaling_simcon_20260328/  ← 脚本 4：Sim-CoT 家族
    ├── rebuttal_scaling_codi_20260328/   ← 脚本 4：CODI 家族
    │
    │  ── 扩展跑（独立 backbone/数据集 对照）──
    ├── latent_collapse_simcon_t16_20260328/
    ├── latent_collapse_codi_t16_20260328/
    ├── latent_collapse_gsmhard_fallback_20260328/
    ├── centroid_reference_gsm8k_mature_20260328/
    ├── trajectory_correctness_full/
    │
    │  ── 中间产物 / 可清理 ──
    ├── _tmp_boundary_check/            ← 调试用中间产物
    ├── _trash/                         ← 早期 smoketest（可删除）
    ├── rebuttal_scaling_simcon/        ← 被 _20260328 版本取代
    └── rebuttal_scaling_codi/          ← 被 _20260328 版本取代
```

---

## 审稿问题 → 可直接引用的关键数字

写 rebuttal 时最值得引用的 4 个点：

1. **SIRCL 在长链 latent 下稳定有效**：Sim-CoT T>=10 平均提升 +7.50 pp（来自脚本 4）
2. **收益不只来自 baseline-correct**：baseline-wrong 分桶上 +27.79 pp 恢复率（来自脚本 3）
3. **方法边界可被几何统计刻画**：still-wrong 的 radius=8.659 < recovered 的 9.825（来自脚本 3）
4. **centroid 是有意义的 anchor**：own/shuffled 距离倍率 2.17x~2.67x（来自脚本 2）
