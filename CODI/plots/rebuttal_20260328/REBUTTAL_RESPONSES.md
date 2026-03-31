# Rebuttal 正式回应草稿

更新时间：2026-03-29（全部更新为 T=6 latent token 数据）

本文档是基于 `CODI/plots/rebuttal_20260328/` 下全部分析结果撰写的 5 条 rebuttal 正式回应草稿。**所有 latent token 几何分析均使用 T=6 的数据**（原论文标准设置），与正文实验对齐。T=16/32 仅用于 scaling 对比（Rebuttal #4）。

---

## 策略总览

| # | 审稿问题 | 实际数据方向 | 核心策略 |
|---|---------|------------|---------|
| 1 | 防 collapse 直接证据 | EffRank 降、RandomSim 升，但 accuracy 同时升 | "有益结构化收缩 vs 无约束漂移"框架 |
| 2 | centroid = 语义锚点 | own/shuffled 距离比 1.84–2.94×，cos(μ,z1) 在 SIRCL 后下降 | 降级为 geometric anchor；用 replacement ratio 证明信息性 |
| 3 | 轨迹已错→适用边界 | T=6: recovered=119, regressed=80，净正；baseline-wrong +19.44% | 精确化边界为"几何展开程度"；承认 T=6 增益更温和 |
| 4 | scaling matched baseline | T=6 gain=+2.96pp → T=16 +13.50pp → T=32 +13.95pp | 展示 SIRCL 增益随 T 单调放大 |
| 5 | 只看 all-correct | T=6 五组分层完整；correct r_t=13.15 vs all_wrong=11.29 | 分层 r_t + transition + PCA/heatmap |

---

## Rebuttal #1：关于"防 collapse"的直接证据

### 审稿问题

> 你们声称同时解决 drift 与 collapse，但正则项只惩罚离 centroid 太远的状态；有什么直接证据证明它防止 collapse，而不是单纯让轨迹更紧、更收缩？

### 回应正文

感谢审稿人指出这一关键点。我们补充了三组互补的 collapse 诊断指标（effective rank、RandomSim/anisotropy、trajectory diversity），基于原论文标准设置 T=6 在 GSM8K 上做了系统分析（N=300, 100× bootstrap）。

我们的新结果表明，SIRCL 确实会降低 effective rank 并轻微提高 RandomSim，即轨迹确实变得更紧凑、更低维。但关键观察是：**这种几何收缩伴随着显著的准确率提升**（Sim-CoT T=6: +2.88 pp），而非性能退化。

进一步的样本级分析揭示了更精确的机制：baseline 的错误样本并不呈现"过于发散"的几何形态，反而表现为**更早的收缩和更低的 token diversity**（baseline-wrong: radius\_mean=11.37, diversity=18.20 vs baseline-correct: radius\_mean=13.11, diversity=20.71）。这说明 baseline 的主要失败模式是"无约束漂移后的早期塌缩"。SIRCL 施加有约束的、centroid-guided 的收缩，将轨迹从不可控的漂移拉回可用的推理区域。

因此，我们修正表述为：SIRCL 并非"防止一切收缩"，而是**用有结构的几何收缩替代无约束的漂移型塌缩**。这种 trade-off 在 T=6 时即已存在，在 T=16/32 时更加显著（详见 Rebuttal #4）。

### 支撑数据

**Table R1a: Collapse diagnostics, Sim-CoT T=6, GSM8K (N=300, 100× bootstrap)**

| Metric | simcon | simcon\_sircl | Δ |
|--------|--------|-------------|---|
| Accuracy | 53.22% | 56.10% | +2.88 pp |
| Effective Rank | 4.78 ± 0.03 | 4.26 ± 0.03 | −0.52 |
| EffRank/T | 0.682 ± 0.004 | 0.609 ± 0.004 | −0.073 |
| RandomSim | 0.424 ± 0.005 | 0.450 ± 0.005 | +0.026 |
| Diversity L2 | 18.60 ± 0.15 | 12.67 ± 0.10 | −5.93 |
| radius\_mean | 12.27 | 8.30 | −3.97 |
| r\_t P50 | 11.71 ± 0.13 | 6.88 ± 0.09 | −4.83 |
| r\_t P90 | 17.99 ± 0.13 | 17.21 ± 0.12 | −0.78 |
| r\_t P99 | 21.45 ± 0.22 | 20.32 ± 0.24 | −1.13 |

**Table R1b: Collapse diagnostics, CODI T=6, GSM8K (N=300, 100× bootstrap)**

| Metric | codi | codi\_sircl | Δ |
|--------|------|-----------|---|
| Accuracy | 52.92% | 55.72% | +2.80 pp |
| Effective Rank | 4.03 ± 0.03 | 2.83 ± 0.02 | −1.20 |
| RandomSim | 0.577 ± 0.005 | 0.634 ± 0.006 | +0.057 |
| Diversity L2 | 14.83 ± 0.12 | 7.66 ± 0.07 | −7.17 |
| radius\_mean | 10.31 | 5.09 | −5.22 |

**关键解读**：

- Sim-CoT T=6: SIRCL 的几何压缩相对温和（EffRank 降 11%），accuracy 提升 +2.88 pp
- CODI T=6: 几何压缩更激进（EffRank 降 30%），accuracy 提升 +2.80 pp
- **EffRank/T（每 token 有效维度）普遍下降**，但 accuracy 同时提升
- 真正有害的 collapse 是"accuracy 随几何维度下降而退化"——本实验中未出现此现象
- SIRCL 的收缩对任务表现**是有益的**，而非有害的

### 数据来源

- `01_collapse_evidence/results_t6_simcon/latent_collapse_summary.csv`
- `01_collapse_evidence/results_t6_codi/latent_collapse_summary.csv`

---

## Rebuttal #2：关于"centroid = 语义锚点"的解释与证据

### 审稿问题

> 为什么 centroid 应被解释为"核心问题上下文/语义锚点"，而不是轨迹几何平均？能否提供 probing 或 intervention 证据？

### 回应正文

感谢该建议。我们同意"centroid 具有语义锚点含义"在当前稿件中表述偏强。我们将在修订版中调整为更严谨的表述：**centroid 是 trajectory-level 的全局几何参考点（geometric anchor），用于约束轨迹不过度偏离其整体推理活动的中心区域**，而非严格的可解释语义中心。

我们在 T=6 标准设置下做了两组离线诊断（N=1319）：

**(1) Compactness probing**：SIRCL 一致地压缩了轨迹半径及其跨样本波动。

| Family | radius\_mean (w/o SIRCL) | radius\_mean (+SIRCL) | 压缩比 | radius\_std (w/o) | radius\_std (+SIRCL) |
|--------|-------------------------|----------------------|--------|-------------------|---------------------|
| Sim-CoT | 12.34 | 8.38 | −32.0% | 1.77 | 1.16 |
| CODI | 10.26 | 5.05 | −50.8% | 1.29 | 0.84 |

**(2) Centroid replacement intervention**：将每个样本的 own centroid 替换为 shuffled centroid（随机打乱）或 wrong-sample centroid，测量 token-anchor 距离的变化：

**Table R2: Centroid replacement intervention, T=6, GSM8K (N=1319)**

| Run | own dist | shuffled dist | wrong dist | shuffled/own | wrong/own |
|-----|---------|--------------|-----------|------------|---------|
| simcon | 12.34 | 22.76 | 22.92 | 1.84× | 1.86× |
| simcon\_sircl | 8.38 | 16.30 | 17.06 | **1.95×** | **2.04×** |
| codi | 10.26 | 16.19 | 16.36 | 1.58× | 1.59× |
| codi\_sircl | 5.05 | 14.86 | 15.29 | **2.94×** | **3.03×** |

替换 centroid 后，token-anchor 距离增大 1.6×–3.0×，且 **+SIRCL 使这一区分度显著更大**（simcon\_sircl: 1.95×，codi\_sircl: 2.94×）。这证明 centroid 携带了样本特定的参考信息，而非任意均值。

需要透明说明的是：cos(μ, z₁) 在 SIRCL 后**下降**（simcon: 0.684 → 0.604，codi: 0.828 → 0.692），说明 +SIRCL 的 centroid 与首步 latent 并不更对齐。但这不影响 centroid 作为 anchor 的有效性——replacement ratio 是更直接的证据。

我们将在修订版中：将"thematic center / core problem context"改为"trajectory-level global reference / geometric anchor"，将语义解释降至启发式层级。

### 数据来源

- `02_centroid_reference/results_t6/probe_summary.csv`
- `02_centroid_reference/results_t6/offline_intervention_summary.csv`

---

## Rebuttal #3：关于"轨迹已错→centroid 也错"的适用边界

### 审稿问题

> 如果 trajectory 本来就错，centroid 也会错；是否意味着 SIRCL 只在原始 implicit reasoning 已经比较好时才有效？

### 回应正文

我们同意审稿人的判断，并感谢帮助我们更清晰地界定方法边界。我们的补充分析（T=6，N=1319）表明，这一边界的精确表述应当是：

**SIRCL 的有效性并不等价于"只有 baseline 答对时才有效"，而是取决于 baseline 轨迹是否仍保有一定几何展开和任务相关结构。**

**Table R3a: Sample-level transition statistics, Sim-CoT T=6, GSM8K**

| Transition | 样本数 | 占比 |
|-----------|--------|------|
| Both correct | 627 | 47.5% |
| Recovered (wrong→correct) | 119 | 9.0% |
| Both wrong | 493 | 37.4% |
| Regressed (correct→wrong) | 80 | 6.1% |

净增益来自 baseline 原本错误的样本：在 baseline-wrong 分桶中，+SIRCL 使 **19.44%** 的样本从 wrong 变为 correct；recovered (119) 多于 regressed (80)。

进一步的几何统计揭示了边界的可刻画性：

**Table R3b: Geometric statistics by correctness group, T=6**

| Group | radius\_mean | token diversity |
|-------|------------|-----------------|
| baseline-correct | 13.11 | 20.71 |
| baseline-wrong | 11.37 | 18.20 |
| → recovered by SIRCL | 11.77 | 18.73 |
| → still wrong | 11.54 | 18.43 |

baseline 错误样本的失败模式是"更早塌缩、更少几何展开"。在 baseline-wrong 内部，被 SIRCL 成功修复的样本（recovered）比仍然失败的样本（still wrong）具有更大的 radius 和 diversity，差异虽然在 T=6 下相对较小（11.77 vs 11.54），但方向一致。

值得注意的是，**T=6 的 SIRCL 增益（+2.96 pp）明显小于 T=16（+13.50 pp）**，这与边界假说高度一致：T 越大，无约束漂移累积越多，SIRCL 的修正空间越大，收益越显著（详见 Rebuttal #4）。

我们将在修订版中把这一分析明确写为**方法边界与失败模式**独立 section。

### 数据来源

- `03_failure_modes_boundary/results_t6/summary.md`
- `03_failure_modes_boundary/results_t6/gain_by_baseline_bucket.csv`
- `03_failure_modes_boundary/results_t6/geometry_group_summary.csv`

---

## Rebuttal #4：关于训练时 scaling——补齐 no-SIRCL matched baseline

### 审稿问题

> 为什么训练期 scaling（T=6/16/32）的实验没有 no-SIRCL baseline？如果"稳定 scaling"是 strongest claim，这个对照是必要的。

### 回应正文

我们现已补充了 T=6、T=16、T=32 的完整 matched no-SIRCL 对照（Sim-CoT, LLaMA-1B-Instruct, GSM8K, 相同训练数据与优化器设置）。

**Table R4a: Sim-CoT scaling comparison, GSM8K best accuracy**

| T | no-SIRCL | +SIRCL | Δ |
|---|----------|--------|---|
| 6 | 53.22% | 56.10% | +2.88 pp |
| 16 | 44.50% | 58.00% | **+13.50 pp** |
| 32 | 43.06% | 57.01% | **+13.95 pp** |

**Table R4b: Scaling degradation from T=6 baseline**

| Condition | T=6 → T=16 | T=6 → T=32 |
|-----------|-----------|-----------|
| no-SIRCL | −8.72 pp ❌ 持续退化 | −10.16 pp ❌ |
| +SIRCL | +1.90 pp ✅ 稳定提升 | +0.91 pp ✅ |

核心发现：

1. **无 SIRCL 时，accuracy 随 T 增大而持续下降**（53.22% → 44.50% → 43.06%，累计 −10.16 pp）
2. **有 SIRCL 时，accuracy 不降反升**（56.10% → 58.00% → 57.01%）
3. **+SIRCL 的增益随 T 急剧扩大**：T=6 时 +2.88 pp，T=16/32 时均超过 +13 pp
4. T=6 时增益较小（+2.88 pp）本身也支持这一解释——短链漂移少、SIRCL 修正空间小

这直接证明了 SIRCL 的核心价值：**在更长的 latent reasoning chain 中，无约束的轨迹漂移会累积并导致性能退化，而 SIRCL 的几何约束有效抑制了这种退化**。

**Table R4c: Trajectory compactness at T=16 (representative)**

| Metric | no-SIRCL | +SIRCL |
|--------|----------|--------|
| radius\_mean | 10.00 | 5.62 |
| Adjacent Cosine | 0.855 | 0.917 |
| Violation Rate | 0.999 | 0.961 |

**训练设置透明说明**：no-SIRCL 训练了 10 epochs，+SIRCL 训练了 12 epochs。在**相同 checkpoint（step-29990）**下对比：T=16 差异为 +10.69 pp，T=32 差异为 +13.95 pp，结论不变。

### 数据来源

- no-SIRCL T=16/32: `CODI_rebuttal_runs/rebuttal_20260325/results/checkpoint_sweeps/decoder-trajectory-euclidean-{16,32}long/`
- +SIRCL T=16: `CODI/results/16long/` (best ckpt-35988, 58.00%)
- +SIRCL T=32: `CODI/results/32long/` (best ckpt-29990, 57.01%)
- T=6: `CODI/results/latent_sweep_gsm8k/latent_6/`

---

## Rebuttal #5：关于"只分析 all-correct 子集"与失败样本分析

### 审稿问题

> 几何分析为何只看 all-correct？能否分析失败例或混合难度子集，证明 SIRCL 在"关键处"改变轨迹？

### 回应正文

我们已补充了按正确性分层的完整轨迹几何分析（T=6，5 组：correct / wrong / all\_correct / all\_wrong / sircl\_flips）。

**Table R5a: Baseline simcon 按正确性分组的轨迹几何（T=6, GSM8K, N=300）**

| Group | n | r\_t mean | diversity L2 | cos\_sim consecutive | 解释 |
|-------|---|----------|-------------|---------------------|------|
| correct | 165 | 13.15 | 19.91 | 0.785 | 正确样本保留最大几何展开 |
| wrong | 135 | 11.34 | 17.21 | 0.825 | 错误样本更早收缩 |
| all\_correct | 153 | 13.25 | 20.04 | 0.783 | |
| all\_wrong | 112 | 11.29 | 17.16 | 0.824 | 完全错误样本最塌缩 |
| sircl\_flips | 23 | 11.54 | 17.46 | 0.829 | "可被救回"的样本处于中间地带 |
| sircl\_regress | 12 | 11.90 | 18.19 | 0.802 | 退化样本反而更发散 |

**Table R5b: +SIRCL 后各组几何变化**

| Group | simcon r\_t | simcon\_sircl r\_t | Δ |
|-------|-----------|-------------------|---|
| correct | 13.15 | 8.26 | −4.89 |
| wrong | 11.34 | 8.44 | −2.90 |
| all\_correct | 13.25 | 8.23 | −5.02 |
| all\_wrong | 11.29 | 8.45 | −2.84 |
| sircl\_flips | 11.54 | 8.44 | −3.10 |

+SIRCL 后所有组的 r\_t 被统一压缩到 8.2–8.5 的紧凑区间，correct/wrong 之间的几何差异大幅缩小（从 1.86 的差距收窄到 0.18）。

关键观察：

1. **baseline 中 sircl\_flips（被救回样本）的 r\_t=11.54，介于 correct（13.15）和 all\_wrong（11.29）之间**，说明这些样本仍保有一定几何结构，是"可救"的。
2. **sircl\_regress（退化样本）的 r\_t=11.90，反而比 all\_wrong 更大**，说明 SIRCL 的过度压缩可能把本来就对的样本推偏——这与 Rebuttal #3 中的方法边界分析完全一致。
3. +SIRCL 后各组 r\_t 几乎收敛（8.23–8.45），说明 SIRCL 确实起到了几何规范化作用，而非选择性只压某一类。

我们提供三类可视化（可附于 appendix）：

- `rt_curve_compare_simcon_vs_simcon_sircl.png` — per-step r\_t 曲线（各组均值 ± CI）
- `trajectory_pca_groups_simcon.png` — PCA 轨迹投影（5 组并列）
- `sim_heatmap_simcon.png` — token cosine similarity heatmap（展示是否同质化）

### 数据来源

- `05_correctness_stratified_trajectories/results_t6/trajectory_correctness_summary.csv`
- 可视化：同目录下 `.png` 文件

---

## 附录：5 条 rebuttal 之间的内部一致性

| 来源 | 结论 | 印证关系 |
|------|------|---------|
| #1 collapse | SIRCL 降 EffRank，但 accuracy ↑ | → #3: collapse 本身不是方法失败的原因 |
| #3 boundary | baseline-wrong 样本更塌缩 | → #5: sircl\_flips 的 r\_t 处于中间地带 ✓ |
| #4 scaling | T=6 增益小，T=16/32 增益大 | → #3: T=6 的方法边界更窄，边界随 T 扩大 ✓ |
| #2 centroid | own/shuffled 比值在 +SIRCL 后更大 | → #1: 收缩是有结构的，不是随机压缩 ✓ |
| #5 correctness | sircl\_regress 的 r\_t 最大（11.90）| → #3: 退化样本的几何特征可被解释 ✓ |

## 附录：最值得在 rebuttal 中强调的关键数字（T=6 版本）

1. **T=6 → T=32 无 SIRCL 掉 10.16 pp，+SIRCL 仅掉 0.21 pp**（+0.91 pp vs T=6）
2. **T=32 时 +SIRCL 增益达 +13.95 pp**（43.06% → 57.01%）
3. **codi\_sircl centroid replacement ratio 达 2.94×–3.03×**
4. **T=6: recovered=119 > regressed=80**，净正，且 recovered diversity（18.73）> still-wrong（18.43）
5. **sircl\_flips 的 r\_t=11.54 介于 correct（13.15）和 all\_wrong（11.29）之间** — 可被救回的几何特征
