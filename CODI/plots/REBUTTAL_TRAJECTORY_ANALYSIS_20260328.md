# Rebuttal Trajectory Analysis Report

更新时间：2026-03-28

这份文档是本次 rebuttal 几何分析的可追踪版本。

对应的图表、CSV、JSON 产物位于：

- `CODI/plots/results/rebuttal_scaling_simcon/`
- `CODI/plots/results/rebuttal_scaling_codi/`
- `CODI/plots/results/reviewer_boundary_simcon_t16/`

## 1. 本次我实际运行了什么

这次分析分成两层：

1. 全局 scaling / stability 扫描
   - `CODI/plots/plot_scaling_stability.py`
   - 直接复用 `CODI/results/latent_sweep_gsm8k`
   - 分别跑了 `Sim-CoT` 家族和 `CODI` 家族

2. 样本级深度分析
   - 选择 `latent_16 / simcon vs simcon_sircl`
   - 这是更贴近审稿问题“长程 implicit reasoning + SIRCL 的适用边界”的设置
   - 运行了下面 4 支脚本：
     - `CODI/plots/analyze_sircl_failure_modes.py`
     - `CODI/plots/analyze_trajectory_by_correctness.py`
     - `CODI/plots/analyze_latent_collapse.py`
     - `CODI/plots/analyze_centroid_reference.py`

## 2. 主要结论

### 2.1 一句话版本

这批结果支持下面这个更精确的说法：

- `SIRCL` 不是只在 “baseline 已经答对” 时才有效。
- 在 `Sim-CoT, T=16` 上，净增益主要来自 `baseline 原本答错` 的样本。
- 但 `SIRCL` 也确实有明显边界：如果 baseline 轨迹已经过早塌缩、几何多样性太低，centroid 能提供的有效约束会变弱，此时很多样本仍然救不回来。

也就是说，更准确的边界不是：

- “baseline 一错，SIRCL 就没用”

而是：

- “baseline 若仍保留一定任务相关的几何结构，SIRCL 往往能把它拉回正确轨道；若轨迹已经早期塌缩或偏离过深，SIRCL 的帮助会明显受限”

### 2.2 最关键的数字

#### 全局 scaling 结论

- `Sim-CoT` 家族中，`+SIRCL` 在 `T>=10` 的平均准确率提升为 `+7.50` 个百分点。
- `Sim-CoT` 家族中，`T>=10` 的提升范围是 `+6.14` 到 `+8.26` 个百分点。
- `CODI` 家族中，`+SIRCL` 在 `T>=10` 的平均准确率提升为 `+2.58` 个百分点。
- `CODI` 家族中，`T>=10` 的提升范围是 `+2.12` 到 `+3.11` 个百分点。

这说明：

- 对长 latent 链条来说，`+SIRCL` 的优势不是偶然点，而是一个相对稳定的趋势。
- 这种趋势在 `Sim-CoT` 上更强，在 `CODI` 上也存在但幅度较小。

#### `T=16, simcon vs simcon_sircl` 的净效果

- baseline `simcon`：`47.61%`
- `+SIRCL simcon_sircl`：`55.57%`
- 绝对提升：`+7.96` 个百分点
- 相对提升：`+16.72%`

#### 最直接回答审稿问题的分桶结果

来自 `failure_modes/summary.md`：

- `baseline correct` 样本：`+SIRCL` 后准确率从 `100.00%` 变成 `86.15%`，变化 `-13.85` 个百分点
- `baseline wrong` 样本：`+SIRCL` 后准确率从 `0.00%` 变成 `27.79%`，变化 `+27.79` 个百分点

对应 transition 统计：

- `Both correct`：`541`
- `Recovered by +SIRCL`：`192`
- `Both wrong`：`499`
- `Regressed with +SIRCL`：`87`

这组数非常重要，因为它说明：

- 净收益并不是来自“把本来就对的题保持住”
- 净收益更大程度上来自“把一部分本来错的题救回来”
- 但 `SIRCL` 也确实会让一部分原本正确样本退化，这是需要在 rebuttal 里明确承认的方法边界

## 3. 各脚本做了什么，以及这次跑出来了什么

### 3.1 `plot_scaling_stability.py`

#### 这支脚本做什么

它把 `latent_sweep_gsm8k` 里不同 `T` 的结果拉平，按：

- `T`
- `condition`（`no-SIRCL` / `+SIRCL`）
- `family`（`Sim-CoT` / `CODI`）

聚合出：

- accuracy
- effective rank ratio
- randomsim
- adjacent cosine
- radius mean

#### 这次怎么跑

```bash
source /data/yhao/baseline/CODI/config.env
source "$CODI_VENV_PATH"

python /data/yhao/baseline/CODI/plots/plot_scaling_stability.py \
  --preset latent_sweep_simcon \
  --output-dir /data/yhao/baseline/CODI/plots/results/rebuttal_scaling_simcon

python /data/yhao/baseline/CODI/plots/plot_scaling_stability.py \
  --preset latent_sweep_codi \
  --output-dir /data/yhao/baseline/CODI/plots/results/rebuttal_scaling_codi
```

#### 关键输出

- `CODI/plots/results/rebuttal_scaling_simcon/sim-cot_summary.md`
- `CODI/plots/results/rebuttal_scaling_simcon/delta_summary.csv`
- `CODI/plots/results/rebuttal_scaling_codi/codi_summary.md`
- `CODI/plots/results/rebuttal_scaling_codi/delta_summary.csv`

#### 结果怎么解读

`Sim-CoT`：

- `T=10..18` 的准确率提升都在 `+6.14` 到 `+8.26` 个百分点之间
- 同时 `radius_mean` 持续下降，说明 `SIRCL` 在长链推理时显著压缩了轨迹半径
- `effective_rank/T` 普遍下降，说明轨迹被压到更低维、更紧的子空间
- 但 accuracy 却上升，这意味着：`SIRCL` 的收益不是“更发散”，而是“更受控的收缩”

`CODI`：

- 同样在所有长链 `T` 上稳定增益
- 但增益幅度显著小于 `Sim-CoT`
- 说明 `SIRCL` 在已经更强的 baseline 上主要是稳态修正，而不是大幅纠偏

一句话总结这支脚本的意义：

- 它证明 `SIRCL` 的收益在长链 latent token 设置下是稳定存在的，而不是个别点碰巧有效

### 3.2 `analyze_sircl_failure_modes.py`

#### 这支脚本做什么

它是最直接回答审稿问题的脚本。输入一对 `run_*` 目录，做样本级配对分析：

- 哪些样本 `wrong -> correct`
- 哪些样本 `correct -> wrong`
- baseline 正确/错误分桶后的 `+SIRCL` 平均收益
- baseline 轨迹的几何统计
- 失败案例 / 恢复案例导出

#### 这次怎么跑

```bash
python /data/yhao/baseline/CODI/plots/analyze_sircl_failure_modes.py \
  --baseline-run /data/yhao/baseline/CODI/results/latent_sweep_gsm8k/latent_16/models/simcon/gsm8k/run_0 \
  --sircl-run /data/yhao/baseline/CODI/results/latent_sweep_gsm8k/latent_16/models/simcon_sircl/gsm8k/run_0 \
  --output-dir /data/yhao/baseline/CODI/plots/results/reviewer_boundary_simcon_t16/failure_modes \
  --max-samples-per-group 100
```

#### 关键输出

- `CODI/plots/results/reviewer_boundary_simcon_t16/failure_modes/summary.md`
- `CODI/plots/results/reviewer_boundary_simcon_t16/failure_modes/gain_by_baseline_bucket.csv`
- `CODI/plots/results/reviewer_boundary_simcon_t16/failure_modes/transition_summary.csv`
- `CODI/plots/results/reviewer_boundary_simcon_t16/failure_modes/geometry_group_summary.csv`
- `CODI/plots/results/reviewer_boundary_simcon_t16/failure_modes/failure_examples.csv`

#### 结果怎么解读

这是本轮最重要的结论来源：

- `baseline wrong` 样本上，`+SIRCL` 直接带来 `+27.79` 个百分点的改善
- `baseline correct` 样本上，存在 `-13.85` 个百分点的退化
- `recovered = 192` 明显多于 `regressed = 87`

这说明：

- `SIRCL` 的收益不是只来自保住 easy / already-correct samples
- 它确实能纠正大量原本错误的长链轨迹

但几何统计也同时告诉我们边界在哪里：

- baseline-wrong 的 `radius_mean = 9.110`，低于 baseline-correct 的 `10.991`
- baseline-wrong 的 `token diversity = 13.480`，低于 baseline-correct 的 `15.923`

也就是：

- baseline 错样本并不是“更乱”，反而更像是“更早塌缩、更少探索”

进一步看 baseline-wrong 内部：

- `recovered` 的 `radius_mean = 9.825`
- `still wrong` 的 `radius_mean = 8.659`
- `recovered` 的 `token diversity = 14.448`
- `still wrong` 的 `token diversity = 12.771`

这组数正好对应审稿问题的核心：

- 当 baseline 虽然错，但轨迹仍有一定几何展开和任务相关结构时，`SIRCL` 更可能把它拉回去
- 当 baseline 已经过早塌缩时，centroid 也会变得不够有信息量，`SIRCL` 的帮助会明显减弱

#### 失败 / 恢复样例也支持这个解释

`failure_examples.csv` 里很典型：

- 恢复样例 `sample 220`：baseline `80 -> gold 70`，`+SIRCL` 改成 `70`
  - `radius_mean: 14.75 -> 5.64`
  - `path_length: 326.78 -> 81.93`

- 恢复样例 `sample 274`：baseline `4800 -> gold 3200`，`+SIRCL` 改成 `3200`
  - `radius_mean: 14.61 -> 6.13`
  - `path_length: 326.12 -> 90.84`

这些样本说明：

- baseline 的错不是“没动”，而是“动得太远、太长、太散”
- `SIRCL` 把这种漂移型错误压回来了

而 regression 样本很多是另一种情况：

- 它们 baseline 本来已经比较紧凑，`SIRCL` 再压一次反而可能把答案推偏

### 3.3 `analyze_trajectory_by_correctness.py`

#### 这支脚本做什么

它按 `correct / wrong / all_correct / all_wrong / mixed / sircl_flips`
把轨迹几何拆开看，并输出：

- per-step `r_t` 曲线
- PCA trajectory projection
- token cosine similarity heatmap

#### 这次怎么跑

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

#### 关键输出

- `CODI/plots/results/reviewer_boundary_simcon_t16/trajectory_correctness/trajectory_correctness_summary.csv`
- `CODI/plots/results/reviewer_boundary_simcon_t16/trajectory_correctness/rt_curve_compare_simcon_vs_simcon_sircl.png`
- `CODI/plots/results/reviewer_boundary_simcon_t16/trajectory_correctness/trajectory_pca_groups_simcon*.png`

#### 结果怎么解读

baseline `simcon`：

- correct: `r_t = 11.11`
- wrong: `r_t = 9.09`
- all_wrong: `r_t = 8.69`
- sircl_flips: `r_t = 9.97`

这说明 baseline 里：

- 正确样本通常保留更大的几何展开
- 完全错误样本更容易早期收缩
- `sircl_flips` 处在两者之间，更接近“还有得救”的那一侧

`simcon_sircl`：

- correct: `r_t = 5.46`
- wrong: `r_t = 5.66`
- all_correct: `r_t = 5.28`
- all_wrong: `r_t = 5.74`

这说明 `+SIRCL` 之后：

- 各类样本都被明显压到更紧的轨迹管道里
- 错误样本不再像 baseline 那样表现出特别大的几何差异
- 但“几何被规范化”不等于“全部答对”，仍有一部分问题需要更强的语义推理能力

所以这支脚本给出的信息是：

- baseline 的失败模式更像“塌缩 / 失去有效展开”
- `SIRCL` 通过几何约束把这部分问题显著缓解了
- 但不是所有几何规范化都会自动转化成正确答案

### 3.4 `analyze_latent_collapse.py`

#### 这支脚本做什么

它更专注于“是否发生 latent collapse”，会计算：

- effective rank
- randomsim / anisotropy
- trajectory diversity
- radius quantiles

#### 这次怎么跑

```bash
python /data/yhao/baseline/CODI/plots/analyze_latent_collapse.py \
  /data/yhao/baseline/CODI/results/latent_sweep_gsm8k/latent_16/models/simcon/gsm8k/run_0 \
  /data/yhao/baseline/CODI/results/latent_sweep_gsm8k/latent_16/models/simcon_sircl/gsm8k/run_0 \
  --labels simcon simcon_sircl \
  --output-dir /data/yhao/baseline/CODI/plots/results/reviewer_boundary_simcon_t16/latent_collapse \
  --max-samples 300 \
  --bootstrap-iters 200
```

#### 关键输出

- `CODI/plots/results/reviewer_boundary_simcon_t16/latent_collapse/latent_collapse_summary.csv`
- `CODI/plots/results/reviewer_boundary_simcon_t16/latent_collapse/gsm8k_collapse_summary.png`

#### 结果怎么解读

`simcon`：

- effective rank: `6.46`
- randomsim: `0.4928`
- diversity L2: `14.20`
- radius p90: `16.92`

`simcon_sircl`：

- effective rank: `5.23`
- randomsim: `0.5340`
- diversity L2: `8.18`
- radius p90: `9.32`

所以：

- `+SIRCL` 不是在“避免一切收缩”
- 相反，它明显在做更强的几何收缩和轨迹压缩

但重点是：

- 这种收缩并没有把模型推向更差结果，反而带来了 `+7.96` 个百分点的准确率提升

因此这里更合理的表述不是“collapse 一定坏”，而是：

- 对这个任务设定来说，baseline 的问题更像是“无约束漂移”
- `SIRCL` 施加的收缩是“有益的结构化收缩”
- 真正的 failure mode 不是“只要收缩就失败”，而是“过早、过度、且无任务相关性的收缩会让 centroid 失去信息量”

### 3.5 `analyze_centroid_reference.py`

#### 这支脚本做什么

它回答的是：

- centroid 到底是不是一个有意义的全局参考点？
- 还是只是一个随便取的均值向量？

它会比较：

- `own centroid`
- `shuffled centroid`
- `wrong centroid`

并看：

- token 到 centroid 的距离
- `z1` 与 centroid 的余弦相似度
- 长度鲁棒性

#### 这次怎么跑

```bash
source /data/yhao/baseline/CODI/train_on_gsm8k_dataset/env.sh

python /data/yhao/baseline/CODI/plots/analyze_centroid_reference.py \
  --dataset gsm8k \
  --run simcon=/data/yhao/baseline/CODI/results/latent_sweep_gsm8k/latent_16/models/simcon/gsm8k/run_0 \
  --run simcon_sircl=/data/yhao/baseline/CODI/results/latent_sweep_gsm8k/latent_16/models/simcon_sircl/gsm8k/run_0 \
  --output-dir /data/yhao/baseline/CODI/plots/results/reviewer_boundary_simcon_t16/centroid_reference
```

这次我跑的是离线版本，没有启用：

- `--auto-model-probe`
- `--run-intervention`

#### 关键输出

- `CODI/plots/results/reviewer_boundary_simcon_t16/centroid_reference/CENTROID_REFERENCE_SUMMARY.md`
- `CODI/plots/results/reviewer_boundary_simcon_t16/centroid_reference/probe_summary.csv`
- `CODI/plots/results/reviewer_boundary_simcon_t16/centroid_reference/offline_intervention_summary.csv`

#### 结果怎么解读

`probe_summary.csv`：

- `simcon`: `radius_mean_mean = 9.98`, `cos(mu, z1) = 0.5868`
- `simcon_sircl`: `radius_mean_mean = 5.59`, `cos(mu, z1) = 0.4691`

`offline_intervention_summary.csv`：

- `simcon` own centroid 距离 `9.98`
- `simcon` shuffled / wrong centroid 距离分别是 `21.67 / 21.88`
- `simcon_sircl` own centroid 距离 `5.59`
- `simcon_sircl` shuffled / wrong centroid 距离分别是 `14.91 / 15.59`

换成更直观的倍率：

- `simcon`: shuffled/own = `2.17x`, wrong/own = `2.19x`
- `simcon_sircl`: shuffled/own = `2.67x`, wrong/own = `2.79x`

这说明：

- centroid 不是一个随便替换都一样的“虚假参考点”
- own centroid 明显比错配 centroid 更接近真实 token 轨迹
- 而且在 `+SIRCL` 后，这种 own-vs-wrong 的区分更明显

也就是说：

- centroid 作为 trajectory-level reference 是有意义的
- 但它的可用性依赖于 trajectory 本身有没有被组织好

这恰好和上面的 failure-mode 分析一致：

- 当 trajectory 还保有一定结构时，centroid 是有信息量的 anchor
- 当 trajectory 已经早期崩坏时，centroid 当然会变差，这就是方法边界

## 4. 直接对应审稿问题的回答建议

如果把上面的结果压成更适合 rebuttal 的中文表述，我建议写成下面这种口径：

> 我们同意审稿人的判断：SIRCL 的有效性依赖于 baseline trajectory 仍保有一定任务相关结构。我们的新分析显示，这个边界并不等价于“只有 baseline 已经答对时才有效”。以 Sim-CoT、T=16 为例，SIRCL 的净收益主要来自 baseline 原本错误的样本：在 baseline-wrong 分桶中，+SIRCL 使 27.79% 的样本从 wrong 变为 correct，而同时仅有 6.60% 的全体样本出现 correct-to-wrong regression。进一步的几何统计表明，baseline-wrong 样本整体比 baseline-correct 样本表现出更低的 radius 和 token diversity，说明很多失败轨迹更接近于早期塌缩而非有益探索；而在 baseline-wrong 内部，被 SIRCL 成功修复的样本又显著比仍然失败的样本具有更大的 radius 和 diversity。这表明：当 baseline 轨迹仍保留一定几何展开和任务相关性时，centroid 仍可作为有效 anchor，SIRCL 能抑制后续漂移并带来恢复；但当 trajectory 过早塌缩或偏离过深时，centroid 会变得噪声化，SIRCL 的帮助将明显受限。我们会将这一点明确写为方法边界与失败模式。

## 5. 本批结果最值得写进 rebuttal 的 4 点

1. `SIRCL` 在长链 latent 设置下是稳定有效的，不是偶然点。
   - `Sim-CoT, T>=10` 平均提升 `+7.50` 个百分点。

2. `SIRCL` 的收益并不只来自“baseline 已经对”的样本。
   - 在 `T=16` 的 `simcon` 配对里，`baseline-wrong` 分桶上有 `+27.79` 个百分点的恢复率。

3. 方法边界是真实存在的，而且可以被几何统计刻画。
   - `still-wrong` 样本比 `recovered` 样本更塌缩、更低 diversity。

4. centroid 作为 anchor 是有意义的，但它依赖 trajectory 质量。

