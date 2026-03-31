# Rebuttal Analysis Workspace — `rebuttal_20260328/`

更新时间：2026-03-29

本目录是 2026-03-28 rebuttal 阶段的分析工作区，包含 5 个分析工具及其对应的结果目录。每个工具精确对应一个审稿问题。如果你要接手后续 rebuttal 工作，从这里开始。

最终回应草稿（含所有数字和表格）：[REBUTTAL_RESPONSES.md](REBUTTAL_RESPONSES.md)

当前多 backbone 训练/评测总表（`cot-sft` / `simcot` / `simcot+sircl` / `codi` / `codi+sircl` × `llama3-3b` / `qwen3-4b`）：[CURRENT_MULTIMODEL_RESULTS_20260329.md](CURRENT_MULTIMODEL_RESULTS_20260329.md)

---

## 目录结构一览

```
rebuttal_20260328/
├── README.md                       ← 你在看的这个文件
├── REBUTTAL_RESPONSES.md           ← 5 条正式 rebuttal 回应草稿（T=6 数据）
│
├── 01_collapse_evidence/           ← 防 collapse 直接证据
├── 02_centroid_reference/          ← centroid 是否是有意义的 anchor
├── 03_failure_modes_boundary/      ← 轨迹已错时 SIRCL 的适用边界
├── 04_scaling_matched_baseline/    ← T=6/16/32 scaling 对照
└── 05_correctness_stratified_trajectories/  ← 按正确性分层的轨迹分析
```

脚本源码统一放在：`CODI/plots/rebuttle_dir/`（注意拼写：rebuttle）

---

## 快速上手

### 环境准备

```bash
cd /data/yhao/baseline
source .venv/bin/activate
source CODI/config.env
```

### 重新运行任意分析

以下命令均在 `/data/yhao/baseline` 下执行。各脚本读取 `CODI/results/latent_sweep_gsm8k/` 下已有的 inference 产物（`latents.json`、`trajectory_stats.json`、`metrics.json`），**无需重新训练**。

---

## 5 个工具详解

---

### 工具 1：`analyze_latent_collapse.py`

**对应审稿问题**：SIRCL 有没有防止 collapse，还是只是让轨迹更收缩？

**做什么**：计算 latent token 的 collapse 诊断指标（effective rank、RandomSim/anisotropy、trajectory diversity、r_t 分位数），对多个 run 做 bootstrap 统计对比。

**输入**：`run_*/latents.json` + `trajectory_stats.json` + `metrics.json`

**运行命令**：

```bash
# Sim-CoT T=6（论文标准设置，推荐）
python CODI/plots/rebuttle_dir/analyze_latent_collapse.py \
  CODI/results/latent_sweep_gsm8k/latent_6/models/simcon/gsm8k/run_0 \
  CODI/results/latent_sweep_gsm8k/latent_6/models/simcon_sircl/gsm8k/run_0 \
  --labels simcon simcon_sircl \
  --output-dir CODI/plots/rebuttal_20260328/01_collapse_evidence/results_t6_simcon \
  --max-samples 300 \
  --bootstrap-iters 100

# CODI T=6
python CODI/plots/rebuttle_dir/analyze_latent_collapse.py \
  CODI/results/latent_sweep_gsm8k/latent_6/models/codi/gsm8k/run_0 \
  CODI/results/latent_sweep_gsm8k/latent_6/models/codi_sircl/gsm8k/run_0 \
  --labels codi codi_sircl \
  --output-dir CODI/plots/rebuttal_20260328/01_collapse_evidence/results_t6_codi \
  --max-samples 300 \
  --bootstrap-iters 100
```

**主要输出**：
- `latent_collapse_summary.csv` — 每个 run 的全部 collapse 指标（含 SE 和 CI95）
- `latent_collapse_summary.json` — 同上，JSON 格式
- `gsm8k_collapse_summary.png` — 对比图

**已有结果**：`01_collapse_evidence/results_t6_simcon/`，`results_t6_codi/`

**关键结论**（T=6）：
- SIRCL 使 Sim-CoT EffRank 从 4.78 降至 4.26，RandomSim 从 0.424 升至 0.450
- 但 accuracy 同时从 53.22% 升至 56.10%
- 解释框架：有益的结构化收缩，而非有害的维度塌缩

---

### 工具 2：`analyze_centroid_reference.py`

**对应审稿问题**：centroid 为什么应该被解释为有意义的参考点，而不是随便取的均值？

**做什么**：比较 own centroid vs shuffled centroid vs wrong-sample centroid 的距离倍率，评估 centroid 的样本特异性。同时计算 cos(μ, z₁) 的分布和半径标准差，评估 centroid 的稳定性。

**输入**：`run_*/latents.json` + `trajectory_stats.json` + `metrics.json`

**注意**：需要设置 `PYTHONPATH`（脚本会 import `src.model`）

**运行命令**：

```bash
PYTHONPATH=/data/yhao/baseline/CODI:$PYTHONPATH \
python CODI/plots/rebuttle_dir/analyze_centroid_reference.py \
  --dataset gsm8k \
  --run simcon=CODI/results/latent_sweep_gsm8k/latent_6/models/simcon/gsm8k/run_0 \
  --run simcon_sircl=CODI/results/latent_sweep_gsm8k/latent_6/models/simcon_sircl/gsm8k/run_0 \
  --run codi=CODI/results/latent_sweep_gsm8k/latent_6/models/codi/gsm8k/run_0 \
  --run codi_sircl=CODI/results/latent_sweep_gsm8k/latent_6/models/codi_sircl/gsm8k/run_0 \
  --output-dir CODI/plots/rebuttal_20260328/02_centroid_reference/results_t6
```

**主要输出**：
- `probe_summary.csv` — 每个 run 的 radius\_mean、radius\_std、cos(μ, z₁) 统计
- `offline_intervention_summary.csv` — own / shuffled / wrong centroid 距离对比
- `CENTROID_REFERENCE_SUMMARY.md` — 自动生成的解读报告
- `centroid_probing_summary.png`，`centroid_offline_intervention.png` — 可视化

**已有结果**：`02_centroid_reference/results_t6/`

**关键结论**（T=6）：
- shuffled/own 距离比：simcon=1.84×，simcon\_sircl=1.95×，codi\_sircl=**2.94×**
- centroid 不可随意替换；+SIRCL 后区分度进一步增大
- cos(μ, z₁) 在 SIRCL 后下降（不要声称 centroid 与首步对齐更稳定）

---

### 工具 3：`analyze_sircl_failure_modes.py`

**对应审稿问题**：如果 trajectory 本来就错，centroid 也会错，SIRCL 是否只在 baseline 答对时才有效？

**做什么**：对一对 baseline/+SIRCL 的 run 做逐样本配对分析，统计 wrong→correct（recovered）、correct→wrong（regressed）的样本数，并按正确性分桶计算几何统计。

**注意**：需要在 `CODI/plots/` 目录下运行（依赖 `color_config`）

**运行命令**：

```bash
cd /data/yhao/baseline/CODI/plots

python rebuttle_dir/analyze_sircl_failure_modes.py \
  --baseline-run ../results/latent_sweep_gsm8k/latent_6/models/simcon/gsm8k/run_0 \
  --sircl-run ../results/latent_sweep_gsm8k/latent_6/models/simcon_sircl/gsm8k/run_0 \
  --output-dir rebuttal_20260328/03_failure_modes_boundary/results_t6 \
  --max-samples-per-group 100

# 完成后回到 baseline 根目录
cd /data/yhao/baseline
```

**主要输出**：
- `summary.md` — 准确率、transition 表、几何统计分桶汇总
- `gain_by_baseline_bucket.csv` — baseline correct/wrong 分桶的 +SIRCL 增益
- `transition_summary.csv` — both\_correct / recovered / both\_wrong / regressed 计数
- `geometry_group_summary.csv` — 各 group 的 radius\_mean、diversity 等
- `failure_examples.csv` — 典型失败/恢复案例（含具体预测值和几何数字）
- 4 张 PNG 图（r\_t 分布、diversity 对比等）

**已有结果**：`03_failure_modes_boundary/results_t6/`

**关键结论**（T=6）：
- recovered=119 > regressed=80，净正
- baseline-wrong 分桶：+SIRCL 带来 +19.44 pp 的恢复率
- 方法边界：recovered 样本的 diversity（18.73）> still-wrong（18.43）
- T=6 增益相对 T=16 更小（+2.96 pp vs +13.50 pp），本身印证 scaling 结论

---

### 工具 4：`plot_scaling_stability.py`

**对应审稿问题**：scaling（T=6/16/32）实验为何没有 no-SIRCL matched baseline？

**做什么**：读取 `latent_sweep_gsm8k` 下不同 T 的结果，按 T / condition（no-SIRCL vs +SIRCL）/ family（Sim-CoT vs CODI）聚合，输出 accuracy delta 和轨迹统计趋势。

**注意**：需要在 `CODI/plots/` 目录下运行（依赖 `color_config`）

**运行命令**：

```bash
cd /data/yhao/baseline/CODI/plots

python rebuttle_dir/plot_scaling_stability.py \
  --preset latent_sweep_simcon \
  --output-dir rebuttal_20260328/04_scaling_matched_baseline/results_simcon

python rebuttle_dir/plot_scaling_stability.py \
  --preset latent_sweep_codi \
  --output-dir rebuttal_20260328/04_scaling_matched_baseline/results_codi

cd /data/yhao/baseline
```

**主要输出**：
- `delta_summary.csv` — 每个 T 下的 no-SIRCL vs +SIRCL accuracy delta
- `grouped_summary.csv` — 按 T 分组的聚合统计
- `*_scaling_summary.png` — scaling 趋势图
- `*_summary.md` — 自动生成的中文解读报告

**已有结果**：`04_scaling_matched_baseline/results_simcon/`，`results_codi/`

**关键结论**：

| T | no-SIRCL | +SIRCL | Δ |
|---|----------|--------|---|
| 6 | 53.22% | 56.10% | +2.88 pp |
| 16 | 44.50% | 58.00% | +13.50 pp |
| 32 | 43.06% | 57.01% | +13.95 pp |

no-SIRCL 随 T 增大持续退化（−10.16 pp），+SIRCL 保持稳定。

**no-SIRCL checkpoint 来源**：
- T=16: `CODI_rebuttal_runs/rebuttal_20260325/results/checkpoint_sweeps/decoder-trajectory-euclidean-16long/`
- T=32: `CODI_rebuttal_runs/rebuttal_20260325/results/checkpoint_sweeps/decoder-trajectory-euclidean-32long/`
- +SIRCL T=16: `CODI/results/16long/`（best: ckpt-35988, 58.00%）
- +SIRCL T=32: `CODI/results/32long/`（best: ckpt-29990, 57.01%）

---

### 工具 5：`analyze_trajectory_by_correctness.py`

**对应审稿问题**：几何分析为何只看 all-correct？能否分析失败样本或混合难度子集？

**做什么**：按正确性将样本分为 correct / wrong / all\_correct / all\_wrong / mixed / sircl\_flips / sircl\_regress 共 7 组，对每组计算 per-step r\_t 曲线、PCA 轨迹投影、token cosine similarity heatmap。

**运行命令**：

```bash
python CODI/plots/rebuttle_dir/analyze_trajectory_by_correctness.py \
  CODI/results/latent_sweep_gsm8k/latent_6/models/simcon/gsm8k/run_0 \
  CODI/results/latent_sweep_gsm8k/latent_6/models/simcon_sircl/gsm8k/run_0 \
  --labels simcon simcon_sircl \
  --output-dir CODI/plots/rebuttal_20260328/05_correctness_stratified_trajectories/results_t6 \
  --max-samples 300 \
  --bootstrap 100 \
  --projection pca \
  --max-trajectories 60
```

**主要输出**：
- `trajectory_correctness_summary.csv` — 7 组的 r\_t、diversity、cosine 统计
- `rt_curve_compare_simcon_vs_simcon_sircl.png` — baseline vs +SIRCL 的 per-step r\_t 对比图
- `trajectory_pca_groups_simcon.png`，`trajectory_pca_groups_simcon_sircl.png` — PCA 轨迹图（按组着色）
- `sim_heatmap_simcon.png`，`sim_heatmap_simcon_sircl.png` — token 两两相似度 heatmap

**已有结果**：`05_correctness_stratified_trajectories/results_t6/`

**关键结论**（T=6, Sim-CoT）：

| Group | baseline r\_t | +SIRCL r\_t |
|-------|-------------|------------|
| correct | 13.15 | 8.26 |
| all\_wrong | 11.29 | 8.45 |
| sircl\_flips | 11.54 | 8.44 |
| sircl\_regress | **11.90** | 8.34 |

+SIRCL 后各组 r\_t 收敛到 8.2–8.5 区间。sircl\_flips 的 baseline r\_t（11.54）高于 all\_wrong（11.29），说明"可被救回"的样本在 baseline 时几何展开更多。

---

## 数据依赖关系

所有工具读取的数据均来自：

```
CODI/results/latent_sweep_gsm8k/
└── latent_{N}/
    └── models/{simcon,simcon_sircl,codi,codi_sircl}/
        └── gsm8k/run_0/
            ├── latents.json          ← 工具 1/2/3/5 的核心输入
            ├── trajectory_stats.json ← 工具 1/4 的输入
            └── metrics.json          ← 准确率等
```

这些文件是已有 checkpoint 的推理产物，无需重新训练。如果你要换 checkpoint 或换数据集，替换上面的路径即可。

---

## 注意事项

| 工具 | 运行目录要求 | PYTHONPATH 要求 |
|------|------------|----------------|
| analyze\_latent\_collapse.py | 任意 | 无 |
| analyze\_trajectory\_by\_correctness.py | 任意 | 无 |
| analyze\_sircl\_failure\_modes.py | 必须在 `CODI/plots/` | 无 |
| plot\_scaling\_stability.py | 必须在 `CODI/plots/` | 无 |
| analyze\_centroid\_reference.py | 任意 | `PYTHONPATH=.../CODI` |

`analyze_sircl_failure_modes.py` 和 `plot_scaling_stability.py` 依赖 `color_config.py`，该文件在 `CODI/plots/` 目录下，所以必须从那个目录运行。
