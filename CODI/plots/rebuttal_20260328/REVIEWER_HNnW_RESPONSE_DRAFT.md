# Response to Reviewer HNnW

We sincerely thank the reviewer for the constructive and detailed feedback, and for recognizing the effectiveness, efficiency, and comprehensive analysis of our work. Below we address each question.

---

## Q1: Missing Related Works

We thank the reviewer for pointing out these relevant recent works. In the revised manuscript, we will cite and discuss **Soft Thinking** (arXiv:2505.15778), **Reasoning Path Compression** (arXiv:2505.13866), and **A Survey on Latent Reasoning** (arXiv:2507.06203). These papers are complementary to our work and help situate SIRCL more clearly within the broader landscape of continuous and latent reasoning.

---

## Q2: Whether a Trust Region Around a Linear Trajectory Would Be Less Restrictive

This is an excellent question, and we have in fact investigated this exact direction. In **Section 5.1** of our paper, we introduced the **Geodesic Deviation Loss** ($\mathcal{L}_{GDL}$, Eq. 7), which constrains each latent token to stay close to the *linear interpolation* between the start and end states:

$$\hat{z}_t = \left(1 - \frac{t}{T}\right) z_1 + \frac{t}{T} z_T, \quad \mathcal{L}_{GDL} = \frac{1}{T} \sum_{t=1}^{T} \| z_t - \hat{z}_t \|_2$$

As shown in **Figure 6**, this linear-trajectory constraint is weaker than centroid-based SIRCL: Geodesic achieves **53.37%** on GSM8K, while Euclidean SIRCL achieves **56.10%** (with the matched unregularized baseline at **53.22%**).

Our interpretation is that the "zigzag" patterns in latent trajectories are **not** merely stochastic noise to be removed. They appear to reflect the Transformer's internal exploratory computation. Forcing the trajectory to stay near a single straight geodesic suppresses this useful local exploration.

In contrast, SIRCL defines a bounded feasible region around the sample-specific trajectory centroid. This acts as a **soft trust region**: it allows free movement inside the region, while only penalizing unusually large deviations that are more likely to correspond to semantic drift. In this sense, SIRCL stabilizes the trajectory **without flattening it**.

We also find that the current constraint preserves enough flexibility to recover trajectories that the baseline initially gets wrong. In our boundary analysis, among the **617 baseline-wrong** GSM8K samples, **+SIRCL solves 19.12% of them** (i.e., **118 recovered cases**). This would be unlikely if the constraint were simply over-restrictive. Moreover, within the baseline-wrong subset, the recovered cases show a larger baseline radius than the still-wrong cases (**11.86 vs. 11.31**), suggesting that SIRCL is able to retain and organize a recoverable amount of exploration rather than collapsing the trajectory indiscriminately.

This is also consistent with our scaling results: the matched no-SIRCL baseline degrades sharply as the chain grows ($T$=6→16→32: 53.22%→44.50%→43.06%), whereas SIRCL remains stable and becomes increasingly advantageous (56.10%→58.00%→57.01%). Together, these results suggest that the current constraint preserves enough flexibility for latent reasoning while still reducing excessive drift.

---

## Q3: Guidelines for Hyperparameter Selection

We thank the reviewer for this practical question. Our procedure is straightforward:

1. **Measure the baseline radius.** We first run the no-SIRCL model and compute the mean per-step distance $\bar{r} = \frac{1}{T}\sum_t d(z_t, \mu)$ over the training set.
2. **Set $R$ relative to this radius.** We use $R \approx 25\%$ of the baseline $\bar{r}$ as the default. For models with especially scattered trajectories, relaxing to $\approx 50\%$ works better.
3. **Calibrate $\lambda$ against the task loss.** We start from $\lambda = 0.01$ and verify that $\lambda \mathcal{L}_{SIRCL}$ is neither dominant nor negligible compared to the task loss.

We will add this guidance to the revised manuscript.

---

## Q4: Experiments on Different Backbones

We agree that cross-backbone evaluation strengthens the paper. For this rebuttal, we therefore report only the additional backbone that is already complete and safe to cite: **LLaMA-3.2-3B-Instruct**. To avoid over-claiming, we do **not** use our Qwen runs as rebuttal evidence here: the Qwen3-4B line is excluded, and the Qwen3-1.7B experiments are still in progress and will be added only in the final version after completion.

**LLaMA-3.2-3B (SIM-CoT, completed checkpoint sweep; best-average checkpoint):**

| Dataset | SIM-CoT | SIM-CoT+SIRCL | $\Delta$ |
| --- | ---: | ---: | ---: |
| GSM8K | 59.97% | **63.23%** | **+3.26pp** |
| MATH500 | **6.80%** | 6.40% | -0.40pp |
| AIME | 0.00% | 0.00% | 0.00pp |
| GSM-Hard | 14.18% | **15.16%** | **+0.98pp** |
| ASDiv | **72.41%** | 72.23% | -0.18pp |
| Best avg | 30.67% | **31.55%** | **+0.88pp** |

This additional backbone reproduces the main pattern of the paper: SIRCL improves the in-domain GSM8K result and the overall average, while remaining broadly competitive on the out-of-domain sets. We will include these LLaMA-3B results in the revised manuscript, and we will add the Qwen-family results only after the corresponding runs are fully completed.

---

## Q5: Qualitative Comparisons

Yes. Our paper already includes a comprehensive geometric analysis in **Appendix D** (526 all-correct samples, Table 6, Figures 7–11). Below we highlight the key qualitative findings and supplement them with a new correctness-stratified analysis on the full 1,319 test samples.

### Appendix D: How SIRCL Reshapes Trajectory Geometry (526 All-Correct Samples)

To isolate the effect of SIRCL on *trajectory quality* (rather than task accuracy), Appendix D analyzes the 526 GSM8K samples correctly solved by all four model variants. We summarize six dimensions of improvement:

| Metric | CODI → +SIRCL | SIM-CoT → +SIRCL |
| --- | ---: | ---: |
| Cosine sim. to final state ↑ | +19.9% | +9.6% |
| Distance to final state ↓ | −58.9% | −43.5% |
| Cluster compactness ↓ | −52.1% | −37.9% |
| Trajectory smoothness ↑ | +58.6% | +245.0% |
| Total path length ↓ | −66.7% | −41.7% |
| Convergence rate (slope mag.) ↑ | +457.4% | +15.9% |

These numbers reveal three qualitative conclusions:

**1. SIRCL eliminates oscillatory wandering and produces more direct trajectories.** Without SIRCL, CODI exhibits strong non-monotonicity in its distance-to-final-token curves (Figure 8): intermediate states oscillate rather than steadily approaching the solution. This is also visible in the cumulative path length plot (Figure 7, bottom-left), where unconstrained CODI accumulates ~107 units of total movement yet only needs ~5 units of net displacement. After adding SIRCL, path length drops by 66.7% (to ~36 units), and path efficiency $\eta_t$ rises sharply (Figure 7, bottom-right), meaning a much larger fraction of latent movement contributes to net progress rather than detours.

**2. SIRCL yields earlier and more monotonic convergence to the final reasoning state.** The boxplots in Figure 8 show that CODI+SIRCL achieves markedly smaller distances to the final token from iteration 1 onward, with tighter interquartile ranges (less cross-sample variance). For SIM-CoT, which starts with the largest distance scale among all variants, +SIRCL substantially reduces distances across all iterations. The convergence rate (linear slope of $d_t$ vs. $t$) improves by +457% for CODI and +16% for SIM-CoT, indicating that the trajectory approaches the solution both faster and more steadily.

**3. SIRCL produces globally more coherent intermediate states.** The inter-iteration cosine similarity heatmaps (Figure 10) provide perhaps the most striking qualitative contrast. CODI without SIRCL shows a fragmented pattern: similarity between early and late iterations can drop below 0.5, revealing abrupt representational jumps mid-reasoning. CODI+SIRCL, by contrast, exhibits near-uniformly high similarity (≥0.9) across iterations 2–7, forming a coherent "block" structure — intermediate states lie on a consistent semantic manifold. Similarly, SIM-CoT+SIRCL increases cross-iteration coherence relative to SIM-CoT, especially among later iterations. Combined with the +58.6%/+245.0% improvement in trajectory smoothness (step-direction consistency $S$), this confirms that SIRCL reduces directional drift and produces more semantically aligned reasoning progressions.

Taken together, the Appendix D analysis demonstrates that SIRCL does not merely improve task accuracy — it fundamentally reshapes how the model reasons in latent space, making trajectories *shorter, more direct, faster-converging, and more self-consistent*.

### Supplementary: Correctness-Stratified Analysis (All 1,319 GSM8K Samples)

To complement Appendix D (which controls for correctness), we additionally stratify the full test set by correctness transition (baseline → +SIRCL):

| Transition | Count | Share |
| --- | ---: | ---: |
| Both correct | 622 | 47.2% |
| Recovered by +SIRCL | 118 | 8.9% |
| Regressed with +SIRCL | 80 | 6.1% |
| Both wrong | 499 | 37.8% |
| **Net gain** | **+38** | **+2.88pp** |

The geometric signatures vary systematically across these groups:

| Category | $n$ | Baseline $\bar{r}$ | +SIRCL $\bar{r}$ | Radius Red. | Path Red. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stable correct | 622 | 13.28 | 8.38 | **36.9%** | **40.7%** |
| Recovered | 118 | 11.86 | 8.51 | 28.3% | 30.7% |
| Regressed | 80 | 12.11 | 8.40 | 30.7% | 36.8% |
| Stable wrong | 499 | 11.31 | 8.35 | 26.2% | 30.0% |

Two observations extend Appendix D to the full dataset: **(1)** Correct trajectories have *larger* baseline radius (13.28) than wrong ones (11.31) — failure is associated with geometric contraction, not excessive exploration. **(2)** SIRCL compresses all groups into a narrow band ($\bar{r}$≈8.3–8.5), with the largest reduction on stable-correct trajectories. The gains concentrate on **medium-to-hard** problems (3–6 reasoning steps: net +37), consistent with the hypothesis that geometric regularization is most useful where drift has room to accumulate.

**Representative recovered cases** (baseline wrong → +SIRCL correct):

| Question (abridged) | Gold | BL | +SIRCL | $\bar{r}$: BL→SL |
| --- | ---: | ---: | ---: | --- |
| *"Lani baked 55 cookies, ate 5, placed rest into 5 jars…"* | 10 | 6 ✗ | 10 ✓ | 14.52→7.85 |
| *"Andrew: 6 days by bus, half by car, round trip…"* | 9 | 12 ✗ | 9 ✓ | 12.95→6.57 |
| *"Gissela 4000 lbs, Gordy +800, total 11600…"* | 2800 | 3200 ✗ | 2800 ✓ | 14.34→8.08 |

Regressions (80 cases) are fewer than recoveries (118), and the net effect is consistently positive. We will add these correctness-stratified results and case studies to the revised appendix.

---

## Q6: Conceptual Novelty vs. Center Loss

We thank the reviewer for this observation. SIRCL is indeed inspired by center-loss-style geometric regularization, and we view this connection positively — center loss (Wen et al., 2016) has proven highly effective in representation learning, and our work extends this geometric intuition to a new and underexplored setting: **latent-token reasoning trajectories**.

The key novelty of SIRCL lies in adapting this idea to the unique structure of implicit reasoning, which involves several non-trivial design choices:

1. **From static embeddings to dynamic trajectories.** Center loss regularizes a single feature vector per sample. In latent reasoning, we regularize an entire *sequence* of latent tokens $\{z_t\}_{t=1}^T$ that are recursively generated and consumed across reasoning steps. The centroid $\mu$ is computed on the fly from the current sample's trajectory — there is no shared class prototype.

2. **Hinge-style trust region instead of continuous pull.** In latent reasoning, some oscillation and local exploration are functional (as evidenced by our geodesic ablation in Section 5.1, where stronger trajectory straightening hurts performance). SIRCL therefore uses a hinge loss with radius $R$: tokens inside the feasible region are unconstrained, and only excessive deviations are penalized. This preserves the Transformer's native exploratory dynamics while preventing semantic drift.

3. **Applied to temporally coupled reasoning states.** Unlike classification features, latent reasoning tokens are temporally coupled — each $z_t$ directly conditions the generation of $z_{t+1}$. Stabilizing such a trajectory requires balancing global coherence with local flexibility, which is the central challenge SIRCL addresses.

We acknowledge center loss as an important precursor and will make this connection more explicit in the revised manuscript, while clarifying the adaptations required for the latent reasoning setting.
