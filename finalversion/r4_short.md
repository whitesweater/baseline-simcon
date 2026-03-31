# Response to Reviewer HNnW

We sincerely thank the reviewer for the constructive feedback. Below we provide a concise response to each question.

---

## Q1: Missing Related Works

We thank the reviewer for pointing out these recent papers. These works are complementary to ours and help better position SIRCL within the broader literature on continuous and latent reasoning.

---

## Q2: Would a linear-trajectory trust region be less restrictive?

We tested a similar idea in Section 5.1 via the **Geodesic Deviation Loss** ($\mathcal{L}_{GDL}$, Eq. 7), which constrains each latent state toward the geodesic between the start and end states — the shortest path in a given geometry. This effectively forces the most direct route. However, it is less effective than SIRCL in practice: on GSM8K, Geodesic reaches only **53.37%**, weaker than SIRCL's **56.10%** (baseline: **53.22%**).

The "zigzag" patterns in latent trajectories are not merely noise—they reflect the Transformer's internal exploratory computation, consistent with geometric analyses showing universal negative step-to-step coherence (Anderson, 2026). A straight-line constraint suppresses this useful exploration. SIRCL instead acts as a **hinge-style trust region**: free movement inside, penalizing only large deviations likely corresponding to semantic drift.

The scaling results confirm this:

| $T$ | No SIRCL | + SIRCL |
| ---: | ---: | ---: |
| 6 | 53.22% | **56.10%** |
| 16 | 44.50% | **58.00%** |
| 32 | 43.06% | **57.01%** |

Without SIRCL, performance degrades sharply as $T$ grows; with SIRCL it stays strong—removing harmful drift while preserving flexibility.

**Ref:** Anderson (2026). The Geometry of Thought. arXiv:2601.13358.

---

## Q3: Hyperparameter selection

Our procedure is simple:

1. Run the no-SIRCL model and measure the baseline mean trajectory radius $\bar{r}$.
2. Set $R$ to about **25% of $\bar{r}$** as the default; for more scattered trajectories, **50%** often works better.
3. Start from $\lambda = 0.1$ and check that $\lambda \mathcal{L}_{SIRCL}$ is neither dominant nor negligible relative to the task loss.

We will add this guidance to the revised manuscript.

---

## Q4: Different backbones

We thank the reviewer for this suggestion. For this rebuttal, we report the completed **LLaMA-3.2-3B-Instruct** results:

| Dataset | SIM-CoT | SIM-CoT+SIRCL | Delta |
| --- | ---: | ---: | ---: |
| GSM8K | 59.97% | **63.23%** | **+3.26pp** |
| GSM-Hard | 14.18% | **15.16%** | **+0.98pp** |
| ASDiv | 72.23% | **72.41%** | **+0.18pp** |
| Best avg | 30.67% | **31.55%** | **+0.88pp** |

These results reproduce the main trend: SIRCL improves in-domain GSM8K and overall average, while remaining competitive on out-of-domain sets.

---

## Q5: Qualitative comparisons

Appendix D provides a geometric analysis on **526 GSM8K samples**. Three key findings:

**1. More direct trajectories.** Without SIRCL, CODI accumulates ~107 units of path length on the matched 526-sample subset, with a low final path efficiency ($\eta_T \approx 0.13$). Adding SIRCL reduces path length to ~35.6 (−66.7%) and raises path efficiency ($\eta_T \approx 0.49$), indicating much more direct latent trajectories.


**2. Faster, more monotonic convergence.** CODI+SIRCL achieves smaller distances to the final token from iteration 1 onward with tighter variance (Figure 8). Convergence rate improves by **+457%** for CODI and **+16%** for SIM-CoT.

**3. More coherent intermediate states.** The cosine similarity heatmaps (Figure 10) show CODI without SIRCL has fragmented patterns (similarity dropping below 0.5), while +SIRCL produces a coherent block structure across iterations. Trajectory smoothness improves by **+58.6%** (CODI) and **+245.0%** (SIM-CoT).

Overall, SIRCL does not merely improve accuracy—it makes latent reasoning shorter, more direct, faster-converging, and more self-consistent.

---

## Q6: Conceptual Novelty vs. Center Loss

Center loss (Wen et al., 2016) is indeed effective for representation learning. Our work extends this geometric intuition to **latent-token reasoning trajectories**, with non-trivial adaptations:

1. **From static embeddings to dynamic trajectories.** Center loss regularizes a single feature vector. SIRCL regularizes an entire *sequence* $\{z_t\}_{t=1}^T$ recursively generated across reasoning steps, with centroid $\mu$ computed on the fly per sample — no shared class prototype.
2. **Hinge-style trust region.** Some oscillation is functional in latent reasoning (cf. our geodesic ablation, Section 5.1). SIRCL uses a hinge loss with radius $R$: tokens inside the region are unconstrained; only excessive deviations are penalized.
3. **Temporally coupled states.** Each $z_t$ directly conditions $z_{t+1}$. Stabilizing such a trajectory requires balancing global coherence with local flexibility — the central challenge SIRCL addresses.
