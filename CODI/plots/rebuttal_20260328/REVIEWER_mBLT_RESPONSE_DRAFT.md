# Response to Reviewer mBLT

We sincerely thank the reviewer for the thoughtful evaluation and for acknowledging the clear motivation, straightforward method design, and strong significance/originality of our work. Below we address each concern in detail.

---

## Q1: Lack of Theoretical Explanation; Concern That Centripetal Constraint May Suppress Diversity and Cause Model Collapse

We appreciate this important question. We agree that SIRCL is best characterized as a principled heuristic grounded in geometric intuition, rather than a method derived from first principles of Transformer theory. We position it accordingly.

That said, we can offer both **empirical evidence** and **geometric reasoning** for why SIRCL works and why it does **not** cause model collapse:

### Why SIRCL does not cause collapse

The key design choice that prevents collapse is the **hinge-style trust region** (Eq. 5): SIRCL only penalizes latent tokens that drift *beyond* radius $R$ from the centroid, while leaving tokens inside the feasible region **completely unconstrained**. This is fundamentally different from a continuous L2 pull (such as center loss), which would always compress representations toward a point.

We offer two arguments — one structural, one empirical — for why SIRCL does not lead to collapse:

**Structural argument: the hinge gradient is exactly zero inside the trust region.** For any token $z_t$ with $d(z_t, \mu) \leq R$, the SIRCL loss contributes zero gradient. This means SIRCL *cannot* compress tokens that are already within the feasible region — it has no mechanism to pull them closer to the centroid. The only tokens that receive a corrective gradient are those that have drifted *unusually far*. In contrast, true collapse would require a persistent inward force on all tokens (as in standard L2 regularization or center loss), which the hinge design explicitly excludes. Furthermore, the centroid $\mu$ is computed **per-sample on the fly** from the current trajectory — there is no shared global attractor across the dataset. Different samples are free to have completely different trajectory geometries; SIRCL only constrains each trajectory relative to its own center.

**Empirical argument: if SIRCL caused collapse, longer chains would perform worse, not better.** The most direct test of the collapse hypothesis is scaling. If SIRCL were progressively destroying representational diversity, then applying it over more latent iterations ($T$) should amplify the damage — we would expect accuracy to degrade as $T$ grows. The opposite is observed:

| $T$ | SIM-CoT (no SIRCL) | SIM-CoT+SIRCL | $\Delta$ |
|-----|---------------------|---------------|----------|
| 6   | 53.22% | 56.10% | +2.88pp |
| 16  | 44.50% | **58.00%** | **+13.50pp** |
| 32  | 43.06% | **57.01%** | **+13.95pp** |

Without SIRCL, longer chains degrade by nearly 10pp ($T$=6→32), consistent with accumulated drift. With SIRCL, accuracy *improves* as $T$ grows and the advantage widens to nearly 14pp. This scaling behavior is **incompatible with collapse**: a collapsing regularizer would amplify degradation at larger $T$, not reverse it. Instead, SIRCL's hinge constraint acts as a stabilizer that bounds drift accumulation while leaving the model's internal exploratory dynamics intact.

### Geometric intuition for why SIRCL works

The centripetal hinge loss implements a simple but effective inductive bias: implicit reasoning should remain *focused on the problem context*. The trajectory centroid $\mu$ serves as a trajectory-level geometric reference of where the model's latent reasoning has been, and the radius $R$ defines a trust region around it. When latent states drift beyond this region, the hinge loss provides a corrective gradient that pulls them back, preventing unbounded semantic drift — while leaving all within-region dynamics entirely to the model.

---

## Q2: Single Backbone Limitation; Need for More Challenging Benchmarks

We fully acknowledge the limitation of the single-backbone evaluation in our original submission. In response, we have extended our experiments to **LLaMA-3.2-3B-Instruct** and evaluated on harder benchmarks including **Math500** and **AIME**.

### Cross-Backbone Results (LLaMA-3B)

**SIM-CoT on LLaMA-3.2-3B (best checkpoint per dataset):**

| Dataset | SIM-CoT | SIM-CoT+SIRCL | $\Delta$ |
|---------|---------|---------------|----------|
| GSM8K | 59.97% | **63.31%** | **+3.34pp** |
| GSM-Hard | 14.25% | **15.16%** | +0.91pp |
| ASDiv | 72.41% | 72.28% | −0.13pp |
| Math500 | 8.80% | 7.20% | −1.60pp |
| AIME | 3.33% | **6.67%** | +3.33pp |

The trend is consistent with our LLaMA-1B findings: SIRCL provides the largest gains on the in-domain benchmark (GSM8K: **+3.34pp**, slightly larger than the +2.88pp on 1B) and the hardest out-of-domain task (AIME: +3.33pp). This demonstrates that **SIRCL's effectiveness transfers to a larger backbone**.

We additionally have **CODI+SIRCL on LLaMA-3B** (52.39% on GSM8K, 72.32% on ASDiv) and **CoT-SFT baselines** (72.40% for LLaMA-3B, 73.62% for Qwen3-4B on GSM8K). Experiments on Qwen3-4B for implicit reasoning methods are in progress and will be included in the final version.

### More Challenging Benchmarks

As shown in the table above, we have evaluated on **Math500** and **AIME** for both LLaMA-1B and LLaMA-3B. While the absolute accuracy on these extremely challenging benchmarks remains limited for 1B/3B-scale models (which is expected—even explicit CoT methods achieve modest results at this scale), the SIRCL trend is encouraging:

- On **AIME** (competition-level math), SIRCL doubles the accuracy on LLaMA-3B (3.33%→6.67%), though absolute numbers remain small.
- On **Math500**, the results are mixed, which is consistent with our honest assessment that SIRCL's primary strength lies in **stability during scaling** rather than universal accuracy gains on every benchmark.

We will include these extended results in the revised manuscript.

---

## Q3: Performance Gap Between SIRCL and SIM-CoT [1]

We thank the reviewer for the careful cross-paper comparison. We address this from three aspects.

### 1. Our SIM-CoT reproduction faithfully follows the original paper

We reproduced SIM-CoT strictly following the methodology described in [1] and based on **the authors' official open-source codebase**. Minor numerical differences from the numbers reported in [1] are due to hardware/software environment differences (their B200 vs. our H800 GPUs). All methods in our Table 1 are trained under **identical conditions**, ensuring fair within-table comparison.

### 2. SIRCL provides substantial and consistent gains

We note that **"SIM-CoT" in our Table 1 corresponds to CODI+SIM-CoT** — i.e., SIM-CoT is already built on top of CODI. Thus the fair comparison is **CODI+SIRCL vs. CODI+SIM-CoT (= "SIM-CoT" in the table)**. Under this comparison, CODI+SIRCL outperforms SIM-CoT on all three benchmarks (GSM8K: 55.3% vs 53.2%; SVAMP: 60.4% vs 60.2%; GSM-Hard: 13.3% vs 12.4%), demonstrating that SIRCL alone already surpasses SIM-CoT's decoder-based approach.

Moreover, combining both techniques (CODI+SIM-CoT+SIRCL) achieves the highest overall performance (56.1% on GSM8K), confirming that SIRCL and SIM-CoT are complementary.

The advantage becomes even more pronounced when scaling the number of latent iterations $T$:

| $T$ | SIM-CoT (no SIRCL) | SIM-CoT+SIRCL | $\Delta$ |
|-----|---------------------|---------------|----------|
| 6   | 53.22% | 56.10% | +2.88pp |
| 16  | 44.50% | **58.00%** | **+13.50pp** |
| 32  | 43.06% | **57.01%** | **+13.95pp** |

Without SIRCL, SIM-CoT degrades by nearly 10pp as $T$ grows from 6 to 32, indicating accumulated latent drift. With SIRCL, accuracy instead *improves*, widening the advantage to nearly **14pp**. This scaling behavior demonstrates that SIRCL's performance gains are not only consistent but **amplify with increased reasoning depth**.

### 3. When SIRCL succeeds and when it does not

We honestly acknowledge that SIRCL's gains on Coconut are smaller than on CODI. Our revised interpretation is that the key question is **when the trajectory centroid remains informative enough for SIRCL to help**.

**When does SIRCL succeed?** In our paired analysis at $T$=16, the main gain comes from initially wrong samples. Within the baseline-wrong subset, +SIRCL corrects 27.79% of the samples, and the corrected cases have distinctly larger trajectory radius and token diversity than the still-wrong ones (radius: 9.83 vs. 8.66; diversity: 14.45 vs. 12.77; path length: 177.26 vs. 153.61). This indicates that successful correction requires the trajectory to retain enough task-relevant geometric structure for the centroid to remain a meaningful anchor.

**When does SIRCL fail?** The clearest failure boundary suggested by our data is the case where the latent chain contracts too early: later states have very low radius, diversity, and path length, so the centroid is no longer summarizing an active reasoning process, but only the center of an already impoverished trajectory. In that regime, geometric regularization can stabilize the path, but it cannot recover reasoning information that has already been lost. Sample 175 illustrates this clearly: its trajectory shrinks from early radius 11.31 to late radius 3.16, with mean radius 4.12, diversity 6.18, and path length 48.80, and both baseline and +SIRCL remain wrong (10 and 0.5 vs. gold 15). Sample 935 shows the same pattern even more sharply (11.24 -> 2.18, mean radius 4.02, diversity 6.21, path length 54.31).

At the same time, we do not claim that every unrecoverable sample is of this form. Some still-wrong trajectories remain geometrically large, e.g., sample 268 (radius 14.50, diversity 20.86, path length 324.78). Thus unrecoverable failures are heterogeneous. Our narrower claim is that **SIRCL becomes weak whenever the centroid no longer carries useful task information**, and early contraction is the most directly supported and most relevant boundary for explaining the Coconut results.

**Case contrast.** Sample 220 illustrates the recoverable regime: baseline predicts 80 instead of the gold 70, with radius 14.75 and path length 326.78; after SIRCL, the answer becomes 70 while the trajectory contracts to radius 5.64 and path length 81.93. In other words, SIRCL helps when a rich trajectory drifts but still preserves enough structure to be pulled back. It fails when the trajectory has already become too compressed to contain useful corrective signal.

**Connecting to Coconut.** The same applicability boundary is visible across chain lengths: at $T$=6, SIRCL's recovery rate is only 9.0% (119/1319), versus 14.6% (192/1319) at $T$=16. Shorter chains provide fewer informative trajectories for SIRCL to exploit. In the Coconut training configuration we actually use, reasoning is introduced by a stage-wise replacement curriculum: for a 3-step rationale, training moves from 3 explicit segments, to 1 latent stage + 2 explicit segments, then 2 + 1, and finally 3 latent stages + 0 explicit segments. Since each latent stage uses 2 latent tokens, the final fully implicit chain contains only 6 latent tokens. This means Coconut constructs reasoning through a short segment-level latent chain rather than a longer end-to-end latent trajectory. As a result, trajectory continuity across the full reasoning process is inherently limited, and SIRCL has less coherent trajectory structure to regularize. Therefore, the smaller gains on Coconut are better interpreted as arising from **more limited trajectory coherence**, rather than from any intrinsic incompatibility between SIRCL and Coconut.

---

## Q4: Typographical and Formatting Errors

We thank the reviewer for the careful reading. Regarding these editorial issues, we appreciate the reviewer for pointing them out, and we will correct them in the revised manuscript and carefully proofread the paper.

- **Unreferenced experimental results** mentioned in Line 386: this experiment is the Geodesic Deviation Loss experiment introduced in **Section 5.1** (`\mathcal{L}_{GDL}`, Eq. 7), corresponding to the third column of Figure 6. The result is **53.37**, which is lower than our standard SIRCL result (**56.1**) and slightly above the baseline (**53.22**). We will revise the manuscript to make the formatting, cross-referencing, and wording clearer and more explicit.
