# Response to Reviewer 3

We sincerely thank the reviewer for the thoughtful evaluation. Below we provide a shorter response to the main concerns.

---

## Q1: Theory / diversity suppression & collapse concern

The key geometric phenomenon we observe without SIRCL: zig-zagging trajectories where latent states repeatedly overshoot and wander, which is consistent with Anderson (2026), who attributes such oscillation to general Transformer residual-stream dynamics rather than task-specific artifacts. 

Appendix D provides direct evidence for our interpretation. On the **526 GSM8K samples correctly solved by all four variants**, SIRCL makes latent trajectories shorter, smoother, and more coherent. For example, path length drops by **66.7%** for CODI and **41.7%** for SIM-CoT; smoothness improves by **58.6%** and **245.0%**; and convergence rate improves by **457.4%** and **15.9%**.

**SIRCL does not suppress diversity or cause collapse.** The hinge-style trust region applies zero gradient to tokens within radius $R$ — states inside the trust region are completely unconstrained and free to maintain diverse representations. Only tokens drifting *beyond* $R$ receive corrective gradient. This means SIRCL bounds the maximum deviation without compressing the distribution within the trust region. A naïve centroid loss (e.g., center loss) would continuously pull *all* states inward, actively causing collapse; SIRCL's hinge design avoids this by construction.
Empirically, if SIRCL suppressed diversity, longer chains would accumulate more collapse and perform worse. Instead, matched scaling gives the opposite pattern: without SIRCL, accuracy drops from **53.22%** to **44.50%** and **43.06%** as `T` grows from 6 to 16 and 32; with SIRCL, it remains strong at **56.10%**, **58.00%**, and **57.01%**. The observed effect is therefore better interpreted as **more efficient and more coherent latent reasoning**, not collapse.

---

## Q2: Single-backbone limitation / harder benchmarks

We thank the reviewer for this suggestion. We have added results on **LLaMA-3.2-3B-Instruct**, including harder benchmarks. We note that AIME and MATH are extremely challenging for 1B-scale models (often scoring near zero), offering little discriminative power; we therefore evaluate on the 3B backbone:

| Benchmark | SIM-CoT | SIM-CoT+SIRCL |
| --- | ---: | ---: |
| GSM8K | 59.97 | **63.31** |
| GSM-Hard | 14.25 | **15.16** |
| SVAMP | 73.3 | **74.3** |
| ASDiv | 72.23 | **72.41** |
| Math500 | 6.4 | **6.8** |

SIRCL transfers to the larger backbone with consistent gains across all five benchmarks.

---

## Q3: Performance gap to SIM-CoT [1] / why Coconut gains are smaller

We reproduced SIM-CoT following [1] using the authors' official codebase. Small numerical differences may stem from hardware/software, but all Table 1 methods are trained identically for fair comparison.

SIRCL provides consistent gains:

| Method | GSM8K | CommonsenseQA | SVAMP | GSM-Hard | ASDiv |
| --- | --- | --- | --- | --- | --- |
| CODI | 52.9 | 70.3 | 59.9 | 12.1 | 65.4 |
| CODI+SIM-CoT ("SIM-CoT" in Table 1) | 53.2 | 71.0 | 60.2 | 12.4 | 64.5 |
| CODI+SIRCL | 55.3 | 70.8 | 60.4 | 13.3 | 64.1 |
| CODI+SIM-CoT+SIRCL ("SIM-CoT+SIRCL" in Table 1) | **56.1** | **71.8** | **60.6** | 12.7 | **65.5** |

When added on top of SIM-CoT, the combined model improves all five benchmarks. SIRCL's benefit amplifies with chain length:

| $T$ | SIM-CoT (no SIRCL) | SIM-CoT+SIRCL | $\Delta$ |
| --- | --- | --- | --- |
| 6 | 53.22% | 56.10% | +2.88pp |
| 16 | 44.50% | 58.00% | +13.50pp |
| 32 | 43.06% | 57.01% | +13.95pp |

At T=6, gains are moderate—but this is precisely because short chains leave limited room for drift.We therefore view SIRCL as a **lightweight training-time regularizer** that is complementary to SIM-CoT, adds **no inference-time cost**, and introduces only minor training overhead.

We also acknowledge that gains on Coconut are smaller. Our interpretation is not incompatibility, but **weaker global trajectory coherence**. In the Coconut setup we use, reasoning is constructed **stage by stage**, and the final fully implicit chain contains only **6 latent tokens**. Because the trajectory is built in segments, the centroid is a less reliable summary of the full reasoning process. We therefore interpret the smaller gains on Coconut as arising from **weaker trajectory consistency and a less representative centroid**.

---

## Q4: Typographical and formatting errors

We thank the reviewer for pointing out these issues. We will correct the editorial problems and carefully proofread the revision.

For the unreferenced result in Line 386: it corresponds to the **Geodesic Deviation Loss** experiment in Section 5.1 ($\mathcal{L}_{GDL}$, Eq. 7), i.e. the third column of Figure 6. The result is **53.37**, which is lower than standard SIRCL (**56.1**) and slightly above the baseline (**53.22**). We will make this cross-reference explicit in the revised manuscript.