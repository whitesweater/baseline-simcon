We thank the reviewer for the feedback. Below we respond concisely.
---
## Q1: Structured contraction vs. representation collapse
We thank the reviewer for raising this point. A more precise statement is: **SIRCL directly mitigates drift, and its hinge design avoids introducing collapse**.
The distinction matters. A naïve centroid loss (e.g., center loss) would continuously pull *all* states toward the centroid, actively causing collapse. SIRCL's hinge formulation instead applies **zero gradient within radius $R$**, leaving tokens inside the trust region free to explore diverse representations. 
Matched scaling further supports this: if SIRCL caused collapse, longer chains would amplify the damage. Instead, without SIRCL accuracy drops from **53.22%** to **43.06%** as `T` grows to 32, whereas with SIRCL it remains at **57.01%**. We will revise the paper to clarify this distinction.

---
## Q2: Evidence for the centroid as a semantic anchor, not just a geometric mean
We will revise it to a **trajectory-level geometric anchor**. Our centroid-replacement intervention on GSM8K (`T=6`, `N=1319`) supports this claim: replacing each sample's own centroid with a shuffled or wrong-sample centroid substantially increases token-to-anchor distance. The shuffled/own ratio is **1.84x** without SIRCL and **1.95x** with SIRCL for Sim-CoT, and **1.58x** without SIRCL and **2.94x** with SIRCL for CODI. So the centroid is not an arbitrary mean, it carries **sample-specific reference information**, and that reference becomes more discriminative after SIRCL.

---

## Q3: Does SIRCL only help when the baseline trajectory is already reasonable?

T=6 gains are moderate with some degradation cases. SIRCL's effectiveness depends on retained geometric structure, not answer correctness.

**Failure conditions and cases:** (1) *Off-manifold trajectory* (as noted in Sec. 4.2): when baseline fundamentally departs from task logic (e.g. COCONUT on GSM-Hard: 8.1% < No-CoT 10.8%), the centroid loses informativeness. (2) *Premature compression*: Sample 175's radius shrinks 11.31→3.16, diversity 6.18, path 48.80; both baseline/+SIRCL remain wrong (10/0.5 vs. gold 15). Analysis (SIM-CoT, T=6, GSM8K, N=1319) shows wrong samples exhibit **premature contraction** (radius=11.37, diversity=18.20 vs. correct: 13.15, 20.71).


---

## Q4: Matched no-SIRCL baseline for the scaling experiment

We have added the matched no-SIRCL baseline under the same backbone, dataset, and optimizer:

| $T$ | SIM-CoT (no SIRCL) | SIM-CoT+SIRCL | $\Delta$ |
| --- | ---: | ---: | ---: |
| 6 | 53.22% | 56.10% | +2.88pp |
| 16 | 44.50% | 58.00% | +13.50pp |
| 32 | 43.06% | 57.01% | +13.95pp |

Without SIRCL, performance degrades sharply as the latent-token budget grows; with SIRCL, it remains stable and substantially stronger.

---

## Q5: Geometric analysis beyond the 526 all-correct samples

The original purpose of the 526 all-correct subset was to isolate trajectory-shape differences while minimizing confounding from answer correctness. We now extend the analysis to **all 1,319 GSM8K samples** (correct + wrong):

| Metric | CODI | +SIRCL | SIM-CoT | +SIRCL |
|---|---:|---:|---:|---:|
| Cos sim  | 0.749 | 0.915 | 0.743 | 0.790 |
| Dist  | 13.578 | 5.677 | 16.485 | 10.412 |
| Compact  | 10.265 | 5.049 | 12.335 | 8.384 |
| Smooth| −0.867 | −0.321 | 0.005 | 0.022 |
| Path  | 109.178 | 37.313 | 87.138 | 55.898 |
| Conv. | −0.411 | −2.135 | −2.276 | −2.364 |

**All improvement directions and relative magnitudes are preserved** compared to the 526-sample subset: (1) more direct trajectories (CODI path 109.2→37.3), (2) faster convergence (conv. rate gap intact), (3) more coherent states (CODI+SIRCL cos sim = 0.915 on both subsets). 
---

## Q6: Revising the "consistent performance gains" claim
We will revise this wording. SIRCL improves many benchmark–baseline combinations, though some degradations exist. Its clearest strength is stability under longer latent-token budgets, where it delivers the largest gains.


---

## Q7: Positioning relative to explicit CoT-SFT

Our claim is **not** that implicit reasoning with SIRCL already surpasses strong explicit CoT-SFT in raw accuracy. Rather, the takeaway is that SIRCL makes implicit reasoning **more stable, more scalable, and more practical**, while preserving **no extra reasoning-token overhead at inference time**.

---

## Q8: Interpreting the large difference in optimal radius $R$ across methods

The absolute value of the best radius $R$ should **not** be compared directly across methods, because the native scale of the latent trajectories differs by architecture. Our practical procedure is simple: measure the baseline mean trajectory radius $\bar{r}$; set $R$ to about **25% of $\bar{r}$** as the default, or **50%** for more scattered trajectories; then start from $\lambda = 0.1$ and verify that $\lambda \mathcal{L}_{SIRCL}$ is neither dominant nor negligible.
