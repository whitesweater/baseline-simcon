# Response to Reviewer RWVq

We sincerely thank the reviewer for the careful and constructive feedback. The questions helped us sharpen both the empirical claims and the intended scope of the paper. Below we revise our responses to be more precise and better aligned with the evidence we currently have.

---

## Q1: Direct Evidence on "Preventing Collapse" vs. Simply Shrinking Trajectories

We agree with the reviewer that the current hinge loss more directly penalizes **excessive deviation from the trajectory center** than it directly enforces "anti-collapse" in a generic sense. We will therefore revise the manuscript to make a more precise claim: **SIRCL does not prevent all contraction; rather, it replaces unconstrained drift with structured geometric contraction that is empirically more useful for reasoning.**

To directly examine this point, we added a simple anisotropy diagnostic on the standard GSM8K setting (`T=6`, `N=300`, 100x bootstrap): **RandomSim**. For Sim-CoT, RandomSim increases only mildly from **0.424** to **0.450**, while accuracy improves from **53.22%** to **56.10%**. This suggests that SIRCL indeed makes the trajectory more compact, but does **not** simply destroy its exploratory capacity. Instead, it preserves useful local exploration while preventing the trajectory from drifting too far away.

We therefore think the fairest interpretation is not that SIRCL "prevents collapse" in a broad sense, but that it enforces a **more structured form of contraction**. The same picture is also consistent with the scaling results: if the regularizer merely caused harmful collapse, longer latent chains should deteriorate further under SIRCL, whereas in fact SIRCL becomes more helpful as the chain grows.

---

## Q2: Why Interpret the Centroid as a Geometric Anchor Rather Than Just a Mean?

We agree that our previous wording around the centroid being the "core problem context" or a semantic anchor was too strong. In the revision, we will use the more careful phrasing that the centroid is a **trajectory-level geometric anchor**: a sample-specific global reference point that constrains the trajectory not to wander too far from its own center of activity.

To support this weaker but better grounded interpretation, we added an offline centroid-replacement intervention on GSM8K (`T=6`, `N=1319`). Replacing each sample's own centroid with a shuffled centroid or a centroid from a different sample substantially increases the token-to-anchor distance. For Sim-CoT, the shuffled/own ratio is **1.84x** without SIRCL and **1.95x** with SIRCL; for CODI it is **1.58x** without SIRCL and **2.94x** with SIRCL. The wrong-sample replacement shows the same pattern. This indicates that the centroid is not an arbitrary average: it carries **sample-specific reference information**, and that reference becomes more discriminative after SIRCL.

We will therefore tone down the semantic interpretation and present the centroid as a **useful geometric reference**, supported by intervention evidence, rather than as a directly interpretable semantic center.

---

## Q3: If the Trajectory Is Already Wrong, Does That Mean the Centroid Is Also Wrong?

We agree that this is a real limitation, but the answer is **not** that SIRCL only helps when the baseline reasoning is already correct. Our evidence is that, at `T=16`, +SIRCL fixes **192/691** baseline-wrong GSM8K samples (**27.79%**). So SIRCL can help even when the original latent reasoning is wrong.

The more precise boundary is this: SIRCL helps when the wrong trajectory is still **recoverable**. In the baseline-wrong subset at `T=16`, the recovered cases have larger radius, diversity, and path length than the still-wrong ones (radius: **9.83 vs. 8.66**; diversity: **14.45 vs. 12.77**; path length: **177.26 vs. 153.61**). Sample **220** is a representative case: the baseline answer is wrong (**80** vs. gold **70**), but the trajectory is still rich (radius **14.75**, path length **326.78**), and +SIRCL corrects it to **70**.

The main failure mode is different: the latent chain contracts too early, so the centroid no longer contains a useful corrective signal. Sample **175** illustrates this clearly: its trajectory shrinks from early radius **11.31** to late radius **3.16**, with mean radius **4.12**, diversity **6.18**, and path length **48.80**, and both baseline and +SIRCL remain wrong (**10** and **0.5** vs. gold **15**). Sample **935** shows the same pattern even more sharply. We will therefore revise the manuscript to state the limitation more precisely: SIRCL does not require the baseline answer to be correct, but it does require the wrong trajectory to remain sufficiently structured.

This is also our conservative interpretation of the Coconut result. In our table, Coconut itself is already below **No-CoT** on several math-oriented benchmarks, including **GSM8K** (**36.4 vs. 39.1**) and **GSM-Hard** (**8.1 vs. 10.8**), which suggests that many Coconut reasoning chains are already off track before SIRCL is applied. This is consistent with its training setup, where the final implicit chain has only **6 latent tokens** (**3 stages x 2 tokens**) and therefore more limited trajectory coherence than a longer end-to-end latent chain. Our point is not that SIRCL is incompatible with Coconut, but that Coconut appears to contain fewer **wrong-but-still-recoverable** trajectories for SIRCL to pull back.

---

## Q4: Why Was There No Matched no-SIRCL Baseline in the Scaling Experiment?

We thank the reviewer for pointing out this gap. We have now added the matched no-SIRCL baseline under the same backbone, dataset, and optimizer setting. The result directly strengthens what should have been our main claim from the start:

| T | no-SIRCL | +SIRCL | Delta |
| --- | ---: | ---: | ---: |
| 6 | 53.22% | 56.10% | +2.88 pp |
| 16 | 44.50% | 58.00% | +13.50 pp |
| 32 | 43.06% | 57.01% | +13.95 pp |

Without SIRCL, performance **degrades sharply** as the latent-token budget grows. With SIRCL, it remains stable and substantially stronger. We will revise the manuscript to make this the central takeaway: **SIRCL's main contribution is stable scaling under longer latent-token chains, rather than uniformly large gains in every short-chain setting.**

To avoid confusion from training-length mismatch, we will also clarify the matched-step comparison in the revision. The conclusion remains the same there as well.

---

## Q5: Why Was the Geometric Analysis Restricted to the 526 All-Correct Samples?

The original motivation for the 526 all-correct subset was to isolate trajectory-shape differences while minimizing confounding from answer correctness. We agree, however, that this analysis alone is not sufficient to show where SIRCL matters most.

We have therefore supplemented it with several additional analyses on the full GSM8K test set. In particular, we now include correctness-stratified geometry statistics, transition-based analysis (`both correct / recovered / both wrong / regressed`), and representative case studies of both recovered and still-wrong samples. These additions let us analyze exactly the cases where SIRCL matters most, rather than only the easy all-correct subset.

The key result is that the geometric differences remain meaningful on failed examples as well. In the baseline model, wrong samples are already more contracted than correct ones (for example, `r_t` mean **11.34** for wrong vs. **13.15** for correct in the T=6 stratified analysis). The samples that SIRCL flips from wrong to correct sit in between: they are not as well-structured as the correct bucket, but they still preserve more usable geometry than the fully unrecoverable failures. We also now include concrete failure cases such as samples **175** and **935**, together with a recoverable contrast case (**220**), to show this boundary at the sample level.

We will therefore keep the all-correct analysis as a controlled view of trajectory shape, but in the revised manuscript we will present these failed-example and correctness-stratified analyses much more prominently so the reader can see where SIRCL changes trajectory behavior in practice.

---

## Q6: The Paper Says "Consistent Performance Gains Across All Evaluated Baselines," but Table 1 Contains Degradations

We agree that this wording is too strong. We will revise it to a more accurate claim: **SIRCL yields mostly positive gains, with some degradations, and its clearest strength is improved stability under longer latent-token budgets.**

This is a more faithful summary of the evidence. In the standard short-chain setting, some gains are modest and a few dataset/baseline combinations do regress slightly. Our rebuttal will no longer frame the method as universally improving every evaluated cell. Instead, we will present the paper's strongest contribution as a **training-time, zero-inference-overhead stabilizer** whose value becomes especially clear when the latent chain becomes longer and more fragile.

---

## Q7: What Is the Intended Takeaway Relative to CoT-SFT?

We will clarify this point in the revised paper. Our claim is **not** that implicit reasoning with SIRCL already universally surpasses strong explicit CoT-SFT in raw accuracy. Rather, the intended takeaway is that SIRCL makes implicit reasoning **more stable, more scalable, and more practically viable** while preserving one of its core advantages: **no extra reasoning-token overhead at inference time**.

We therefore view strong CoT-SFT as an explicit upper-bound reference, not as the target we claim to have fully exceeded. A better framing is that SIRCL helps close part of the gap while improving the stability of the implicit reasoning route, especially when the latent-token budget increases.

---

## Q8: How Should Readers Interpret the Large Difference in the Best Radius $R$ Across Methods?

We agree this needs a clearer explanation. The absolute numerical value of the best radius $R$ should **not** be compared directly across methods, because the native scale of the latent trajectories differs by architecture. Different methods produce different raw distance statistics, so an $R$ that is appropriate for Coconut may be numerically much larger than an $R$ for CODI or SIM-CoT without implying a looser effective constraint.

Our practical procedure is straightforward:

1. **Measure the baseline radius.** We first run the no-SIRCL model and compute the mean per-step distance $\bar{r} = \frac{1}{T}\sum_t d(z_t, \mu)$ over the training set.
2. **Set $R$ relative to this radius.** We use $R \approx 25\%$ of the baseline $\bar{r}$ as the default. For models with especially scattered trajectories, relaxing to $\approx 50\%$ works better.
3. **Calibrate $\lambda$ against the task loss.** We start from $\lambda = 0.01$ and verify that $\lambda \mathcal{L}_{SIRCL}$ is neither dominant nor negligible compared to the task loss.

We will add this guidance to the revised manuscript and make the baseline `r_t` statistics more explicit so that the cross-method difference in optimal $R$ becomes interpretable rather than seeming arbitrary.
