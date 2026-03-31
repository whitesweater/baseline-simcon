# Response to Reviewer RWVq

We sincerely thank the reviewer for the careful and constructive feedback. The questions helped us sharpen both the empirical claims and the intended scope of the paper. Below we revise our responses to be more precise and better aligned with the evidence we currently have.

---

## Q1: Direct Evidence on "Preventing Collapse" vs. Simply Shrinking Trajectories

We agree with the reviewer that the current hinge loss more directly penalizes **excessive deviation from the trajectory center** than it directly enforces "anti-collapse" in a generic sense. We will therefore revise the manuscript to make a more precise claim: **SIRCL does not prevent all contraction; rather, it replaces unconstrained drift with structured geometric contraction that is empirically more useful for reasoning.**

To directly examine this point, we added three complementary collapse diagnostics on the standard GSM8K setting (`T=6`, `N=300`, 100x bootstrap): effective rank, RandomSim/anisotropy, and within-trajectory diversity. For Sim-CoT, SIRCL improves accuracy from **53.22%** to **56.10%**, while the latent trajectories also become more compact (effective rank: **4.78 -> 4.26**; RandomSim: **0.424 -> 0.450**; diversity L2: **18.60 -> 12.67**; radius mean: **12.27 -> 8.30**). We observe the same qualitative pattern for CODI (**52.92% -> 55.72%**, with lower effective rank and diversity as well). So the data do show stronger contraction, but importantly this contraction comes with **better** task performance rather than degradation.

We therefore think the fairest interpretation is not "SIRCL prevents collapse" in the sense of maximizing rank or preserving all diversity, but rather that it induces a **beneficial, within-sample geometric regularization**. This is also consistent with the scaling results: if the regularizer merely caused harmful collapse, longer latent chains should deteriorate further under SIRCL, whereas in fact SIRCL becomes more helpful as the chain grows.

---

## Q2: Why Interpret the Centroid as a Geometric Anchor Rather Than Just a Mean?

We agree that our previous wording around the centroid being the "core problem context" or a semantic anchor was too strong. In the revision, we will use the more careful phrasing that the centroid is a **trajectory-level geometric anchor**: a sample-specific global reference point that constrains the trajectory not to wander too far from its own center of activity.

To support this weaker but better grounded interpretation, we added an offline centroid-replacement intervention on GSM8K (`T=6`, `N=1319`). Replacing each sample's own centroid with a shuffled centroid or a centroid from a different sample substantially increases the token-to-anchor distance. For Sim-CoT, the shuffled/own ratio is **1.84x** without SIRCL and **1.95x** with SIRCL; for CODI it is **1.58x** without SIRCL and **2.94x** with SIRCL. The wrong-sample replacement shows the same pattern. This indicates that the centroid is not an arbitrary average: it carries **sample-specific reference information**, and that reference becomes more discriminative after SIRCL.

We will therefore tone down the semantic interpretation and present the centroid as a **useful geometric reference**, supported by intervention evidence, rather than as a directly interpretable semantic center.

---

## Q3: If the Trajectory Is Already Wrong, Does That Mean the Centroid Is Also Wrong?

We agree that this is a real limitation, and we will make it explicit in the revised manuscript. The more precise boundary suggested by our analysis is not that "SIRCL only helps when the baseline is already correct," but that **SIRCL helps when the baseline trajectory still retains enough recoverable geometric structure**.

In our paired sample-level analysis on GSM8K (`T=6`), SIRCL recovers **119** of the **612** baseline-wrong samples, i.e. **19.44%** of them. At the same time, baseline-wrong samples are already more contracted than baseline-correct ones (radius mean: **11.37 vs. 13.11**; token diversity: **18.20 vs. 20.71**). Within the baseline-wrong subset, the recovered cases remain slightly larger and more diverse than the still-wrong ones (radius: **11.77 vs. 11.54**; diversity: **18.73 vs. 18.43**). This suggests that SIRCL can still help on wrong trajectories, but mainly when they have not yet lost too much useful structure.

We will therefore present this as a **failure-mode boundary**: once the latent trajectory has already contracted too early or become too weakly structured, the centroid can no longer provide a strong corrective signal. We will also connect this more carefully to the Coconut result. In our actual Coconut setup, reasoning is introduced through a stage-wise replacement curriculum that moves from fully explicit segments to partially latent stages and finally to **3 latent stages with 2 latent tokens per stage**, i.e. a fully implicit chain of only **6 latent tokens**. This makes full-trajectory coherence more limited than in a longer end-to-end latent chain. Our interpretation is therefore not that SIRCL is intrinsically incompatible with Coconut, but that **settings with more limited trajectory coherence leave less recoverable structure for centroid-based regularization to exploit**.

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

We therefore added a correctness-stratified analysis on the full GSM8K test set. The key result is that the geometric differences remain meaningful on failed examples as well. In the baseline model, wrong samples are already more contracted than correct ones (for example, `r_t` mean **11.34** for wrong vs. **13.15** for correct in the T=6 stratified analysis). The samples that SIRCL flips from wrong to correct sit in between: they are not as well-structured as the correct bucket, but they still preserve more usable geometry than the fully unrecoverable failures. After applying SIRCL, the different groups are pulled into a much narrower geometric band.

We will therefore keep the all-correct analysis for controlled visualization, but we will also add the failed-example and correctness-stratified results much more prominently so the reader can see where SIRCL changes trajectory behavior in practice.

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

In the revision, we will explain the tuning rule more clearly: the right reference is each method's **baseline trajectory radius distribution**, not the raw number itself. Concretely, one can first measure the native `r_t` scale of the baseline model, then choose $R$ relative to that scale, e.g. near a high quantile such as `P90(r_t)` or a fixed fraction of the baseline radius, and finally tune the regularization weight so it does not dominate the task loss.

We will also report the baseline `r_t` statistics more explicitly so that the cross-method difference in optimal $R$ becomes interpretable rather than seeming arbitrary.
