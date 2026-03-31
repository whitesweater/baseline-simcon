## Response to Reviewer

We thank the reviewer and address the concerns below.

---

### W1: Heuristic nature of the method

We thank the reviewer for this thoughtful critique. As discussed in Section 5.1, the "zig-zag" oscillation in latent trajectories is not merely noise — Anderson (2026) shows it is a **universal property of Transformer residual-stream dynamics**, reflecting the model's internal exploratory computation across layers.

SIRCL is designed to **preserve this functional oscillation** while constraining only excessive drift. Its hinge-style trust region applies zero gradient within radius $R$, so latent states remain free to oscillate locally; only deviations beyond the trust region are penalized. This distinguishes SIRCL from a naïve "straightening" approach.

To verify this, we tested the **Geodesic Deviation Loss** ($\mathcal{L}_{GDL}$, Eq. 7) in Section 5.1, which forces each latent state toward the geodesic (straight line) between start and end states. This suppresses oscillation entirely — and performs worse: $\mathcal{L}_{GDL}$ reaches only **53.37%** on GSM8K, compared to SIRCL's **56.10%** (baseline: **53.22%**). The result confirms that some oscillation is beneficial, and SIRCL's advantage lies in bounding drift without eliminating useful exploration.

---

### Q1: When and why does SIRCL succeed or fail?

We thank the reviewer for this insightful question. We first summarize the matched Table 1 results to clarify the overall pattern:

| Method | GSM8K | CommonsenseQA | SVAMP | GSM-Hard | ASDiv |
| --- | --- | --- | --- | --- | --- |
| CODI | 52.9 | 70.3 | 59.9 | 12.1 | 65.4 |
| CODI+SIM-CoT("SIM-CoT" in Table 1) | 53.2 | 71.0 | 60.2 | 12.4 | 64.5 |
| CODI+SIRCL | 55.3 | 70.8 | 60.4 | 13.3 | 64.1 |
| CODI+SIM-CoT+SIRCL | **56.1** | **71.8** | **60.6** | 12.7 | **65.5** |

Two observations emerge. First, compared with CODI+SIM-CoT, CODI+SIRCL improves on GSM8K (55.3 vs. 53.2), SVAMP (60.4 vs. 60.2), and GSM-Hard (13.3 vs. 12.4), with marginal decreases on CommonsenseQA (70.8 vs. 71.0) and ASDiv (64.1 vs. 64.5). Second, adding SIRCL on top of SIM-CoT improves all five columns. Overall, SIRCL provides an effective training-time regularization that is complementary to SIM-CoT, while introducing no additional inference cost and only minor training overhead.

**When does SIRCL fail?** The most consistent failure mode we observe is **premature trajectory contraction**: later latent states collapse to very low radius, diversity, and path length, so the centroid no longer summarizes an active reasoning process but merely marks the center of an already impoverished trajectory. In that regime, geometric regularization can stabilize the path but cannot recover reasoning information that has already been lost. Sample 175 illustrates this clearly: its trajectory shrinks from early radius 11.31 to late radius 3.16, with mean radius 4.12, diversity 6.18, and path length 48.80, and both baseline and +SIRCL remain wrong (10 and 0.5 vs. gold 15). Sample 935 shows the same pattern even more sharply (11.24 -> 2.18, mean radius 4.02, diversity 6.21, path length 54.31).


**Case contrast.** Sample 220 illustrates the recoverable regime: baseline predicts 80 instead of the gold 70, with radius 14.75 and path length 326.78; after SIRCL, the answer becomes 70 while the trajectory contracts to radius 5.64 and path length 81.93. In other words, SIRCL helps when a rich trajectory drifts but still preserves enough structure to be pulled back. It fails when the trajectory has already become too compressed to contain useful corrective signal.



---

### Q2: Scaling beyond T=32 — where are the limits?

We thank the reviewer for raising this question. Performance peaks at **T=16** and declines at **T=32**, so we did not go further. We added matched comparisons with and without SIRCL under the same backbone (SIM-CoT, LLaMA-1B, GSM8K) and optimizer:

| T | w/o SIRCL | +SIRCL | Δ |
| --- | ---: | ---: | ---: |
| 6 | 53.22% | 56.10% | +2.88 pp |
| 16 | 44.50% | 58.00% | **+13.50 pp** |
| 32 | 43.06% | 57.01% | **+13.95 pp** |

**Without SIRCL, performance degrades monotonically** (53.22→44.50→43.06, cumulative −10.16 pp), confirming accumulated drift. **With SIRCL, performance remains stable** (56.10→58.00→57.01, +0.91 pp from T=6). SIRCL's gain amplifies with T: +2.88 pp at T=6 to ~+14 pp at T=16–32, consistent with the geometric intuition — longer chains accumulate more correctable drift.

The plateau from T=16 to T=32 suggests T≈16–32 approaches the effective reasoning capacity ceiling for 1B-scale models. The modest Table 1 gains at T=6 are themselves consistent with our theory: short chains have little drift to correct. **SIRCL's primary contribution is enabling stable scaling** rather than boosting short-chain performance.
