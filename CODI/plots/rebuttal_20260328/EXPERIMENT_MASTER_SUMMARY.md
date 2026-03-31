# Experiment Master Summary

更新时间：2026-03-31

## Ready-to-cite

### Main Table

| System | GSM8K | MATH500 | AIME | GSM-Hard | ASDiv | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LLaMA-3B SIM-CoT | 59.97 | 8.80 | 3.33 | 14.25 | 72.41 | Best per-dataset checkpoint from the deeper offline-side multi-dataset sweep |
| LLaMA-3B SIM-CoT + SIRCL | 63.31 | 7.20 | 6.67 | 15.16 | 72.28 | Best per-dataset checkpoint from the rebuttal multi-dataset sweep |
| LLaMA-3B CODI + SIRCL | 52.39 | 7.60 | 0.00 | 12.36 | 72.32 | Best per-dataset checkpoint from the rebuttal multi-dataset sweep |
| Coconut Qwen3-4B CoT-SFT | 73.62 | - | - | - | - | GSM8K only; multi-dataset evaluation attempt is not citeable |

### LLaMA-3B Current Reportable Snapshot

这张表按“当前可直接汇报”的口径整理，保留你最关心的 `latest/best ckpt` 和 `Best avg`。

| 方法 | 当前状态 | 最新/最佳 ckpt | GSM8K | MATH500 | AIME | SVAMP | GSM-HARD | ASDIV | Best avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cot-sft` | 只有 `gsm8k` 可汇报 | `checkpoint_25` | `72.40%` | `-` | `-` | `-` | `-` | `-` | `-` |
| `simcot` | `sweep complete` | `ckpt-53982` | `59.97%` | `6.80%` | `0.00%` | `75.50%` | `14.18%` | `72.41%` | `30.67%` |
| `simcot+sircl` | `sweep complete` | 最新 `ckpt-59980`，最佳 `ckpt-53982` | `63.23%` | `6.40%` | `0.00%` | `71.00%*` | `15.16%` | `72.23%` | `31.55%` |
| `codi` | `sweep complete` | 最新 `ckpt-31992`，最佳 `ckpt-27993` | `60.50%` | `8.60%` | `0.00%` | `78.00%` | `14.71%` | `72.54%` | `39.06%` |
| `codi+sircl` | `sweep complete` | 最新 `ckpt-47984`，最佳 `ckpt-11997` | `41.47%` | `7.40%` | `0.00%` | `64.50%` | `9.48%` | `67.55%` | `31.73%` |

补充：
- 2026-03-31 手工补测已补上 `llama3-3b` 两条 `ckpt-53982` 的 `svamp`：`simcot = 75.50% (151/200)`，`simcot+sircl = 71.00% (142/200)`。
- 同一 checkpoint 上，两者的 `aime` 都仍为 `0.00% (0/30)`；因此这次补测主要是修复 `svamp` 缺口，而不是改变原有 `aime` 结论。
- `simcot+sircl` 行里带 `*` 的 `SVAMP` 与 `Best avg` 对齐到最佳 `ckpt-53982`；同一行其他主指标仍保留当前文档原先展示的最新主 sweep 数字。

### Qwen3-4B Current Reportable Snapshot

| 方法 | 当前状态 | 最新/最佳 ckpt | GSM8K | MATH500 | AIME | GSM-HARD | ASDIV | Best avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cot-sft` | 只有 `gsm8k` 可汇报 | `checkpoint_25` | `73.62%` | `-` | `-` | `-` | `-` | `-` |
| `simcot` | 无 checkpoint | `-` | `-` | `-` | `-` | `-` | `-` | `-` |
| `simcot+sircl` | 无 checkpoint | `-` | `-` | `-` | `-` | `-` | `-` | `-` |
| `codi` | 有 checkpoint，未做 sweep | `ckpt-41944` | `-` | `-` | `-` | `-` | `-` | `-` |
| `codi+sircl` | `sweep complete`，但结果异常低 | `ckpt-47936` | `2.43%` | `4.20%` | `0.00%` | `0.91%` | `1.69%` | `1.85%` |

### Qwen3-1.7B Current Reportable Snapshot

| 方法 | 当前状态 | 最新/最佳 ckpt | GSM8K | MATH500 | AIME | GSM-HARD | ASDIV | Best avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cot-sft` | 训练完成，但 all-eval 失败 | `checkpoint_25` | `-` | `-` | `-` | `-` | `-` | `-` |
| `simcot` | `sweep complete` | 最新 `ckpt-29960`，最佳 `ckpt-5992` | `31.46%` | `10.00%` | `3.33%` | `7.66%` | `61.74%` | `26.86%` |
| `simcot+sircl` | `partial sweep` | 最新 `ckpt-29960`，最佳已评 `ckpt-8988` | `2.35%` | `4.60%` | `0.00%` | `0.91%` | `2.26%` | `2.02%*` |
| `codi` | `sweep complete` | 最新 `ckpt-23968`，最佳 `ckpt-20972` | `49.51%` | `11.00%` | `3.33%` | `13.72%` | `75.27%` | `30.57%` |
| `codi+sircl` | `sweep complete` | 最新 `ckpt-23968`，最佳 `ckpt-8988` | `31.16%` | `10.20%` | `0.00%` | `8.04%` | `67.07%` | `23.29%` |

### Matched Scaling (GSM8K)

| T | no-SIRCL | +SIRCL | Delta |
| --- | ---: | ---: | ---: |
| 6 | 53.22 | 56.10 | +2.88 pp |
| 16 | 44.50 | 58.00 | +13.50 pp |
| 32 | 43.06 | 57.01 | +13.95 pp |

## Partial / do-not-cite

- Qwen3-4B implicit runs are not a positive backbone story right now. The best finished implicit checkpoint sweep we have is `CODI + SIRCL`, with GSM8K 2.43, MATH500 4.20, AIME 0.00, GSM-Hard 0.91, and ASDiv 1.69.
- The Qwen3-4B SIM-CoT + SIRCL run hit OOM early. Logged config: per-device batch 12, grad accum 1, effective global batch 48.
- Coconut Qwen3-4B multi-dataset evaluation is not citeable because it multi-dataset eval attempt failed with EADDRINUSE on port 29500.
- Qwen3-1.7B 现在已有两条可 cite 的 implicit 结果线：`CODI` 的 `Best avg = 30.57%`，以及 `SIM-CoT` 的完整 sweep `Best avg = 26.86%`。但 `SIM-CoT + SIRCL` 仍然只有 partial sweep，而且数值极低；`cot-sft` 虽然训练完成到 `checkpoint_25`，但 batch eval 因数据集字段类型错误失败。

## Current claims we can safely make

- The strongest rebuttal evidence is the matched no-SIRCL scaling comparison: long latent chains degrade sharply without SIRCL and stay stable with SIRCL.
- All geometry-heavy rebuttal analyses are aligned to the paper setting `T=6`; `T=16/32` are used only for scaling.
- The current extra-backbone story should stay conservative: LLaMA-3B is still the strongest and most complete backbone; Qwen3-1.7B now has usable `CODI` (`30.57%`) and `SIM-CoT` (`26.86%`) lines but a very weak `SIM-CoT + SIRCL` partial sweep; and Qwen3-4B implicit runs are not a positive story yet.

## Companion Report

- `THREE_BACKBONE_SUMMARY_20260330.md` consolidates all 15 method-backbone combinations into a single readable report for direct status checks and rebuttal drafting.

## Minimal TODOs

- Qwen3-1.7B `SIM-CoT + SIRCL` still needs finish + remaining checkpoint sweep.
- Qwen3-1.7B `cot-sft` now mainly needs a batch-eval fix, not a from-scratch rerun.
- Qwen3-4B `codi` still needs checkpoint sweep.
- A short `R/λ` selection note based on baseline `r_t` scale and regularizer/task-loss balance.
- Final typo / formatting / missing-citation pass.
