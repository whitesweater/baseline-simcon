# Three-Backbone Summary Report

更新时间：2026-03-31

范围：`llama3-3b` / `qwen3-4b` / `qwen3-1.7b` 三个 backbone，覆盖 `cot-sft` / `simcot` / `simcot+sircl` / `codi` / `codi+sircl` 五种方法。

说明：
- `Best avg` 直接取自各 run 的 `summary/comparison_matrix.csv`。
- 不同方法的 `Best avg` 口径并不完全一致：大多数 implicit run 含 `svamp`，而 `qwen3-1.7b` 的 `codi` / `codi+sircl` 当前是 5 指标平均，所以跨行比较要保守。

## 一句话结论

- `llama3-3b` 仍然是最完整、最稳、最适合写进 rebuttal 主表的 backbone，而且主线 `codi` 新 summary 已补齐，当前 live `Best avg = 39.06%`，是现有 summary 里最高的一条。
- `qwen3-1.7b` 现在不再只有一条可用 implicit 线：`codi = 30.57%` 之外，`simcot` 也已经补成完整 sweep，`Best avg = 26.86%`；但 `simcot+sircl` 仍然接近塌缩，`cot-sft` 则是训练完成但 batch eval 失败。
- `qwen3-4b` 目前依然不是正面故事：`cot-sft` 只有单数据集 GSM8K，可完成的 implicit sweep 只有 `codi+sircl` 一条，而且结果异常低。

## Backbone 总览

| Backbone | 当前可汇报情况 | 当前最强结果 | 当前判断 |
| --- | --- | --- | --- |
| `llama3-3b` | 5/5 方法都有可汇报信息，其中 4 条已有 checkpoint summary，1 条 `gsm8k only` | `codi`，`Best avg = 39.06%` | 当前最完整、最稳的主 backbone |
| `qwen3-1.7b` | 3 条 implicit 线已有多数据集 summary，其中 2 条 `sweep complete`、1 条 `partial sweep`；`cot-sft` 训练完成但 all-eval 失败 | `codi`，`Best avg = 30.57%` | 已有可用补充证据，但方法敏感性很强 |
| `qwen3-4b` | 1 条 `gsm8k only`，1 条异常低的 implicit sweep，1 条只有 checkpoint 未 sweep，剩余 2 条无 checkpoint | `cot-sft`，`GSM8K = 73.62%` | 不适合作为 implicit 主故事 |

## LLaMA-3B 汇总

| 方法 | 当前状态 | 最新/最佳 ckpt | GSM8K | MATH500 | AIME | SVAMP | GSM-HARD | ASDIV | Best avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cot-sft` | 只有 `gsm8k` 可汇报 | `checkpoint_25` | `72.40%` | `-` | `-` | `-` | `-` | `-` | `-` |
| `simcot` | `sweep complete` | `ckpt-53982` | `59.97%` | `6.80%` | `0.00%` | `75.50%` | `14.18%` | `72.41%` | `30.67%` |
| `simcot+sircl` | `sweep complete` | 最新 `ckpt-59980`，最佳 `ckpt-53982` | `63.23%` | `6.40%` | `0.00%` | `71.00%*` | `15.16%` | `72.23%` | `31.55%` |
| `codi` | `sweep complete` | 最新 `ckpt-31992`，最佳 `ckpt-27993` | `60.50%` | `8.60%` | `0.00%` | `78.00%` | `14.71%` | `72.54%` | `39.06%` |
| `codi+sircl` | `sweep complete` | 最新 `ckpt-47984`，最佳 `ckpt-11997` | `41.47%` | `7.40%` | `0.00%` | `64.50%` | `9.48%` | `67.55%` | `31.73%` |

判断：
- `llama3-3b` 现在已经不只是 `simcot` / `simcot+sircl` 完整，主线 `codi` 的 main-root summary 也补齐了，而且当前 live `Best avg` 最高。
- `llama3-3b + codi+sircl` 仍然有 summary，但最佳点已经不在最终 checkpoint，而是在更早的 `ckpt-11997`。
- 2026-03-31 手工补测把 `llama3-3b` 两条 `ckpt-53982` 的 `svamp` 都补齐了：`simcot = 75.50% (151/200)`，`simcot+sircl = 71.00% (142/200)`；两者在该 checkpoint 的 `aime` 都还是 `0.00% (0/30)`。
- 如果 rebuttal 里想强调“当前最完整的一组 live implicit 结果”，`llama3-3b` 现在是最稳的主 backbone。

注：
- `simcot` / `simcot+sircl` 的 `SVAMP` 来自 2026-03-31 对 `ckpt-53982` 的手工补测。
- `simcot+sircl` 行里带 `*` 的 `SVAMP` 与 `Best avg` 对齐到最佳 `ckpt-53982`；同一行其他主指标仍保留当前文档原先展示的最新主 sweep 数字。

## Qwen3-1.7B 汇总

| 方法 | 当前状态 | 最新/最佳 ckpt | GSM8K | MATH500 | AIME | GSM-HARD | ASDIV | Best avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cot-sft` | 训练完成，但 all-eval 失败 | `checkpoint_25` | `-` | `-` | `-` | `-` | `-` | `-` |
| `simcot` | `sweep complete` | 最新 `ckpt-29960`，最佳 `ckpt-5992` | `31.46%` | `10.00%` | `3.33%` | `7.66%` | `61.74%` | `26.86%` |
| `simcot+sircl` | `partial sweep` | 最新 `ckpt-29960`，最佳已评 `ckpt-8988` | `2.35%` | `4.60%` | `0.00%` | `0.91%` | `2.26%` | `2.02%*` |
| `codi` | `sweep complete` | 最新 `ckpt-23968`，最佳 `ckpt-20972` | `49.51%` | `11.00%` | `3.33%` | `13.72%` | `75.27%` | `30.57%` |
| `codi+sircl` | `sweep complete` | 最新 `ckpt-23968`，最佳 `ckpt-8988` | `31.16%` | `10.20%` | `0.00%` | `8.04%` | `67.07%` | `23.29%` |

判断：
- `qwen3-1.7b + simcot` 现在已经是一条真正可汇报的 implicit 线，不再是之前那种只有 4 指标 partial 的旧状态；最佳点在早期 `ckpt-5992`，后期 checkpoint 明显退化。
- `qwen3-1.7b + codi` 仍然是这一 backbone 上最强的结果，`Best avg = 30.57%`。
- `qwen3-1.7b + simcot+sircl` 仍然非常弱，而且目前只覆盖了 5/10 个 checkpoint 的 summary，不能写成完整正面故事。
- `qwen3-1.7b + cot-sft` 虽然 checkpoint 已经到 `25`，但 batch eval 在 `Coconut/dataset.py` 里因为 `answer` 字段类型问题报错，因此目前还没有可汇报的多数据集结果。

注：`simcot+sircl` 行里的 `2.02%*` 来自当前已评 checkpoint 的平均值；因为这条还是 `partial sweep`，不能和完整 sweep 的 `Best avg` 直接横向比较。

## Qwen3-4B 汇总

| 方法 | 当前状态 | 最新/最佳 ckpt | GSM8K | MATH500 | AIME | GSM-HARD | ASDIV | Best avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cot-sft` | 只有 `gsm8k` 可汇报 | `checkpoint_25` | `73.62%` | `-` | `-` | `-` | `-` | `-` |
| `simcot` | 无 checkpoint | `-` | `-` | `-` | `-` | `-` | `-` | `-` |
| `simcot+sircl` | 无 checkpoint | `-` | `-` | `-` | `-` | `-` | `-` | `-` |
| `codi` | 有 checkpoint，未做 sweep | `ckpt-41944` | `-` | `-` | `-` | `-` | `-` | `-` |
| `codi+sircl` | `sweep complete`，但结果异常低 | `ckpt-47936` | `2.43%` | `4.20%` | `0.00%` | `0.91%` | `1.69%` | `1.85%` |

判断：
- `qwen3-4b` 当前只有 `cot-sft` 的单数据集结果是正向可用的。
- implicit 主线里，`simcot` / `simcot+sircl` 还没有实际 checkpoint，`codi` 也还没出 checkpoint sweep。
- `codi+sircl` 虽然有 summary，但数值异常低，因此这一 backbone 目前不适合当作正面补充故事。

## 当前最适合写进 rebuttal 的口径

1. 主 backbone 仍然写 `llama3-3b`。它现在既有完整的 `simcot` / `simcot+sircl`，也有主线 `codi` 的完整 summary，整体最完整。
2. 补充 backbone 最值得写的是 `qwen3-1.7b`，因为它现在至少有两条可用 implicit 线：`codi = 30.57%`、`simcot = 26.86%`。
3. 如果要讲跨 backbone 的 SIRCL 效果，表述要保守，因为 `qwen3-1.7b + simcot+sircl` 仍然很弱，`qwen3-4b` implicit 结果也依旧不正向。

## 对应总表

- `CURRENT_MULTIMODEL_RESULTS_20260329.md`
- `CURRENT_MULTIMODEL_RESULTS_20260329.csv`
- `EXPERIMENT_MASTER_SUMMARY.md`
