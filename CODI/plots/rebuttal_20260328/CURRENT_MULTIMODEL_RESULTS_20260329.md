# Current Multi-Backbone Results Summary

更新时间：2026-03-31

范围：`cot-sft` / `simcot` / `simcot+sircl` / `codi` / `codi+sircl` 这 5 个方法，配 `llama3-3b` / `qwen3-4b` / `qwen3-1.7b` 这 3 个 backbone。

说明：
- `Source=main` 指当前主线目录 `CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1`
- `Source=offline-side` 指旁支目录 `CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1_simcon_20260327_offline`
- `Source=Coconut` 指 `Coconut/ckpts/*` 下的 CoT-SFT baseline
- 同一个方法/模型若存在多个尝试，这里优先保留“当前最有信息量”的那一条：优先有评测，再看训练进度
- `Best ckpt(avg)` / `Best avg` 只在存在 `comparison_matrix.csv` 时填写
- `Best avg` 直接来自各 run 的 `comparison_matrix.csv`；部分行包含 `svamp`，而 `qwen3-1.7b` 的 `codi` 家族当前只含 5 个主指标，所以跨行平均值只能作粗略比较
- `cot-sft` 里 `llama3-3b` 当前已经补齐 `checkpoint_25` 的 6 个主评测数据集结果；`qwen3-4b` 仍然只有 GSM8K 可以直接汇报；`qwen3-1.7b` 已经训练到 `checkpoint_25`，但 batch multi-dataset eval 失败

## LLaMA-3B 可汇报总表

| 方法 | 当前状态 | 最新/最佳 ckpt | GSM8K | MATH500 | AIME | SVAMP | GSM-HARD | ASDIV | Best avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cot-sft` | `checkpoint_25` 多数据集可汇报 | `checkpoint_25` | `72.40%` | `8.40%` | `0.00%` | `74.40%` | `20.24%` | `2.00%` | `-` |
| `simcot` | `sweep complete` | `ckpt-53982` | `59.97%` | `6.80%` | `0.00%` | `75.50%` | `14.18%` | `72.41%` | `30.67%` |
| `simcot+sircl` | `sweep complete` | 最新 `ckpt-59980`，最佳 `ckpt-53982` | `63.23%` | `6.40%` | `0.00%` | `71.00%*` | `15.16%` | `72.23%` | `31.55%` |
| `codi` | `sweep complete` | 最新 `ckpt-31992`，最佳 `ckpt-27993` | `60.50%` | `8.60%` | `0.00%` | `78.00%` | `14.71%` | `72.54%` | `39.06%` |
| `codi+sircl` | `sweep complete` | 最新 `ckpt-47984`，最佳 `ckpt-11997` | `41.47%` | `7.40%` | `0.00%` | `64.50%` | `9.48%` | `67.55%` | `31.73%` |

注：
- `cot-sft` 行来自 `checkpoint_25` 的 live 产物：`GSM8K` 取 2026-03-29 日志里的 `955/1319 = 72.40%`，`SVAMP` 取 `data/svamp_all.json` 的 live JSON `744/1000 = 74.40%`，不使用旁路 `svamp_test_300_coconut` 的 300 题结果。
- `simcot` / `simcot+sircl` 的 `SVAMP` 来自 2026-03-31 对 `ckpt-53982` 的手工补测。
- `simcot+sircl` 行里带 `*` 的 `SVAMP` 与 `Best avg` 对齐到最佳 `ckpt-53982`；同一行其他主指标仍保留当前文档原先展示的最新主 sweep 数字。

## Qwen3-4B 可汇报总表

| 方法 | 当前状态 | 最新/最佳 ckpt | GSM8K | MATH500 | AIME | GSM-HARD | ASDIV | Best avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cot-sft` | 只有 `gsm8k` 可汇报 | `checkpoint_25` | `73.62%` | `-` | `-` | `-` | `-` | `-` |
| `simcot` | 无 checkpoint | `-` | `-` | `-` | `-` | `-` | `-` | `-` |
| `simcot+sircl` | 无 checkpoint | `-` | `-` | `-` | `-` | `-` | `-` | `-` |
| `codi` | 有 checkpoint，未做 sweep | `ckpt-41944` | `-` | `-` | `-` | `-` | `-` | `-` |
| `codi+sircl` | `sweep complete`，但结果异常低 | `ckpt-47936` | `2.43%` | `4.20%` | `0.00%` | `0.91%` | `1.69%` | `1.85%` |

## Qwen3-1.7B 可汇报总表

| 方法 | 当前状态 | 最新/最佳 ckpt | GSM8K | MATH500 | AIME | GSM-HARD | ASDIV | Best avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cot-sft` | 训练完成，但 all-eval 失败 | `checkpoint_25` | `-` | `-` | `-` | `-` | `-` | `-` |
| `simcot` | `sweep complete` | 最新 `ckpt-29960`，最佳 `ckpt-5992` | `31.46%` | `10.00%` | `3.33%` | `7.66%` | `61.74%` | `26.86%` |
| `simcot+sircl` | `partial sweep` | 最新 `ckpt-29960`，最佳已评 `ckpt-8988` | `2.35%` | `4.60%` | `0.00%` | `0.91%` | `2.26%` | `2.02%*` |
| `codi` | `sweep complete` | 最新 `ckpt-23968`，最佳 `ckpt-20972` | `49.51%` | `11.00%` | `3.33%` | `13.72%` | `75.27%` | `30.57%` |
| `codi+sircl` | `sweep complete` | 最新 `ckpt-23968`，最佳 `ckpt-8988` | `31.16%` | `10.20%` | `0.00%` | `8.04%` | `67.07%` | `23.29%` |

注：`simcot+sircl` 这一行的 `2.02%*` 来自当前已评 checkpoint 的平均值；因为这条还是 `partial sweep`，不能和完整 sweep 的 `Best avg` 直接横向比较。

| Method | Backbone | Source | Progress | Eval | Latest ckpt | GSM8K | MATH500 | AIME | GSM-HARD | ASDIV | Best ckpt(avg) | Best avg | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cot-sft` | `llama3-3b` | `Coconut` | `train complete (checkpoint_25 present)` | `single-ckpt multi-dataset results` | `checkpoint_25` | `72.40%` | `8.40%` | `0.00%` | `20.24%` | `2.00%` | `-` | `-` | `2026-03-29` 的 batch all-eval 先跑出 `GSM8K = 955/1319`，随后在 `gsm_hard` 因数据类型问题中断；`2026-03-31` 已对同一 `checkpoint_25` 补跑 `gsm_hard/svamp/math500/asdiv/aime`，其中 `SVAMP = 74.40% (744/1000)`；当前还没有 live `multi_arith` 产物 |
| `cot-sft` | `qwen3-4b` | `Coconut` | `train complete (checkpoint_25 present)` | `gsm8k only; all-eval failed` | `checkpoint_25` | `73.62%` | `-` | `-` | `-` | `-` | `-` | `-` | 当前可汇报 GSM8K 来自 `checkpoint_24` 单独评测；batch all-eval 启动即撞上 `port 29500 already in use` |
| `simcot` | `llama3-3b` | `offline-side` | `9/10 epoch (53982/59980)` | `sweep complete` | `ckpt-53982` | `59.97%` | `6.80%` | `0.00%` | `14.18%` | `72.41%` | `ckpt-53982` | `30.67%` | 更深 offline-side run 的 9 个 checkpoint 已全部 sweep 完成；当前最佳即 `ckpt-53982`，主线同名 run 仍只到 `ckpt-11996`。2026-03-31 手工补测已补上 `ckpt-53982` 的 `svamp/aime`：`SVAMP = 75.50% (151/200)`，`AIME = 0.00% (0/30)` |
| `simcot` | `qwen3-4b` | `main` | `no checkpoint` | `no sweep` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | 目前只有 `ep_8/ep_10` 目录骨架，没有实际 checkpoint |
| `simcot+sircl` | `llama3-3b` | `main` | `10/10 epoch (59980/59980)` | `sweep complete` | `ckpt-59980` | `63.23%` | `6.40%` | `0.00%` | `15.16%` | `72.23%` | `ckpt-53982` | `31.55%` | 当前 `llama3-3b` 上最强的正向 SIRCL 结果线。2026-03-31 手工补测已补上 `ckpt-53982` 的 `svamp`：`SVAMP = 71.00% (142/200)`；同 checkpoint 的 `AIME = 0.00% (0/30)` 仍来自已有主 sweep |
| `simcot+sircl` | `qwen3-4b` | `main` | `no checkpoint` | `no sweep` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | 目前只有 `ep_8/ep_10` 目录骨架，没有实际 checkpoint |
| `codi` | `llama3-3b` | `main` | `8/8 epoch (31992/31992)` | `sweep complete` | `ckpt-31992` | `60.50%` | `8.60%` | `0.00%` | `14.71%` | `72.54%` | `ckpt-27993` | `39.06%` | main-root `comparison_matrix.csv` 现已补齐；当前 live `Best avg` 最高 |
| `codi` | `qwen3-4b` | `main` | `7/8 epoch (41944/47936)` | `no sweep` | `ckpt-41944` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | 训练接近完成，但还没有对应 checkpoint sweep 结果 |
| `codi+sircl` | `llama3-3b` | `main` | `8/8 epoch (47984/47984)` | `sweep complete` | `ckpt-47984` | `41.47%` | `7.40%` | `0.00%` | `9.48%` | `67.55%` | `ckpt-11997` | `31.73%` | 最佳 `avg` 出现在更早的 `ckpt-11997`；后期 checkpoint 的单项分数更高，但覆盖不完全一致 |
| `codi+sircl` | `qwen3-4b` | `main` | `8/10 epoch (47936/59920)` | `sweep complete` | `ckpt-47936` | `2.43%` | `4.20%` | `0.00%` | `0.91%` | `1.69%` | `ckpt-47936` | `1.85%` | 已有 summary，但数值异常低，当前不适合作为正面 implicit 证据 |
| `cot-sft` | `qwen3-1.7b` | `Coconut` | `train complete (checkpoint_25 present)` | `all-eval failed` | `checkpoint_25` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `checkpoint_1..25` 已存在；batch eval 在 `Coconut/dataset.py:29` 因 `answer` 字段为 `float` 报 `TypeError` |
| `simcot` | `qwen3-1.7b` | `main` | `10/10 epoch (29960/29960)` | `sweep complete` | `ckpt-29960` | `31.46%` | `10.00%` | `3.33%` | `7.66%` | `61.74%` | `ckpt-5992` | `26.86%` | 现在已经是完整 10-checkpoint summary；最佳点在早期 `ckpt-5992`，后期 checkpoint 明显塌缩 |
| `simcot+sircl` | `qwen3-1.7b` | `main` | `10/10 epoch (29960/29960)` | `partial sweep` | `ckpt-29960(train)` | `2.35%` | `4.60%` | `0.00%` | `0.91%` | `2.26%` | `ckpt-8988` | `2.02%*` | 当前 summary 只覆盖 5/10 个 checkpoint，整体仍然接近塌缩 |
| `codi` | `qwen3-1.7b` | `main` | `8/8 epoch (23968/23968)` | `sweep complete` | `ckpt-23968` | `49.51%` | `11.00%` | `3.33%` | `13.72%` | `75.27%` | `ckpt-20972` | `30.57%` | `qwen3-1.7b` 当前最强可汇报组合；最佳点在 `ckpt-20972`，最终 `ckpt-23968` 略低 |
| `codi+sircl` | `qwen3-1.7b` | `main` | `8/8 epoch (23968/23968)` | `sweep complete` | `ckpt-23968` | `31.16%` | `10.20%` | `0.00%` | `8.04%` | `67.07%` | `ckpt-8988` | `23.29%` | 已完成 checkpoint sweep；最优出现在较早的 `ckpt-8988`，后续 checkpoint 明显退化 |

简短结论：
- 当前 live summary 里 `Best avg` 最高的是 `llama3-3b + codi = 39.06%`
- 如果只看 `SIRCL` 的正向主线，当前最稳的是 `llama3-3b + simcot+sircl = 31.55%`
- 2026-03-31 手工补测把 `llama3-3b` 两条 `ckpt-53982` 的 `svamp` 都补齐了：`simcot = 75.50% (151/200)`，`simcot+sircl = 71.00% (142/200)`；两者在该 checkpoint 的 `aime` 都还是 `0.00% (0/30)`
- 2026-03-31 的 Coconut 补测也把 `llama3-3b + cot-sft` 从 `gsm8k only` 更新成了单 checkpoint 的多数据集结果线：`GSM8K = 72.40%`、`SVAMP = 74.40%`、`GSM-HARD = 20.24%`、`MATH500 = 8.40%`、`ASDIV = 2.00%`、`AIME = 0.00%`
- `qwen3-1.7b` 现在已经不止一条 usable implicit 线：`codi = 30.57%`，`simcot = 26.86%`
- `qwen3-1.7b + simcot+sircl` 依然很弱，而 `qwen3-1.7b + cot-sft` 训练虽完成，但还没有 citeable multi-dataset eval
- `qwen3-4b` 当前只有 `cot-sft` 的 GSM8K 可以正向引用，implicit 结果还不适合作为正面故事

对应 CSV 总表：`CURRENT_MULTIMODEL_RESULTS_20260329.csv`
