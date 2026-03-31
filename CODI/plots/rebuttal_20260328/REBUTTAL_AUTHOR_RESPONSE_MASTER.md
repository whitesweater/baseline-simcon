# Rebuttal Author Response Master

更新时间：2026-03-29

本稿是给作者内部使用的 reviewer-by-reviewer 工作底稿，只使用当前已经 ready-to-cite 的结果。凡是尚未完成、或当前不适合写进 rebuttal 的内容，都显式标成 `[TODO]`，避免把讨论引到我们来不及补完的大实验上。

## Reviewer RWVq

### 当前主答复口径

- 我们最强的新证据是 matched no-SIRCL scaling 对照。`T=6/16/32` 下，no-SIRCL 在 GSM8K 上从 `53.22 -> 44.50 -> 43.06` 持续退化，而 `+SIRCL` 为 `56.10 -> 58.00 -> 57.01`，对应增益分别为 `+2.88pp / +13.50pp / +13.95pp`。这直接支撑“长 latent budget 下的 stability / scalability / zero inference overhead”。
- 对 collapse 的回应应改成更精确的表述：SIRCL 更直接抑制的是 drift / outliers，而不是宣称阻止所有收缩。我们已有的 T=6 分析表明，EffRank 下降但 accuracy 同时上升；因此更合理的解释是“有结构的几何收缩替代无约束漂移”，而不是简单的有害 collapse。
- centroid 的表述要降级为 `trajectory-level geometric reference`。当前最直接的证据是 replacement ratio：own centroid 被 shuffled / wrong centroid 替换后，token-anchor 距离上升约 `1.84x-3.03x`，且 `+SIRCL` 后区分度更大。
- 方法边界可以明确承认：当 baseline trajectory 早期已经失去几何展开时，centroid 也可能总结错误轨迹。我们已有样本转移统计支持这个边界刻画：`recovered=119 > regressed=80`，说明方法不是“只对 baseline 已答对样本才有用”，但它确实更依赖原始轨迹还保留可恢复结构。
- “只看 all-correct”的质疑已经可以正面回应。现有 T=6 分层分析已经包含 `correct / wrong / all_correct / all_wrong / sircl_flips / sircl_regress`，可以直接写进 rebuttal。
- 两处文案必须收紧：一是把过强的“全部 baseline 都提升”改成 `mostly positive gains with some degradations`；二是把与 CoT-SFT 的 narrative 改成“我们强调稳定性、可扩展性和推理零开销，而不是宣称隐式推理已经全面超过强显式 CoT-SFT”。

### 可以直接粘贴的短版答复

我们已补齐训练期 scaling 的 matched no-SIRCL 对照。在相同 backbone / 数据 / 优化设置下，GSM8K 上 no-SIRCL 随 latent budget 增大持续退化（`53.22 -> 44.50 -> 43.06`），而 `+SIRCL` 保持稳定并提升到 `56.10 / 58.00 / 57.01`。因此，我们会把 strongest claim 明确收敛为：SIRCL 的主要价值在于长 latent-token chain 下的稳定扩展，而不是短链设置下的普遍大幅提分。

对 drift vs collapse 的表述我们也会更精确：SIRCL 更直接抑制的是无约束漂移。我们在 T=6 的新分析中确实观察到 trajectory 更紧凑，但这伴随 accuracy 提升，而不是退化，因此更合理的解释是“有结构的几何收缩替代无约束漂移”。相应地，我们会把 centroid 从语义化表述收敛为 `trajectory-level geometric reference`；当前最直接的证据是 replacement ratio 在 shuffled / wrong centroid 替换下显著恶化，并且 `+SIRCL` 后差异进一步放大。

### 当前只保留为 TODO 的点

- `[TODO]` 若 reviewer 继续追问更细的 `R/λ` 规则，我们补一段短的方法说明：先统计 baseline `r_t` 分布，再让 `R` 落在该分布的高分位附近，并调 `λ` 使正则不主导主任务损失。

## Reviewer mBLT

### 当前主答复口径

- 直接接受“heuristic but practical”的定位，不去硬讲从 Transformer 原理推导出来的理论。
- 标准设置下增益偏小是可以解释的，因为短链 `T=6` 下本来 drift 累积就少，所以 SIRCL 的修正空间有限；这个解释被 matched scaling 强力支持，因为增益在 `T=16/32` 急剧扩大。
- 不承诺 `T=48/64`。目前最稳的写法是：`T=32` 已经足够证明“无 SIRCL 会退化，而 +SIRCL 能显著扩大稳定区间”；更大的 `T` 可以作为讨论期或最终版补充，但不放进当前主线。

### 可以直接粘贴的短版答复

我们同意 SIRCL 更适合被定位为一个极简、训练期-only 的稳定化正则，而不是从第一性原理推导的机制性理论。当前最重要的新证据是 matched scaling 对照：短链 `T=6` 下增益较温和，但在更长 chain 下，无 SIRCL 明显退化，而 `+SIRCL` 维持稳定并带来超过 `13pp` 的增益。这说明 SIRCL 的贡献主要体现在 budget 扩展场景，而这也正是 implicit reasoning 当前最脆弱的部分。

### 当前只保留为 TODO 的点

- `[TODO]` 若 reviewer 明确要求更大 `T` 的单点实验，我们只写成进行中，不把 `T=48/64` 放进当前 rebuttal 主文。

## Reviewer QHNB

### 当前主答复口径

- 单 backbone 限制要承认，但也要把当前已有的 stronger evidence 说清楚：我们已经有 LLaMA-3B 上完整的多数据集 ready-to-cite 结果，以及 matched scaling 对照。
- 更难 benchmark 不能说“没有”。当前 multi-dataset sweep 已包含 `math500` 和 `aime`，虽然绝对分数不高，但足以说明我们已经开始在更难题目上检验，不是只停留在 GSM / SVAMP。
- 额外 backbone 只写成 `Qwen3-1.7B ongoing`。不把当前 Qwen3-4B implicit 结果写成正面泛化证据，因为它现在并不支持这个 narrative。
- 对 “为什么低于 SIM-CoT 报告值” 的回答要走对齐口径：先强调设置差异，再把我们的主张收敛到“在统一训练设定内，SIRCL 对稳定 scaling 的价值是清楚的”，而不是强行横向比较不同论文里未对齐的数值。

### 可以直接粘贴的短版答复

我们同意当前版本在 backbone 覆盖上仍偏窄，因此在 rebuttal 中不会把“广泛跨 backbone 泛化”作为已完成结论。当前已经 ready-to-cite 的额外证据主要有两类：一是 LLaMA-3B 上的多数据集结果，二是 matched no-SIRCL scaling 对照。对于更难 benchmark，我们当前 sweep 已包含 `math500` 和 `aime`；虽然这些题目上的绝对准确率仍然有限，但它们已能帮助我们说明 SIRCL 的贡献更偏向稳定性和扩展性，而不是声称已经全面超过强显式推理。

关于与已有 SIM-CoT 报告值的对比，我们会更谨慎地写成“需要在统一 backbone / 数据处理 / 训练细节下做 apples-to-apples 对齐”。本轮 rebuttal 里，我们更愿意强调当前最干净、最直接的新证据：在统一设置下，no-SIRCL 会随 latent budget 增大显著退化，而 `+SIRCL` 则显著扩大稳定区间。

### 当前只保留为 TODO 的点

- `[TODO]` `Qwen3-1.7B` 的 `SIM-CoT / CODI` T=6 主结果。
- `[TODO]` typo / formatting / missing citation 的最终修订清单。

## Reviewer HNnW

### 当前主答复口径

- 引用补充、center-loss 关系、选参指南，这三件事都可以在当前写作层面先处理，不依赖新实验。
- 线性参考轨迹约束不放进当前 rebuttal 主线。最稳的做法是承认这是合理对照，但不在这轮回复里承诺重跑。
- qualitative appendix 也是最终版增强项，不放进 rebuttal 成败关键路径。

### 可以直接粘贴的短版答复

我们感谢 reviewer 对写作和分析完整度的认可。最终版本中，我们会补充相关并行工作，并更明确区分 SIRCL 与经典 center-loss 式思路的差异：SIRCL 面向的是 sample-specific latent trajectory 的稳定组织，约束形式是 hinge-style trust region，而不是持续把表示拉向一个类别中心。我们也会加入一段简洁的选参说明：先在 baseline 上统计 `r_t` 的典型尺度，再让 `R` 落在其高分位附近，并将 `λ` 调到“不会主导主任务损失”的范围内。

对“是否应围绕线性参考轨迹而不是 centroid”这一点，我们会在最终版中把它写成一个明确的替代设计方向，而不是在当前 rebuttal 中作超出已完成证据的承诺。qualitative 对比同理，更适合作为最终版本 appendix 的增强内容。

### 当前只保留为 TODO 的点

- `[TODO]` missing citations 的最终整理和插入位置。
- `[TODO]` appendix qualitative figure selection。

## 当前统一写作约束

- 不再使用带强语义锚定色彩的表述，统一替换为 `trajectory-level geometric reference`。
- 不再使用“所有 baseline 全都提升”的强表述，统一改成 `mostly positive gains with some degradations`。
- 相对 CoT-SFT 的叙事统一为：`stability / scalability / zero inference overhead`，而不是准确率压制。
