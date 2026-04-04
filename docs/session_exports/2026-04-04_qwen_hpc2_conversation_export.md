# 2026-04-04 Qwen HPC2 对话导出

## 主题

本次对话围绕 CODI 项目在 HKUST-GZ HPC2 上运行 Qwen3 LoRA / SIM-CoT 实验展开，重点包括环境修复、离线数据准备、Slurm 提交与监控、OOM 自动恢复、已完成训练结果分析，以及 Qwen3-1.7B 低性能的根因排查。

## 用户核心目标

1. 让 Qwen3 相关训练和评测在 HPC2 上稳定跑通。
2. 尽量通过环境、路径、缓存和脚本层面修复问题，避免不必要的源码改动。
3. 建立可持续的监控和自动 OOM 重提交流程。
4. 分析 Qwen3-1.7B SIM-CoT 训练后结果异常偏低的原因。
5. 验证原始 base Qwen3-1.7B 本身是否已经有明显问题。
6. 对比 Qwen、GPT-2、LLaMA 的 tokenizer 行为，检查 explain supervision marker 是否错配。

## 关键背景

- 工作区根目录是 baseline，而不是 CODI。
- 当前有效实验输出根目录是 CODI_rebuttal_runs/rebuttal_20260325。
- 主要代码入口在 CODI/train.py、CODI/src/model.py、CODI/test.py、CODI/test_baseline.py。
- 当前阶段以 rebuttal / revision 实验为主，重点是 GSM8K、Math500、AIME 等数学推理任务。

## 已完成的环境与运行修复

### 1. 配置与超参修正

- 修正了部分训练脚本中的学习率设置，最终统一到当前 Qwen3 common 脚本使用的 0.0005。
- 修正了 config.env 中 GPU 数量的条件赋值逻辑。
- 将 SIRCL_RADIUS_THRESHOLD 从 2 提升到 4。

### 2. 离线数据与缓存修复

- GSM8K-Aug 的 Hugging Face 缓存不完整，导致离线 compute node 无法直接 load_dataset。
- 重新下载并整理了 GSM8K-Aug 的 train/test 原始 JSON。
- 将数据转换为本地可加载的 DatasetDict，并补齐 HPC2 离线运行需要的缓存。
- prepare_assets.py 后续也补上了优先走本地路径的逻辑，避免 compute node 再访问外网。

### 3. Python / Torch 运行环境修复

- 早期环境中的 PyTorch 2.11.0+cu130 与节点驱动不兼容。
- 通过重新同步环境，实际使用版本回到 PyTorch 2.9.1+cu128。
- 这一步解决了训练无法在目标节点上正常启动的问题。

### 4. 训练缓存路径修复

- train.py 中原先有一个硬编码 tokenized cache 路径。
- 保留的源码修改只有这一项：改为优先使用 CODI_CACHE_DIR / CODI_TOKENIZED_CACHE_DIR 驱动缓存目录。
- 这符合“优先修环境，不大改代码”的要求。

## Slurm 与训练运行阶段

### 1. 提交与跑通

- 在修复离线数据和环境问题后，Qwen3 相关生产训练任务已成功启动。
- 训练主要运行在 HPC2 的 A800 相关分区上，调试任务使用 debug 分区。

### 2. 资源调优

- 逐步尝试了不同 per-device batch size 和 GPU 数量组合。
- 一度尝试过 8 GPU 和 7 GPU 配置，但受分区和资源约束影响，最终回到更现实的配置。
- 为避免频繁 OOM，中途多次基于显存和作业状态做了调整。

## OOM Watchdog

### 1. 初版实现

- 用户要求增加 OOM watchdog，自动检测作业 OOM 并重提交流程。
- 初版使用 nohup 运行，后续发现会话持久性不理想。

### 2. 稳定版实现

- Watchdog 后续迁移到 tmux 中运行，避免因 shell 退出而中断。
- 加入了更稳妥的重提交流程：先确认旧作业完全退出，再以更小 batch size 重提。
- 已观察到多次实际自动重提交流程生效，例如从 batch 24 自动降到 20 并重新提交。

## 已完成结果分析

### 1. 完成作业 9660433 的排查

- 对 qwen3_1p7b simcon 的已完成训练作业进行了日志、结果表、checkpoint sweep 和 TensorBoard 事件文件检查。
- 结论是训练 loss 的下降趋势正常，训练本身看起来不是明显数值发散。
- 但最终 GSM8K 等评测结果明显异常偏低，和预期不符。

### 2. TensorBoard

- 试图启动 TensorBoard 查看训练曲线。
- 6006 端口被占用，后续尝试了其他端口，但当时没有完整继续展开到可用的对外访问状态。

## 低性能根因排查

### 1. train.py 和 model.py 逻辑对比

- 对比了 Qwen 与 LLaMA 的训练数据预处理、特殊 token、answer prompt、LoRA target module 和 explain supervision 相关逻辑。
- train.py 层面的 source / cot / answer 拼接逻辑总体一致。
- 关键差异主要落在模型家族差异、tokenizer 行为、Qwen fallback 路径，以及 model.py 中不同模型分支对 latent / decoder 的承接方式。

### 2. Base Qwen3-1.7B 独立评测

- 新建了一个 debug Slurm 脚本，专门用 test_baseline.py 评测原始 base Qwen3-1.7B 在 GSM8K 上的表现。
- 第一次提交时，因为 Slurm spool 环境下 BASH_SOURCE 路径解析错误，脚本试图往 /var/spool/slurmd 下写日志，导致失败。
- 后续改成绝对路径后重新提交，评测成功。
- 结果表明 raw base Qwen3-1.7B 在 GSM8K test 上达到 21.99%。

### 3. 这一步的意义

- 该结果说明 base model 和最基础的 eval path 并没有彻底坏掉。
- 因此问题更可能出现在 CODI / SIM-CoT 训练路径本身，而不是 Qwen3-1.7B 这个底模完全不可用。

## GPT-2 与 Qwen 对比分析

- 用户指出 GPT-2 旧脚本可以工作，要求解释为什么 Qwen 训练结果反而异常。
- 对比了旧脚本 CODI/scripts/train_gpt2_gsm8k-aug-decoder-2.sh 与当前 Qwen3 通用训练脚本。
- 初步结论包括：
  - GPT-2 规模更小，训练路径更简单。
  - GPT-2 脚本没有像当前 Qwen3 脚本那样显式放大 distill_loss_factor 到 20。
  - Qwen3 当前训练路径把更多监督项和 latent / decoder 机制叠加在一起，更容易把最终 answer generation 学偏。

## Tokenizer 对比结论

本次对话最后阶段，直接加载了 Qwen3-1.7B、LLaMA 1B、GPT-2 的 tokenizer，对 explain supervision 和答案提示相关字符串做了实测对比。

### 实测对象

- <<
- 空格加 <<
- >>
- The answer is:
- The answer is: 42
- The next step result is:
- The next step result is: 42

### 实测结论

- 三个 tokenizer 都能把 <<、空格加 <<、>> 识别为单 token。
- 三个 tokenizer 对 The answer is: 和 The next step result is: 的切分结构基本一致。
- 差异主要体现在具体数字上：
  - Qwen 会把 42 切成空格、4、2。
  - LLaMA 会把 42 作为一个 token。
  - GPT-2 会把空格加 42 作为一个 token。
- 由于训练中 answer prompt 的定位只匹配到冒号为止，不包含具体数字，这个差异不构成 explain marker 根本错配的直接证据。

### 当前判断

- << / >> marker 的 tokenizer 行为目前看起来不是 Qwen3 训练异常的主因。
- 更可疑的仍然是多 loss 联合训练下，Qwen 在 latent / decoder / distill / ref_ce / explain 之间的优化目标偏移。

## 代码与脚本层面的关键产物

- 保留的 train.py 修改：tokenized cache 路径改为基于环境变量。
- prepare_assets.py 增加本地数据路径优先逻辑。
- 新增了 raw base Qwen3-1.7B GSM8K debug 评测脚本。
- 新增并迭代了 OOM watchdog 脚本，最终以 tmux 持续运行。

## 当前状态

1. HPC2 上的 Qwen3 训练和评测链路已基本跑通。
2. 离线数据、Torch 兼容性、缓存路径、Slurm 提交和 OOM 自动恢复都已具备可复用流程。
3. 已确认 raw base Qwen3-1.7B 本身并未彻底失效。
4. Qwen3-1.7B SIM-CoT 训练后效果异常偏低的问题仍未最终定位。
5. tokenizer marker 错配这一假设当前证据不足。

## 建议的后续排查方向

1. 在少量样本上打印 ref_answer_position、model_answer_position、decoded explain span，确认监督落点在 Qwen 上没有语义错位。
2. 直接比较 ce_loss、distill_loss、ref_ce_loss、explain_loss 的量级和主导关系，确认是否是 loss 组合把模型推离最终答案学习目标。
3. 检查新增 bot / eot token 及 latent bridge 在 Qwen 上的嵌入初始化和训练动态，看看是否比 LLaMA 更容易破坏生成能力。
4. 如需继续做实验对照，可考虑保持实验定义不变的前提下，做更小规模的局部诊断 run，而不是立即大规模重训。

## 备注

- 用户在整个过程中明确偏好“优先修环境、路径、缓存、脚本”，不希望为了绕过运行问题而大改核心代码。
- 用户对可复用的 HPC2 运行流程、自动恢复机制和诊断证据链要求较高，因此后续所有改动都应尽量保持可验证和可回溯。