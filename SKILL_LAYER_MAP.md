# Skill Layer Map

这份文档不是新的 source of truth。

它的作用只有一个：

把当前 `baseline/` 仓库、HPC2 访问方式、迁移与同步流程、以及这次真实排障过程中已经形成的知识，整理成一组可复用的 skill 分层地图，方便以后：

- 换项目时复用抽象能力
- 换 AI 助手时快速迁移上下文
- 避免重复写已经存在的 SOP

如果某一层已经有可靠文档或 skill，这里只做索引，不重复抄写细节。

---

## 1. 先复用什么，不要重复造轮子

### 项目级 source of truth

- [PROJECT_GUIDE.md](./PROJECT_GUIDE.md)
- [NEWCOMER_HANDOVER.md](./NEWCOMER_HANDOVER.md)
- [CODI/README.md](./CODI/README.md)
- [CODI/train_on_gsm8k_dataset/](./CODI/train_on_gsm8k_dataset)
- [scripts/migrate_baseline_minimal.sh](./scripts/migrate_baseline_minimal.sh)
- [scripts/migrate_baseline_hpc2_longterm.sh](./scripts/migrate_baseline_hpc2_longterm.sh)

### 已有可直接复用的 skills

- `hpc-login-ssh`
- `hpc2-gpu-runtime`
- `hpc-migration`
- `hpc-sbatch`
- `baseline-rebuttal-prepare`
- `baseline-hpc2-codi-runtime`
- `hpc2-codi-train-triage`
- `repo-source-of-truth-nav`
- `remote-runtime-topology`
- `project-sync-rsync-layered`
- `runtime-env-rebuild-verify`
- `experiment-contract-guard`
- `train-monitor-eta`

结论：

- 仓库认知、HPC2 接入、迁移、训练运行、训练排障，这五大块已经不是空白能力。
- 后面应优先做“抽象层重组”和“跨项目可复用封装”，而不是重复写 baseline 专属步骤。

---

## 2. 推荐的 Skill 分层

建议把能力拆成 4 层：

1. 通用基础设施层
2. 通用科研工程层
3. 项目运行约束层
4. 任务实例排障层

这样做的好处是：

- 越底层越能跨项目复用
- 越上层越贴近当前项目事实
- 新项目只需要替换第 3 层
- 新故障只需要补第 4 层

---

## 3. 分层清单

### L0. Repo Source Of Truth Navigator

抽象目标：

- 进入一个新仓库时，先建立“该看什么、不该看什么”的导航
- 找出 source of truth、活跃入口、历史遗留目录、可信结果优先级

适用范围：

- 任何科研仓库
- 任何多子目录、多历史产物、多 handover 文档的代码库

在本仓库中的来源：

- [PROJECT_GUIDE.md](./PROJECT_GUIDE.md)
- [NEWCOMER_HANDOVER.md](./NEWCOMER_HANDOVER.md)

建议 skill 名：

- `repo-source-of-truth-nav`

建议抽象内容：

- Git 根目录识别
- 活跃目录与历史目录区分
- 方法名映射
- 当前主训练入口
- 结果可信度优先级

状态：

- 已提炼成跨项目 skill
- 见 `imported_skills/repo-source-of-truth-nav/SKILL.md`

### L1. Remote Access Topology

抽象目标：

- 区分 login node、dev container、queue、jump host、VPN 这几个角色
- 明确什么工作该在哪一层做

适用范围：

- 所有带 VPN、堡垒机、登录节点、容器节点、调度系统的环境

在本仓库中的来源：

- `hpc-login-ssh`
- `hpc2-gpu-runtime`
- `hpc4-docker-vpn`
- `hpc4-end-to-end-access`

建议 skill 名：

- `remote-runtime-topology`

建议抽象内容：

- 先验证网络路径，再碰业务
- login node 不等于计算节点
- container `运行/等待/退出` 的语义
- 一个任务绑定一个运行时

状态：

- 已提炼成跨项目 skill
- 见 `imported_skills/remote-runtime-topology/SKILL.md`

### L2. Project Migration And Sync

抽象目标：

- 在不同机器之间同步代码、模型、缓存、结果
- 明确哪些东西应同步，哪些东西不应同步
- 优先复用仓库内迁移脚本

适用范围：

- 所有跨机器科研项目
- 所有需要 `rsync` 分层同步的工作流

在本仓库中的来源：

- `hpc-migration`
- [scripts/migrate_baseline_minimal.sh](./scripts/migrate_baseline_minimal.sh)
- [scripts/migrate_baseline_hpc2_longterm.sh](./scripts/migrate_baseline_hpc2_longterm.sh)

建议 skill 名：

- `project-sync-rsync-layered`

建议抽象内容：

- 代码、模型、缓存、输出分层同步
- 默认不要同步 `.venv`
- 默认不要无脑同步 `outputs/` 和 `results/`
- 先同步代码，再在目标机重建环境
- 有现成迁移脚本时优先复用脚本

状态：

- 已提炼成跨项目 skill
- 见 `imported_skills/project-sync-rsync-layered/SKILL.md`

### L3. Runtime Environment Rebuild

抽象目标：

- 在新机器或新容器上，验证并重建可用运行环境
- 区分“代码路径”和“解释器路径”是否一致

适用范围：

- 所有容器化或共享存储环境

在本仓库中的来源：

- `baseline-hpc2-codi-runtime`
- `hpc-migration`

建议 skill 名：

- `runtime-env-rebuild-verify`

建议抽象内容：

- `which python`
- `which torchrun`
- `python -V`
- 只在目标机重建 `.venv`
- 不把环境问题误判成代码问题

状态：

- 已提炼成跨项目 skill
- 见 `imported_skills/runtime-env-rebuild-verify/SKILL.md`

### L4. Experiment Contract Guard

抽象目标：

- 区分哪些参数属于“实验定义”
- 区分哪些参数属于“运行时妥协”
- 在排障时保护复现性

适用范围：

- 所有论文复现实验
- 所有需要保持 method definition 稳定的训练任务

在本仓库中的来源：

- `PROJECT_GUIDE.md`
- `hpc2-codi-train-triage`
- 这次关于 `decoder`、`num_latent`、`max_token_num` 的明确约束

建议 skill 名：

- `experiment-contract-guard`

建议抽象内容：

- 什么不能改
- 什么可以作为运行时妥协
- 什么时候必须先问用户
- 不要把“能跑起来”误当成“实验仍然等价”

状态：

- 已提炼成跨项目 skill
- 见 `imported_skills/experiment-contract-guard/SKILL.md`

### L5. Stage Workspace Layout

抽象目标：

- 把一次实验运行看成一个 stage
- 明确 stage 根目录下的模型、日志、缓存、结果、manifest 组织

适用范围：

- 所有需要把新实验和历史实验隔离的项目

在本仓库中的来源：

- `baseline-rebuttal-prepare`
- `baseline-hpc2-codi-runtime`
- [PROJECT_GUIDE.md](./PROJECT_GUIDE.md)

建议 skill 名：

- `stage-workspace-layout`

建议抽象内容：

- stage root 固定
- outputs / results / logs / manifests / cache 分离
- 不把新运行写回历史目录
- 每轮实验有自己的隔离空间

状态：

- 当前是 baseline 专属强约束
- 但抽象后非常适合别的论文项目

### L6. Canonical Entry Point Selection

抽象目标：

- 在一个复杂仓库里，找到“权威入口脚本”
- 优先走已有入口，而不是临时造 wrapper

适用范围：

- 所有脚本多、历史脚本多、入口混乱的项目

在本仓库中的来源：

- `baseline-rebuttal-prepare`
- `baseline-hpc2-codi-runtime`
- `CODI/train_on_gsm8k_dataset/`

建议 skill 名：

- `canonical-entrypoint-selector`

建议抽象内容：

- 准备入口
- 训练入口
- 评测入口
- 历史脚本与当前主线脚本的区分

状态：

- 已经在项目技能里体现
- 值得提成通用 skill

### L7. Training Monitoring And ETA

抽象目标：

- 用统一方式判断训练是否真的在推进
- 粗估剩余时间

适用范围：

- 任何 CLI / torchrun / tqdm 风格训练

在本仓库中的来源：

- `baseline-hpc2-codi-runtime`
- `hpc2-codi-train-triage`

建议 skill 名：

- `train-monitor-eta`

建议抽象内容：

- `ps`
- `nvidia-smi`
- `perl -pe 's/\r/\n/g'`
- 日志时间戳
- ETA 估算

状态：

- 已提炼成跨项目 skill
- 见 `imported_skills/train-monitor-eta/SKILL.md`

### L8. Failure Signature Triage

抽象目标：

- 把失败从“现象”提升到“签名”
- 不重复重试已经被证伪的组合

适用范围：

- 所有模型训练排障

在本仓库中的来源：

- `hpc2-codi-train-triage`
- 本次 8B 的 OOM / DDP checkpointing / Slurm CPU fallback 经验

建议 skill 名：

- `training-failure-signature-triage`

建议抽象内容：

- step0 OOM
- queue 侧 driver 不兼容
- CPU fallback
- reentrant checkpointing 与 DDP/LoRA 冲突
- 数据预处理假卡死与真实卡死的区分

状态：

- 已经开始沉淀
- 这是最适合随着项目推进持续追加的 skill

---

## 4. 哪些 skill 是“跨项目”的，哪些是“baseline 专属”的

### 高复用，建议未来所有项目都保留

- `repo-source-of-truth-nav`
- `remote-runtime-topology`
- `project-sync-rsync-layered`
- `runtime-env-rebuild-verify`
- `experiment-contract-guard`
- `canonical-entrypoint-selector`
- `train-monitor-eta`
- `training-failure-signature-triage`

### 中等复用，需要替换项目事实

- `stage-workspace-layout`

### 当前 baseline / CODI 强绑定

- `baseline-rebuttal-prepare`
- `baseline-hpc2-codi-runtime`
- `hpc2-codi-train-triage`

也就是说：

- L0-L2 更像基础设施 skill
- L3-L6 更像科研工程 skill
- L7-L8 更像任务运维 skill
- 只有最上面的 baseline/CODI 事实层需要随着项目切换而替换

---

## 5. 当前已落地的一批抽象层 skill

目前已经沉淀完成的跨项目 skill 有：

- `repo-source-of-truth-nav`
- `remote-runtime-topology`
- `project-sync-rsync-layered`
- `runtime-env-rebuild-verify`
- `experiment-contract-guard`
- `train-monitor-eta`

这一批基本覆盖了：

- 新仓库接手导航
- 远端访问拓扑判断
- 跨机器同步迁移
- 目标运行时环境重建
- 实验定义保护
- 训练监控与 ETA

如果后面要继续补，下一批更值得做的是：

### 1. `stage-workspace-layout`

原因：

- 很适合把新实验与历史实验隔离
- 对任何 rebuttal / revision / ablation 阶段都实用

### 2. `canonical-entrypoint-selector`

原因：

- 很多仓库不是没有脚本，而是脚本太多
- 能显著减少 AI 助手随手造 wrapper 的问题

### 3. `training-failure-signature-triage`

原因：

- 最适合随着真实故障持续追加
- 能把“重试经验”升级成可复用的故障签名库

---

## 6. 针对本仓库的最终建议

以后遇到新 AI 或新协作者，不要把整段聊天记录喂进去。

更好的最小上下文组合是：

1. [PROJECT_GUIDE.md](./PROJECT_GUIDE.md)
2. [NEWCOMER_HANDOVER.md](./NEWCOMER_HANDOVER.md)
3. 相关现有 skill
   - HPC2 访问：`hpc-login-ssh`、`hpc2-gpu-runtime`
   - 同步迁移：`hpc-migration`
   - baseline rebuttal 准备：`baseline-rebuttal-prepare`
   - HPC2 运行：`baseline-hpc2-codi-runtime`
   - 训练排障：`hpc2-codi-train-triage`
4. 当前任务对应的入口脚本
   - `CODI/train_on_gsm8k_dataset/*`
   - 或 `scripts/migrate_baseline_*`

一句话总结：

> 对这个仓库最好的 skill 化方式，不是继续堆更多 baseline 专属细节，而是把“仓库认知、远程拓扑、同步迁移、实验约束保护、训练监控、失败签名”这 6 类抽象能力沉淀出来，再让 baseline 文档负责提供项目事实。
