# Rebuttal Workspace Rules

`CODI/REBUTTAL_WORKSPACE.md` 不再单独维护完整的 workspace 规则。

rebuttal / revision 阶段的输出隔离、Git 边界、可信历史结果、SemCoT 处理原则和提交规则，已经合并到仓库根目录：

- `../PROJECT_GUIDE.md`

如果你现在要开始跑新的 CODI 实验，先做这几步：

```bash
cd /data/yhao/baseline/CODI
source config.env
```

确认以下变量已经指向新的 rebuttal 输出根目录：

- `CODI_RUN_ROOT`
- `CODI_SAVE_DIR`
- `CODI_RESULT_DIR`

当前默认目录是：

```bash
/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325
```

如果你跑的是当前这轮 multi-backbone rebuttal，新入口是：

- `train_on_gsm8k_dataset/prepare_assets.sh`
- `train_on_gsm8k_dataset/train_llama1b.sh`
- `train_on_gsm8k_dataset/train_llama3b.sh`
- `train_on_gsm8k_dataset/train_llama8b.sh`
- `train_on_gsm8k_dataset/train_qwen3.sh`
- `train_on_gsm8k_dataset/eval_llama1b_math500_aime.sh`

如果你是从旧文档跳转到这里，请改为先阅读：

1. `../PROJECT_GUIDE.md`
2. `README.md`
3. `train_on_gsm8k_dataset/`
4. `TESTING_GUIDE.md`
