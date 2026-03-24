# CODI Project Guide

`CODI/PROJECT_GUIDE.md` 不再是整个仓库的总指南。

全项目级的 handover、目录边界、方法映射、可信结果、rebuttal 规则和 Git 工作流，已经上移到仓库根目录：

- `../PROJECT_GUIDE.md`

如果你是从旧链接或旧笔记跳转到这里，建议现在改为下面的阅读顺序：

1. `../PROJECT_GUIDE.md`
2. `README.md`
3. `REBUTTAL_WORKSPACE.md`
4. `train_on_gsm8k_dataset/`
5. `TESTING_GUIDE.md`
6. `train.py`
7. `src/model.py`

## 这个文件现在保留什么

这个文件现在只承担两个作用：

- 兼容旧入口，避免原来的文档链接失效
- 作为 CODI 子项目的轻量导读页，把读者引回根级总指南

## 进入 CODI 子项目时先看哪里

如果你已经明确是在处理 `CODI/` 这条代码线，优先看：

- `README.md`
- `train_on_gsm8k_dataset/`
- `TESTING_GUIDE.md`
- `train.py`
- `src/model.py`
- `scripts/`
- `train_on_svamp_dataset/`
- `train_on_multiarith_dataset/`

## 仍然适用于 CODI 的关键事实

- Git 根目录是 `/data/yhao/baseline`，不是 `CODI/`
- `simcon` 对应论文里的 SIM-CoT 方法线
- `SIRCL` 是可插拔的统一稳定器，不只服务 CODI
- `CODI/local_datasets/` 是当前默认本地数据入口
- `CODI/SemCoT/` 只是外部参考目录
- 从 2026-03-25 起，新实验默认写到 `../CODI_rebuttal_runs/rebuttal_20260325`
- 当前多 backbone rebuttal 训练优先走 `train_on_gsm8k_dataset/`

如果这些事实和其他旧文档冲突，以根目录 `PROJECT_GUIDE.md` 为准。
