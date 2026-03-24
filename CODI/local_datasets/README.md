# Local Datasets Used By CODI

This directory vendors only the dataset files that the active CODI codebase actually uses.

Scope:
- `coin_flip/{train_42.json, eval_42.json, LICENSE}`
- `multiarith/{train_42.json, eval_42.json}`
- `svamp/{train_42.json, eval_42.json, LICENSE}`

Why this exists:
- We treat `CODI/SemCoT/` as an external reference repository.
- The active CODI training and evaluation code only depends on these JSON files.
- Copying the exact files here removes the hidden runtime dependency on the whole `SemCoT` repo.

Source of truth:
- The copied files currently come from `CODI/SemCoT/datasets/...` on the H800 machine.
- If the data is refreshed in the future, update the files here intentionally.
