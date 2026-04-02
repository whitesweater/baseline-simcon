# Experiment Matrix

Generated from `scripts/locate_experiment.py`.

This file captures the 15 GSM8K experiment combinations across five methods and three backbones.

## `llama3-3b`

### `cot-sft` + `llama3-3b`

- Status: `supported`
- Train files:
  - `/data/yhao/baseline/Coconut/args/gsm_cot_llama3.yaml`
  - `/data/yhao/baseline/Coconut/args/gsm_cot_llama3_eval.yaml`
  - `/data/yhao/baseline/Coconut/scripts/batch_eval_cot_sft.sh`
- Train command: `cd /data/yhao/baseline/Coconut && torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_cot_llama3.yaml`
- Test files:
  - `/data/yhao/baseline/Coconut/args/gsm_cot_llama3_eval.yaml`
- Test command: `cd /data/yhao/baseline/Coconut && python run.py args/gsm_cot_llama3_eval.yaml`
- Batch eval command: `cd /data/yhao/baseline/Coconut && bash scripts/batch_eval_cot_sft.sh`
- Report paths:
  - `/data/yhao/baseline/Coconut/logs/eval_cot_llama3_*.log`
  - `/data/yhao/baseline/Coconut/logs/eval_llama3_all_*.log`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/THREE_BACKBONE_SUMMARY_20260330.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/EXPERIMENT_MASTER_SUMMARY.md`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/EXPERIMENT_PROGRESS_REPORT_20260328.md`
- Result paths:
  - `/data/yhao/baseline/Coconut/ckpts/gsm-cot-llama3/multi_eval_*.json`
  - `/data/yhao/baseline/Coconut/logs`
- Weight paths:
  - `/data/yhao/baseline/Coconut/ckpts/gsm-cot-llama3`
  - `/data/yhao/baseline/Coconut/ckpts/gsm-cot-llama3/checkpoint_*`
- Observed artifacts:
  - status: `checkpoints or eval outputs must be confirmed from Coconut artifacts`
  - checkpoint count: `0`
  - latest checkpoint: `-`
  - best checkpoint from summary: `-`
- Notes: Current checked-in train and eval args are ready. Batch multi-dataset eval uses scripts/batch_eval_cot_sft.sh.

### `simcot` + `llama3-3b`

- Status: `supported`
- Train files:
  - `/data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_llama3b.sh`
- Train command: `bash /data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_llama3b.sh`
- Test files:
  - `/data/yhao/baseline/CODI/test_multi_dataset.py`
- Test command: `cd /data/yhao/baseline/CODI && python test_multi_dataset.py --model_name_or_path "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/models/Llama-3.2-3B-Instruct" --ckpt_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_simcon/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11/checkpoint-<step>" --datasets "gsm8k math500 aime svamp gsm-hard asdiv" --num_runs 1 --result_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_simcon/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11" --seed 11 --model_max_length 512 --bf16 --lora_r 128 --lora_alpha 32 --lora_init --batch_size 8 --greedy True --num_latent 6 --use_prj True --prj_dim 3072 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 --remove_eos True --use_lora True`
- Report paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_simcon/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11/summary/comparison_matrix.csv`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_simcon/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11/summary/all_results.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/THREE_BACKBONE_SUMMARY_20260330.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/EXPERIMENT_MASTER_SUMMARY.md`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/EXPERIMENT_PROGRESS_REPORT_20260328.md`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1_simcon_20260327_offline/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_simcon_20260327_offline_gsm8k_llama3b_simcon/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11/summary/comparison_matrix.csv`
- Result paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_simcon/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_simcon/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11/datasets`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_simcon/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11/models`
- Weight paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_simcon/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_simcon/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11/checkpoint-*`
- Observed artifacts:
  - status: `checkpoints present, no comparison_matrix.csv`
  - checkpoint count: `2`
  - latest checkpoint: `checkpoint-11996`
  - best checkpoint from summary: `-`
- Notes: Train wrappers already run post-train multi-dataset evaluation automatically unless CODI_POST_TRAIN_EVAL disables it. The current reportable SIM-CoT sweep for llama3-3b lives in the deeper offline-side run under rebuttal_20260325/multimodel_gsm8k_math500_aime_v1_simcon_20260327_offline; the main-root observation shown here may be shallower.

### `simcot+sircl` + `llama3-3b`

- Status: `supported`
- Train files:
  - `/data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_llama3b.sh`
- Train command: `bash /data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_llama3b.sh --sircl`
- Test files:
  - `/data/yhao/baseline/CODI/test_multi_dataset.py`
- Test command: `cd /data/yhao/baseline/CODI && python test_multi_dataset.py --model_name_or_path "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/models/Llama-3.2-3B-Instruct" --ckpt_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_simcon_sircl/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11/checkpoint-<step>" --datasets "gsm8k math500 aime svamp gsm-hard asdiv" --num_runs 1 --result_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_simcon_sircl/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11" --seed 11 --model_max_length 512 --bf16 --lora_r 128 --lora_alpha 32 --lora_init --batch_size 8 --greedy True --num_latent 6 --use_prj True --prj_dim 3072 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 --remove_eos True --use_lora True`
- Report paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_simcon_sircl/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11/summary/comparison_matrix.csv`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_simcon_sircl/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11/summary/all_results.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/THREE_BACKBONE_SUMMARY_20260330.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/EXPERIMENT_MASTER_SUMMARY.md`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/EXPERIMENT_PROGRESS_REPORT_20260328.md`
- Result paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_simcon_sircl/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_simcon_sircl/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11/datasets`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_simcon_sircl/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11/models`
- Weight paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_simcon_sircl/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_simcon_sircl/Llama-3.2-3B-Instruct/ep_10/lr_0.0003/seed_11/checkpoint-*`
- Observed artifacts:
  - status: `summary present`
  - checkpoint count: `10`
  - latest checkpoint: `checkpoint-59980`
  - best checkpoint from summary: `checkpoint-53982`
- Notes: Train wrappers already run post-train multi-dataset evaluation automatically unless CODI_POST_TRAIN_EVAL disables it.

### `codi` + `llama3-3b`

- Status: `supported`
- Train files:
  - `/data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_llama3b_codi.sh`
- Train command: `bash /data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_llama3b_codi.sh`
- Test files:
  - `/data/yhao/baseline/CODI/test_multi_dataset.py`
- Test command: `cd /data/yhao/baseline/CODI && python test_multi_dataset.py --model_name_or_path "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/models/Llama-3.2-3B-Instruct" --ckpt_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_codi/Llama-3.2-3B-Instruct/ep_8/lr_0.0003/seed_11/checkpoint-<step>" --datasets "gsm8k math500 aime svamp gsm-hard asdiv" --num_runs 1 --result_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_codi/Llama-3.2-3B-Instruct/ep_8/lr_0.0003/seed_11" --seed 11 --model_max_length 512 --bf16 --lora_r 128 --lora_alpha 32 --lora_init --batch_size 8 --greedy True --num_latent 6 --use_prj True --prj_dim 3072 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 --remove_eos True --use_lora True`
- Report paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_codi/Llama-3.2-3B-Instruct/ep_8/lr_0.0003/seed_11/summary/comparison_matrix.csv`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_codi/Llama-3.2-3B-Instruct/ep_8/lr_0.0003/seed_11/summary/all_results.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/THREE_BACKBONE_SUMMARY_20260330.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/EXPERIMENT_MASTER_SUMMARY.md`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/EXPERIMENT_PROGRESS_REPORT_20260328.md`
- Result paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_codi/Llama-3.2-3B-Instruct/ep_8/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_codi/Llama-3.2-3B-Instruct/ep_8/lr_0.0003/seed_11/datasets`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_codi/Llama-3.2-3B-Instruct/ep_8/lr_0.0003/seed_11/models`
- Weight paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_codi/Llama-3.2-3B-Instruct/ep_8/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_codi/Llama-3.2-3B-Instruct/ep_8/lr_0.0003/seed_11/checkpoint-*`
- Observed artifacts:
  - status: `no checkpoints observed`
  - checkpoint count: `0`
  - latest checkpoint: `-`
  - best checkpoint from summary: `-`
- Notes: Train wrappers already run post-train multi-dataset evaluation automatically unless CODI_POST_TRAIN_EVAL disables it. The current reportable llama3-3b CODI row is a partial-sweep story tracked in the stage summary docs; do not infer reportable completeness from the main-root observation alone.

### `codi+sircl` + `llama3-3b`

- Status: `supported`
- Train files:
  - `/data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_llama3b_codi.sh`
- Train command: `bash /data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_llama3b_codi.sh --sircl`
- Test files:
  - `/data/yhao/baseline/CODI/test_multi_dataset.py`
- Test command: `cd /data/yhao/baseline/CODI && python test_multi_dataset.py --model_name_or_path "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/models/Llama-3.2-3B-Instruct" --ckpt_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_codi_sircl/Llama-3.2-3B-Instruct/ep_8/lr_0.0003/seed_11/checkpoint-<step>" --datasets "gsm8k math500 aime svamp gsm-hard asdiv" --num_runs 1 --result_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_codi_sircl/Llama-3.2-3B-Instruct/ep_8/lr_0.0003/seed_11" --seed 11 --model_max_length 512 --bf16 --lora_r 128 --lora_alpha 32 --lora_init --batch_size 8 --greedy True --num_latent 6 --use_prj True --prj_dim 3072 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 --remove_eos True --use_lora True`
- Report paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_codi_sircl/Llama-3.2-3B-Instruct/ep_8/lr_0.0003/seed_11/summary/comparison_matrix.csv`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_codi_sircl/Llama-3.2-3B-Instruct/ep_8/lr_0.0003/seed_11/summary/all_results.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/THREE_BACKBONE_SUMMARY_20260330.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/EXPERIMENT_MASTER_SUMMARY.md`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/EXPERIMENT_PROGRESS_REPORT_20260328.md`
- Result paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_codi_sircl/Llama-3.2-3B-Instruct/ep_8/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_codi_sircl/Llama-3.2-3B-Instruct/ep_8/lr_0.0003/seed_11/datasets`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_codi_sircl/Llama-3.2-3B-Instruct/ep_8/lr_0.0003/seed_11/models`
- Weight paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_codi_sircl/Llama-3.2-3B-Instruct/ep_8/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_llama3b_codi_sircl/Llama-3.2-3B-Instruct/ep_8/lr_0.0003/seed_11/checkpoint-*`
- Observed artifacts:
  - status: `summary present`
  - checkpoint count: `8`
  - latest checkpoint: `checkpoint-47984`
  - best checkpoint from summary: `checkpoint-47984`
- Notes: Train wrappers already run post-train multi-dataset evaluation automatically unless CODI_POST_TRAIN_EVAL disables it.

## `qwen3-4b`

### `cot-sft` + `qwen3-4b`

- Status: `supported`
- Train files:
  - `/data/yhao/baseline/Coconut/scripts/train_cot_qwen3.sh`
  - `/data/yhao/baseline/Coconut/scripts/eval_cot_qwen3.sh`
  - `/data/yhao/baseline/Coconut/args/gsm_cot_qwen3.yaml`
  - `/data/yhao/baseline/Coconut/args/gsm_cot_qwen3_eval.yaml`
  - `/data/yhao/baseline/Coconut/scripts/batch_eval_cot_sft.sh`
- Train command: `cd /data/yhao/baseline/Coconut && bash scripts/train_cot_qwen3.sh 4`
- Test files:
  - `/data/yhao/baseline/Coconut/scripts/eval_cot_qwen3.sh`
- Test command: `cd /data/yhao/baseline/Coconut && bash scripts/eval_cot_qwen3.sh 4`
- Batch eval command: `cd /data/yhao/baseline/Coconut && bash scripts/batch_eval_cot_sft.sh`
- Report paths:
  - `/data/yhao/baseline/Coconut/logs/eval_cot_qwen3_*.log`
  - `/data/yhao/baseline/Coconut/logs/eval_qwen3_all_*.log`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/THREE_BACKBONE_SUMMARY_20260330.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/EXPERIMENT_MASTER_SUMMARY.md`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/EXPERIMENT_PROGRESS_REPORT_20260328.md`
- Result paths:
  - `/data/yhao/baseline/Coconut/ckpts/gsm-qwen3-cot-sft/multi_eval_*.json`
  - `/data/yhao/baseline/Coconut/logs`
- Weight paths:
  - `/data/yhao/baseline/Coconut/ckpts/gsm-qwen3-cot-sft`
  - `/data/yhao/baseline/Coconut/ckpts/gsm-qwen3-cot-sft/checkpoint_*`
- Observed artifacts:
  - status: `checkpoints or eval outputs must be confirmed from Coconut artifacts`
  - checkpoint count: `0`
  - latest checkpoint: `-`
  - best checkpoint from summary: `-`
- Notes: Current checked-in train and eval wrappers target Qwen3-4B directly.

### `simcot` + `qwen3-4b`

- Status: `supported`
- Train files:
  - `/data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_qwen3.sh`
- Train command: `bash /data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_qwen3.sh`
- Test files:
  - `/data/yhao/baseline/CODI/test_multi_dataset.py`
- Test command: `cd /data/yhao/baseline/CODI && python test_multi_dataset.py --model_name_or_path "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/models/Qwen3-4B" --ckpt_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_simcon/Qwen3-4B/ep_8/lr_0.0003/seed_11/checkpoint-<step>" --datasets "gsm8k math500 aime svamp gsm-hard asdiv" --num_runs 1 --result_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_simcon/Qwen3-4B/ep_8/lr_0.0003/seed_11" --seed 11 --model_max_length 512 --bf16 --lora_r 128 --lora_alpha 32 --lora_init --batch_size 8 --greedy True --num_latent 6 --use_prj True --prj_dim 2560 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 --remove_eos True --use_lora True`
- Report paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_simcon/Qwen3-4B/ep_8/lr_0.0003/seed_11/summary/comparison_matrix.csv`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_simcon/Qwen3-4B/ep_8/lr_0.0003/seed_11/summary/all_results.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/THREE_BACKBONE_SUMMARY_20260330.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/EXPERIMENT_MASTER_SUMMARY.md`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/EXPERIMENT_PROGRESS_REPORT_20260328.md`
- Result paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_simcon/Qwen3-4B/ep_8/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_simcon/Qwen3-4B/ep_8/lr_0.0003/seed_11/datasets`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_simcon/Qwen3-4B/ep_8/lr_0.0003/seed_11/models`
- Weight paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_simcon/Qwen3-4B/ep_8/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_simcon/Qwen3-4B/ep_8/lr_0.0003/seed_11/checkpoint-*`
- Observed artifacts:
  - status: `no checkpoints observed`
  - checkpoint count: `0`
  - latest checkpoint: `-`
  - best checkpoint from summary: `-`
- Notes: Train wrappers already run post-train multi-dataset evaluation automatically unless CODI_POST_TRAIN_EVAL disables it.

### `simcot+sircl` + `qwen3-4b`

- Status: `supported`
- Train files:
  - `/data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_qwen3.sh`
- Train command: `bash /data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_qwen3.sh --sircl`
- Test files:
  - `/data/yhao/baseline/CODI/test_multi_dataset.py`
- Test command: `cd /data/yhao/baseline/CODI && python test_multi_dataset.py --model_name_or_path "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/models/Qwen3-4B" --ckpt_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_simcon_sircl/Qwen3-4B/ep_8/lr_0.0003/seed_11/checkpoint-<step>" --datasets "gsm8k math500 aime svamp gsm-hard asdiv" --num_runs 1 --result_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_simcon_sircl/Qwen3-4B/ep_8/lr_0.0003/seed_11" --seed 11 --model_max_length 512 --bf16 --lora_r 128 --lora_alpha 32 --lora_init --batch_size 8 --greedy True --num_latent 6 --use_prj True --prj_dim 2560 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 --remove_eos True --use_lora True`
- Report paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_simcon_sircl/Qwen3-4B/ep_8/lr_0.0003/seed_11/summary/comparison_matrix.csv`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_simcon_sircl/Qwen3-4B/ep_8/lr_0.0003/seed_11/summary/all_results.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/THREE_BACKBONE_SUMMARY_20260330.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/EXPERIMENT_MASTER_SUMMARY.md`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/EXPERIMENT_PROGRESS_REPORT_20260328.md`
- Result paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_simcon_sircl/Qwen3-4B/ep_8/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_simcon_sircl/Qwen3-4B/ep_8/lr_0.0003/seed_11/datasets`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_simcon_sircl/Qwen3-4B/ep_8/lr_0.0003/seed_11/models`
- Weight paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_simcon_sircl/Qwen3-4B/ep_8/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_simcon_sircl/Qwen3-4B/ep_8/lr_0.0003/seed_11/checkpoint-*`
- Observed artifacts:
  - status: `no checkpoints observed`
  - checkpoint count: `0`
  - latest checkpoint: `-`
  - best checkpoint from summary: `-`
- Notes: Train wrappers already run post-train multi-dataset evaluation automatically unless CODI_POST_TRAIN_EVAL disables it.

### `codi` + `qwen3-4b`

- Status: `supported`
- Train files:
  - `/data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_qwen3_codi.sh`
- Train command: `bash /data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_qwen3_codi.sh`
- Test files:
  - `/data/yhao/baseline/CODI/test_multi_dataset.py`
- Test command: `cd /data/yhao/baseline/CODI && python test_multi_dataset.py --model_name_or_path "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/models/Qwen3-4B" --ckpt_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_codi/Qwen3-4B/ep_8/lr_0.0003/seed_11/checkpoint-<step>" --datasets "gsm8k math500 aime svamp gsm-hard asdiv" --num_runs 1 --result_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_codi/Qwen3-4B/ep_8/lr_0.0003/seed_11" --seed 11 --model_max_length 512 --bf16 --lora_r 128 --lora_alpha 32 --lora_init --batch_size 8 --greedy True --num_latent 6 --use_prj True --prj_dim 2560 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 --remove_eos True --use_lora True`
- Report paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_codi/Qwen3-4B/ep_8/lr_0.0003/seed_11/summary/comparison_matrix.csv`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_codi/Qwen3-4B/ep_8/lr_0.0003/seed_11/summary/all_results.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/THREE_BACKBONE_SUMMARY_20260330.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/EXPERIMENT_MASTER_SUMMARY.md`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/EXPERIMENT_PROGRESS_REPORT_20260328.md`
- Result paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_codi/Qwen3-4B/ep_8/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_codi/Qwen3-4B/ep_8/lr_0.0003/seed_11/datasets`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_codi/Qwen3-4B/ep_8/lr_0.0003/seed_11/models`
- Weight paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_codi/Qwen3-4B/ep_8/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_codi/Qwen3-4B/ep_8/lr_0.0003/seed_11/checkpoint-*`
- Observed artifacts:
  - status: `checkpoints present, no comparison_matrix.csv`
  - checkpoint count: `7`
  - latest checkpoint: `checkpoint-41944`
  - best checkpoint from summary: `-`
- Notes: Train wrappers already run post-train multi-dataset evaluation automatically unless CODI_POST_TRAIN_EVAL disables it.

### `codi+sircl` + `qwen3-4b`

- Status: `supported`
- Train files:
  - `/data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_qwen3_codi.sh`
- Train command: `bash /data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_qwen3_codi.sh --sircl`
- Test files:
  - `/data/yhao/baseline/CODI/test_multi_dataset.py`
- Test command: `cd /data/yhao/baseline/CODI && python test_multi_dataset.py --model_name_or_path "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/models/Qwen3-4B" --ckpt_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_codi_sircl/Qwen3-4B/ep_8/lr_0.0003/seed_11/checkpoint-<step>" --datasets "gsm8k math500 aime svamp gsm-hard asdiv" --num_runs 1 --result_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_codi_sircl/Qwen3-4B/ep_8/lr_0.0003/seed_11" --seed 11 --model_max_length 512 --bf16 --lora_r 128 --lora_alpha 32 --lora_init --batch_size 8 --greedy True --num_latent 6 --use_prj True --prj_dim 2560 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 --remove_eos True --use_lora True`
- Report paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_codi_sircl/Qwen3-4B/ep_8/lr_0.0003/seed_11/summary/comparison_matrix.csv`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_codi_sircl/Qwen3-4B/ep_8/lr_0.0003/seed_11/summary/all_results.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/THREE_BACKBONE_SUMMARY_20260330.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/EXPERIMENT_MASTER_SUMMARY.md`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/EXPERIMENT_PROGRESS_REPORT_20260328.md`
- Result paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_codi_sircl/Qwen3-4B/ep_8/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_codi_sircl/Qwen3-4B/ep_8/lr_0.0003/seed_11/datasets`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_codi_sircl/Qwen3-4B/ep_8/lr_0.0003/seed_11/models`
- Weight paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_codi_sircl/Qwen3-4B/ep_8/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_4b_codi_sircl/Qwen3-4B/ep_8/lr_0.0003/seed_11/checkpoint-*`
- Observed artifacts:
  - status: `no checkpoints observed`
  - checkpoint count: `0`
  - latest checkpoint: `-`
  - best checkpoint from summary: `-`
- Notes: Train wrappers already run post-train multi-dataset evaluation automatically unless CODI_POST_TRAIN_EVAL disables it.

## `qwen3-1.7b`

### `cot-sft` + `qwen3-1.7b`

- Status: `supported`
- Train files:
  - `/data/yhao/baseline/Coconut/args/gsm_cot_qwen3_1p7b.yaml`
  - `/data/yhao/baseline/Coconut/args/gsm_cot_qwen3_1p7b_eval.yaml`
  - `/data/yhao/baseline/Coconut/scripts/train_cot_qwen3_1p7b.sh`
  - `/data/yhao/baseline/Coconut/scripts/eval_cot_qwen3_1p7b.sh`
  - `/data/yhao/baseline/Coconut/scripts/batch_eval_cot_sft.sh`
- Train command: `cd /data/yhao/baseline/Coconut && bash scripts/train_cot_qwen3_1p7b.sh 4`
- Test files:
  - `/data/yhao/baseline/Coconut/scripts/eval_cot_qwen3_1p7b.sh`
- Test command: `cd /data/yhao/baseline/Coconut && bash scripts/eval_cot_qwen3_1p7b.sh 4`
- Batch eval command: `cd /data/yhao/baseline/Coconut && bash scripts/batch_eval_cot_sft.sh`
- Report paths:
  - `/data/yhao/baseline/Coconut/logs/eval_qwen3_1p7b_multi_*.log`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/THREE_BACKBONE_SUMMARY_20260330.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/EXPERIMENT_MASTER_SUMMARY.md`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/EXPERIMENT_PROGRESS_REPORT_20260328.md`
- Result paths:
  - `/data/yhao/baseline/Coconut/ckpts/gsm-qwen3-1p7b-cot-sft/multi_eval_*.json`
  - `/data/yhao/baseline/Coconut/logs`
- Weight paths:
  - `/data/yhao/baseline/Coconut/ckpts/gsm-qwen3-1p7b-cot-sft`
  - `/data/yhao/baseline/Coconut/ckpts/gsm-qwen3-1p7b-cot-sft/checkpoint_*`
- Observed artifacts:
  - status: `checkpoints present, no comparison_matrix.csv`
  - checkpoint count: `1`
  - latest checkpoint: `checkpoint_1`
  - best checkpoint from summary: `-`
- Notes: Train, single-dataset eval, and batch multi-dataset eval wrappers are now checked in. A real run already exists and stopped early; confirm logs before claiming a reportable result.

### `simcot` + `qwen3-1.7b`

- Status: `supported`
- Train files:
  - `/data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_qwen3_1p7b.sh`
- Train command: `bash /data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_qwen3_1p7b.sh`
- Test files:
  - `/data/yhao/baseline/CODI/test_multi_dataset.py`
- Test command: `cd /data/yhao/baseline/CODI && python test_multi_dataset.py --model_name_or_path "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/models/Qwen3-1.7B" --ckpt_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_simcon/Qwen3-1.7B/ep_10/lr_0.0003/seed_11/checkpoint-<step>" --datasets "gsm8k math500 aime svamp gsm-hard asdiv" --num_runs 1 --result_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_simcon/Qwen3-1.7B/ep_10/lr_0.0003/seed_11" --seed 11 --model_max_length 512 --bf16 --lora_r 128 --lora_alpha 32 --lora_init --batch_size 16 --greedy True --num_latent 6 --use_prj True --prj_dim 2048 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 --remove_eos True --use_lora True`
- Report paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_simcon/Qwen3-1.7B/ep_10/lr_0.0003/seed_11/summary/comparison_matrix.csv`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_simcon/Qwen3-1.7B/ep_10/lr_0.0003/seed_11/summary/all_results.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/THREE_BACKBONE_SUMMARY_20260330.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/EXPERIMENT_MASTER_SUMMARY.md`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/EXPERIMENT_PROGRESS_REPORT_20260328.md`
- Result paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_simcon/Qwen3-1.7B/ep_10/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_simcon/Qwen3-1.7B/ep_10/lr_0.0003/seed_11/datasets`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_simcon/Qwen3-1.7B/ep_10/lr_0.0003/seed_11/models`
- Weight paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_simcon/Qwen3-1.7B/ep_10/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_simcon/Qwen3-1.7B/ep_10/lr_0.0003/seed_11/checkpoint-*`
- Observed artifacts:
  - status: `checkpoints present, no comparison_matrix.csv`
  - checkpoint count: `6`
  - latest checkpoint: `checkpoint-17976`
  - best checkpoint from summary: `-`
- Notes: Train wrappers already run post-train multi-dataset evaluation automatically unless CODI_POST_TRAIN_EVAL disables it. qwen3-1.7b now has real artifacts in the main run root; verify live summaries before describing a line as complete.

### `simcot+sircl` + `qwen3-1.7b`

- Status: `supported`
- Train files:
  - `/data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_qwen3_1p7b.sh`
- Train command: `bash /data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_qwen3_1p7b.sh --sircl`
- Test files:
  - `/data/yhao/baseline/CODI/test_multi_dataset.py`
- Test command: `cd /data/yhao/baseline/CODI && python test_multi_dataset.py --model_name_or_path "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/models/Qwen3-1.7B" --ckpt_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_simcon_sircl/Qwen3-1.7B/ep_10/lr_0.0003/seed_11/checkpoint-<step>" --datasets "gsm8k math500 aime svamp gsm-hard asdiv" --num_runs 1 --result_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_simcon_sircl/Qwen3-1.7B/ep_10/lr_0.0003/seed_11" --seed 11 --model_max_length 512 --bf16 --lora_r 128 --lora_alpha 32 --lora_init --batch_size 16 --greedy True --num_latent 6 --use_prj True --prj_dim 2048 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 --remove_eos True --use_lora True`
- Report paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_simcon_sircl/Qwen3-1.7B/ep_10/lr_0.0003/seed_11/summary/comparison_matrix.csv`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_simcon_sircl/Qwen3-1.7B/ep_10/lr_0.0003/seed_11/summary/all_results.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/THREE_BACKBONE_SUMMARY_20260330.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/EXPERIMENT_MASTER_SUMMARY.md`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/EXPERIMENT_PROGRESS_REPORT_20260328.md`
- Result paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_simcon_sircl/Qwen3-1.7B/ep_10/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_simcon_sircl/Qwen3-1.7B/ep_10/lr_0.0003/seed_11/datasets`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_simcon_sircl/Qwen3-1.7B/ep_10/lr_0.0003/seed_11/models`
- Weight paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_simcon_sircl/Qwen3-1.7B/ep_10/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_simcon_sircl/Qwen3-1.7B/ep_10/lr_0.0003/seed_11/checkpoint-*`
- Observed artifacts:
  - status: `checkpoints present, no comparison_matrix.csv`
  - checkpoint count: `6`
  - latest checkpoint: `checkpoint-17976`
  - best checkpoint from summary: `-`
- Notes: Train wrappers already run post-train multi-dataset evaluation automatically unless CODI_POST_TRAIN_EVAL disables it. qwen3-1.7b now has real artifacts in the main run root; verify live summaries before describing a line as complete.

### `codi` + `qwen3-1.7b`

- Status: `supported`
- Train files:
  - `/data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_qwen3_1p7b_codi.sh`
- Train command: `bash /data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_qwen3_1p7b_codi.sh`
- Test files:
  - `/data/yhao/baseline/CODI/test_multi_dataset.py`
- Test command: `cd /data/yhao/baseline/CODI && python test_multi_dataset.py --model_name_or_path "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/models/Qwen3-1.7B" --ckpt_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_codi/Qwen3-1.7B/ep_8/lr_0.0003/seed_11/checkpoint-<step>" --datasets "gsm8k math500 aime svamp gsm-hard asdiv" --num_runs 1 --result_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_codi/Qwen3-1.7B/ep_8/lr_0.0003/seed_11" --seed 11 --model_max_length 512 --bf16 --lora_r 128 --lora_alpha 32 --lora_init --batch_size 16 --greedy True --num_latent 6 --use_prj True --prj_dim 2048 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 --remove_eos True --use_lora True`
- Report paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_codi/Qwen3-1.7B/ep_8/lr_0.0003/seed_11/summary/comparison_matrix.csv`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_codi/Qwen3-1.7B/ep_8/lr_0.0003/seed_11/summary/all_results.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/THREE_BACKBONE_SUMMARY_20260330.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/EXPERIMENT_MASTER_SUMMARY.md`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/EXPERIMENT_PROGRESS_REPORT_20260328.md`
- Result paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_codi/Qwen3-1.7B/ep_8/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_codi/Qwen3-1.7B/ep_8/lr_0.0003/seed_11/datasets`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_codi/Qwen3-1.7B/ep_8/lr_0.0003/seed_11/models`
- Weight paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_codi/Qwen3-1.7B/ep_8/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_codi/Qwen3-1.7B/ep_8/lr_0.0003/seed_11/checkpoint-*`
- Observed artifacts:
  - status: `summary present`
  - checkpoint count: `8`
  - latest checkpoint: `checkpoint-23968`
  - best checkpoint from summary: `checkpoint-20972`
- Notes: Train wrappers already run post-train multi-dataset evaluation automatically unless CODI_POST_TRAIN_EVAL disables it. qwen3-1.7b now has real artifacts in the main run root; verify live summaries before describing a line as complete.

### `codi+sircl` + `qwen3-1.7b`

- Status: `supported`
- Train files:
  - `/data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_qwen3_1p7b_codi.sh`
- Train command: `bash /data/yhao/baseline/CODI/train_on_gsm8k_dataset/train_qwen3_1p7b_codi.sh --sircl`
- Test files:
  - `/data/yhao/baseline/CODI/test_multi_dataset.py`
- Test command: `cd /data/yhao/baseline/CODI && python test_multi_dataset.py --model_name_or_path "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/models/Qwen3-1.7B" --ckpt_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_codi_sircl/Qwen3-1.7B/ep_8/lr_0.0003/seed_11/checkpoint-<step>" --datasets "gsm8k math500 aime svamp gsm-hard asdiv" --num_runs 1 --result_dir "/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_codi_sircl/Qwen3-1.7B/ep_8/lr_0.0003/seed_11" --seed 11 --model_max_length 512 --bf16 --lora_r 128 --lora_alpha 32 --lora_init --batch_size 16 --greedy True --num_latent 6 --use_prj True --prj_dim 2048 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 --remove_eos True --use_lora True`
- Report paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_codi_sircl/Qwen3-1.7B/ep_8/lr_0.0003/seed_11/summary/comparison_matrix.csv`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_codi_sircl/Qwen3-1.7B/ep_8/lr_0.0003/seed_11/summary/all_results.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/THREE_BACKBONE_SUMMARY_20260330.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.md`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/CURRENT_MULTIMODEL_RESULTS_20260329.csv`
  - `/data/yhao/baseline/CODI/plots/rebuttal_20260328/EXPERIMENT_MASTER_SUMMARY.md`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/EXPERIMENT_PROGRESS_REPORT_20260328.md`
- Result paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_codi_sircl/Qwen3-1.7B/ep_8/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_codi_sircl/Qwen3-1.7B/ep_8/lr_0.0003/seed_11/datasets`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/results/checkpoint_sweeps/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_codi_sircl/Qwen3-1.7B/ep_8/lr_0.0003/seed_11/models`
- Weight paths:
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_codi_sircl/Qwen3-1.7B/ep_8/lr_0.0003/seed_11`
  - `/data/yhao/baseline/CODI_rebuttal_runs/rebuttal_20260325/multimodel_gsm8k_math500_aime_v1/outputs/multimodel_gsm8k_math500_aime_v1_gsm8k_qwen3_1p7b_codi_sircl/Qwen3-1.7B/ep_8/lr_0.0003/seed_11/checkpoint-*`
- Observed artifacts:
  - status: `summary present`
  - checkpoint count: `8`
  - latest checkpoint: `checkpoint-23968`
  - best checkpoint from summary: `checkpoint-8988`
- Notes: Train wrappers already run post-train multi-dataset evaluation automatically unless CODI_POST_TRAIN_EVAL disables it. qwen3-1.7b now has real artifacts in the main run root; verify live summaries before describing a line as complete.
