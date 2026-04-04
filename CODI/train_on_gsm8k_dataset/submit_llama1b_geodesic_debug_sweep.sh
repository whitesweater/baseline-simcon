#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PATH=/opt/slurm/bin:${PATH}

# shellcheck disable=SC1091
source "${CODI_DIR}/config.env"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

EXPT_NAME="${CODI_MULTIMODEL_TAG}_gsm8k_llama1b_simcon_geodesic_only"
MODEL_NAME="${CODI_MM_LLAMA1B_PATH##*/}"
NUM_EPOCHS=12
LEARNING_RATE=0.0008
SEED=11
EVAL_BATCH_SIZE="${CODI_EVAL_BATCH_SIZE:-32}"
EVAL_PARTITION="${CODI_LLAMA1B_GEO_EVAL_PARTITION:-i64m1tga800u}"
MERGE_PARTITION="${CODI_LLAMA1B_GEO_MERGE_PARTITION:-i64m512u}"

CHECKPOINT_ROOT="${CODI_MULTIMODEL_SAVE_DIR}/${EXPT_NAME}/${MODEL_NAME}/ep_${NUM_EPOCHS}/lr_${LEARNING_RATE}/seed_${SEED}"
SWEEP_RESULT_DIR="${CODI_MULTIMODEL_RESULT_DIR}/checkpoint_sweeps/${EXPT_NAME}/${MODEL_NAME}/ep_${NUM_EPOCHS}/lr_${LEARNING_RATE}/seed_${SEED}"
SHARD_ROOT="${CODI_MULTIMODEL_RESULT_DIR}/checkpoint_sweeps_shards/${EXPT_NAME}/${MODEL_NAME}/ep_${NUM_EPOCHS}/lr_${LEARNING_RATE}/seed_${SEED}"
EXPECTED_LIST="${SHARD_ROOT}/expected_checkpoints.txt"
SUBMIT_MANIFEST="${SHARD_ROOT}/submission_manifest.tsv"

if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
  echo "Checkpoint root does not exist: ${CHECKPOINT_ROOT}"
  exit 1
fi

mkdir -p "${SHARD_ROOT}"

find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | xargs -r -n 1 basename > "${EXPECTED_LIST}"

mapfile -t CHECKPOINT_NAMES < "${EXPECTED_LIST}"

if [[ ${#CHECKPOINT_NAMES[@]} -eq 0 ]]; then
  echo "No checkpoint-* directories found under: ${CHECKPOINT_ROOT}"
  exit 1
fi

echo "Checkpoint root : ${CHECKPOINT_ROOT}"
echo "Sweep result    : ${SWEEP_RESULT_DIR}"
echo "Shard root      : ${SHARD_ROOT}"
echo "Checkpoint count: ${#CHECKPOINT_NAMES[@]}"
echo "Eval batch size : ${EVAL_BATCH_SIZE}"
echo "Eval partition  : ${EVAL_PARTITION}"
echo "Merge partition : ${MERGE_PARTITION}"
echo

printf "job_id\tcheckpoint\tckpt_dir\tshard_dir\n" > "${SUBMIT_MANIFEST}"

job_ids=()
for ckpt_name in "${CHECKPOINT_NAMES[@]}"; do
  ckpt_dir="${CHECKPOINT_ROOT}/${ckpt_name}"
  shard_dir="${SHARD_ROOT}/${ckpt_name}"
  submit_output="$(
    sbatch --parsable \
      -p "${EVAL_PARTITION}" \
      -J "geo_${ckpt_name#checkpoint-}" \
      --export=ALL,CODI_EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
      "${SCRIPT_DIR}/slurm_debug_eval_llama1b_geodesic_checkpoint.sh" \
      "${ckpt_dir}" \
      "${shard_dir}"
  )"
  job_id="${submit_output%%;*}"
  job_ids+=("${job_id}")
  printf "%s\t%s\t%s\t%s\n" "${job_id}" "${ckpt_name}" "${ckpt_dir}" "${shard_dir}" >> "${SUBMIT_MANIFEST}"
  echo "submitted eval  : ${job_id} ${ckpt_name}"
done

dependency_chain="$(IFS=:; echo "${job_ids[*]}")"
merge_submit_output="$(
  sbatch --parsable \
    -p "${MERGE_PARTITION}" \
    -J "geo_merge" \
    --dependency="afterany:${dependency_chain}" \
    "${SCRIPT_DIR}/slurm_debug_merge_llama1b_geodesic_sweep.sh" \
    "${SHARD_ROOT}" \
    "${SWEEP_RESULT_DIR}" \
    "${EXPECTED_LIST}"
)"
merge_job_id="${merge_submit_output%%;*}"

echo
echo "submitted merge : ${merge_job_id} afterany:${dependency_chain}"
echo "manifest        : ${SUBMIT_MANIFEST}"
echo "expected list   : ${EXPECTED_LIST}"
echo "final result    : ${SWEEP_RESULT_DIR}"
