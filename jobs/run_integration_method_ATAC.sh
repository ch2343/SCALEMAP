#!/bin/bash
#SBATCH --output logs/atac_integrate-%A_%a.out
#SBATCH --error logs/atac_integrate-%A_%a.err
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array 0-2
#SBATCH --job-name atac_integrate
#SBATCH --mem 30g
#SBATCH --cpus-per-task 30
#SBATCH --time 1:00:00

DATASETS=("D22_ATAC" "D23_ATAC" "tea_seq_ATAC")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

METHOD=${METHOD:?METHOD not set}
VERSION=${VERSION:?VERSION not set}
OUTPUT_ROOT=${OUTPUT_ROOT:-./results}
SEED=${SEED:?SEED not set}

EXTRA_ARGS=()

if [[ "${METHOD}" == "scalemap" ]]; then
  EXTRA_ARGS+=(--preprocess_mode rna_atac)
elif [[ "${METHOD}" == "scmodal" ]]; then
  EXTRA_ARGS+=(--preprocess_mode rna_atac)
elif [[ "${METHOD}" == "scglue" ]]; then
  EXTRA_ARGS+=(--preprocess_mode rna_atac --n_pca_rna 100 --n_pca_mod2 100 --rna_prob_model NB --mod2_prob_model Normal)
elif [[ "${METHOD}" == "bindsc" ]]; then
  EXTRA_ARGS+=(--preprocess_mode rna_atac)
elif [[ "${METHOD}" == "maxfuse" ]]; then
  EXTRA_ARGS+=(--preprocess_mode rna_atac --active_pcs_rna 30 --active_pcs_mod2 30)
else
  echo "Unknown method: ${METHOD}"
  exit 1
fi

echo "[INFO] METHOD=${METHOD}"
echo "[INFO] DATASET=${DATASET}"
echo "[INFO] VERSION=${VERSION}"
echo "[INFO] SEED=${SEED}"

python -u script_1_integration.py \
  --dataset "${DATASET}" \
  --modalities RNA_ATAC \
  --method "${METHOD}" \
  --version "${VERSION}" \
  --output_root "${OUTPUT_ROOT}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"