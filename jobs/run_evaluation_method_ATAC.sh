#!/bin/bash
#SBATCH --output logs/atac_eval-%A_%a.out
#SBATCH --error logs/atac_eval-%A_%a.err
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array 0-2
#SBATCH --job-name atac_eval
#SBATCH --mem 32g
#SBATCH --cpus-per-task 10
#SBATCH --time 1:00:00

DATASETS=("D22_ATAC" "D23_ATAC" "tea_seq_ATAC")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

METHOD=${METHOD:?METHOD not set}
VERSION=${VERSION:?VERSION not set}
OUTPUT_ROOT=${OUTPUT_ROOT:-./results}

echo "[INFO] METHOD=${METHOD}"
echo "[INFO] DATASET=${DATASET}"
echo "[INFO] VERSION=${VERSION}"

python -u script_2_evaluation.py \
  --dataset "${DATASET}" \
  --modalities RNA_ATAC \
  --method "${METHOD}" \
  --version "${VERSION}" \
  --output_root "${OUTPUT_ROOT}" \
  --batch_key modality \
  --celltype_key celltype \
  --pair_modalities RNA ATAC \
  --force_shared_obs_names