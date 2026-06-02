#!/bin/bash
#SBATCH --output=logs/eval_%x_%A_%a.out
#SBATCH --error=logs/eval_%x_%A_%a.err
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --mem=32g
#SBATCH --cpus-per-task=30
#SBATCH --time=1:00:00

set -euo pipefail

DATASETS=("bmcite" "D22" "D23" "GSE164378" "COMBAT_subset_005" "tea_seq")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

METHOD=${METHOD:?METHOD not set}
VERSION=${VERSION:?VERSION not set}
OUTPUT_ROOT=${OUTPUT_ROOT:-./results}

python -u script_2_evaluation.py \
  --dataset "${DATASET}" \
  --modalities RNA_ADT \
  --method "${METHOD}" \
  --version "${VERSION}" \
  --output_root "${OUTPUT_ROOT}" \
  --batch_key modality \
  --celltype_key celltype \
  --force_shared_obs_names \
  --pair_modalities RNA ADT