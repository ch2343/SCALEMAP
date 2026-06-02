#!/bin/bash
#SBATCH --output logs/integration_%x_%A_%a.out
#SBATCH --error logs/integration_%x_%A_%a.err
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --mem=30g
#SBATCH --cpus-per-task=30
#SBATCH --time=00:20:00

DATASETS=("bmcite" "D22" "D23" "GSE164378" "COMBAT_subset_005" "tea_seq")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

METHOD=${METHOD:?METHOD not set}
VERSION=${VERSION:?VERSION not set}
OUTPUT_ROOT=${OUTPUT_ROOT:-./results}
SEED=${SEED:-10}

python -u script_1_integration.py \
  --dataset "${DATASET}" \
  --modalities RNA_ADT \
  --method "${METHOD}" \
  --version "${VERSION}" \
  --output_root "${OUTPUT_ROOT}" \
  --seed "${SEED}"