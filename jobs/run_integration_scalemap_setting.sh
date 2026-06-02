#!/bin/bash
#SBATCH --output logs/integration_scalemap_%A_%a.out
#SBATCH --error logs/integration_scalemap_%A_%a.err
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --mem=30g
#SBATCH --cpus-per-task=10
#SBATCH --time=00:20:00

DATASETS=("bmcite" "D22" "D23" "GSE164378" "COMBAT_subset_005" "tea_seq")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

cd /gpfs/gibbs/project/guan_leying/ch2343/multi-omics/benchmark_script

VERSION=${VERSION:?VERSION not set}
OUTPUT_ROOT=${OUTPUT_ROOT:-./results}
EXTRA_ARGS=${EXTRA_ARGS:-}

python -u script_1_integration.py \
  --dataset "${DATASET}" \
  --modalities RNA_ADT \
  --method scalemap \
  --version "${VERSION}" \
  --output_root "${OUTPUT_ROOT}" \
  ${EXTRA_ARGS}