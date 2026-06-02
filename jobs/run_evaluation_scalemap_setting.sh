#!/bin/bash
#SBATCH --output logs/eval_scalemap_%A_%a.out
#SBATCH --error logs/eval_scalemap_%A_%a.err
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --mem=64g
#SBATCH --cpus-per-task=10
#SBATCH --time=00:40:00

DATASETS=("bmcite" "D22" "D23" "GSE164378" "COMBAT_subset_005" "tea_seq")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

cd /gpfs/gibbs/project/guan_leying/ch2343/multi-omics/benchmark_script

VERSION=${VERSION:?VERSION not set}
OUTPUT_ROOT=${OUTPUT_ROOT:-./results}

python -u script_2_evaluation.py \
  --dataset "${DATASET}" \
  --modalities RNA_ADT \
  --method scalemap \
  --version "${VERSION}" \
  --output_root "${OUTPUT_ROOT}" \
  --batch_key modality \
  --celltype_key celltype \
  --pair_modalities RNA ADT \
  --force_shared_obs_names