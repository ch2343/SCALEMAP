#!/bin/bash
#SBATCH --output logs/maxfuse_eval-%A_%a.out
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array 0-5
#SBATCH --job-name maxfuse_eval
#SBATCH --error logs/maxfuse_eval-%A_%a.err
#SBATCH --mem 32g
#SBATCH --cpus-per-task 30
#SBATCH --time 4:00:00

DATASETS=("bmcite" "D22" "D23" "GSE164378" "COMBAT_subset_005" "tea_seq")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

cd /gpfs/gibbs/project/guan_leying/ch2343/multi-omics/benchmark_script

python -u script_2_evaluation.py \
  --dataset "${DATASET}" \
  --modalities RNA_ADT \
  --method maxfuse \
  --version quickcheck_maxfuse_v1 \
  --output_root ./results \
  --batch_key modality \
  --celltype_key celltype \
  --force_shared_obs_names