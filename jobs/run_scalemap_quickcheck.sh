#!/bin/bash
#SBATCH --output logs/scalemap-%A_%a.out
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array 0-5
#SBATCH --job-name scalemap_quickcheck
#SBATCH --error logs/scalemap-%A_%a.err
#SBATCH --mem 64g
#SBATCH --cpus-per-task 30
#SBATCH --time 8:00:00

DATASETS=("bmcite" "D22" "D23" "GSE164378" "COMBAT_subset_005" "tea_seq")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

cd /gpfs/gibbs/project/guan_leying/ch2343/multi-omics/benchmark_script

python -u script_1_integration.py \
  --dataset "${DATASET}" \
  --modalities RNA_ADT \
  --method scalemap \
  --version quickcheck_scalemap_v1 \
  --output_root ./results