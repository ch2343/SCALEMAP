#!/bin/bash
#SBATCH --output logs/bindsc-%A_%a.out
#SBATCH --error logs/bindsc-%A_%a.err
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --mem=64g
#SBATCH --cpus-per-task=10
#SBATCH --time=1:00:00

DATASETS=("bmcite" "D22" "D23" "GSE164378" "COMBAT_subset_005" "tea_seq")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

python -u script_1_integration.py \
  --dataset "${DATASET}" \
  --modalities RNA_ADT \
  --method bindsc \
  --version quickcheck_bindsc_v1 \
  --output_root ./results