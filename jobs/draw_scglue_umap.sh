#!/bin/bash
#SBATCH --output logs/consistent_umap_%A_%a.out
#SBATCH --error logs/consistent_umap_%A_%a.err
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array=0-3
#SBATCH --job-name consistent_umap
#SBATCH --mem=48g
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00

DATASETS=("tea_seq_ATAC")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

python -u draw_consistent_umaps_from_scalemap.py \
  --dataset "${DATASET}" \
  --result_root /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script/results