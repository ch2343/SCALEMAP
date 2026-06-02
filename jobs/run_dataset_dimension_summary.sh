#!/bin/bash
#SBATCH --job-name dataset_dim_summary
#SBATCH --output logs/dataset_dim_summary-%j.out
#SBATCH --error logs/dataset_dim_summary-%j.err
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

set -euo pipefail

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

mkdir -p logs results

python -u summarize_dataset_shapes.py \
  --output_csv ./results/dataset_dimension_summary.csv