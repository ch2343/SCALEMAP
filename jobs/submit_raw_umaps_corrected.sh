#!/bin/bash
set -euo pipefail

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

# 1) array job for all non-COMBAT_full datasets (30 min each)
SMALL_JOBID=$(sbatch --parsable \
  --job-name raw_umap_small \
  --output logs/raw_umap_small-%A_%a.out \
  --error logs/raw_umap_small-%A_%a.err \
  --partition=scavenge \
  --requeue \
  --array 0-8 \
  --mem 32g \
  --cpus-per-task 8 \
  --time 00:30:00 \
  --wrap='
DATASETS=("bmcite" "D22" "D23" "GSE164378" "COMBAT_subset_005" "tea_seq" "D22_ATAC" "D23_ATAC" "tea_seq_ATAC")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script
python -u generate_raw_umaps_corrected.py --dataset "$DATASET"
')
echo "Small-dataset raw UMAP job: ${SMALL_JOBID}"

# 2) separate job for COMBAT_full (4 hours)
COMBAT_JOBID=$(sbatch --parsable \
  --job-name raw_umap_combat \
  --output logs/raw_umap_combat-%j.out \
  --error logs/raw_umap_combat-%j.err \
  --partition=scavenge \
  --requeue \
  --mem 256g \
  --cpus-per-task 8 \
  --time 04:00:00 \
  --wrap='
cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script
python -u generate_raw_umaps_corrected.py --dataset COMBAT_full
')
echo "COMBAT_full raw UMAP job: ${COMBAT_JOBID}"

echo "All raw UMAP jobs submitted."