#!/bin/bash
set -euo pipefail

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

INT_JOBID=$(sbatch --parsable jobs/run_scalemap_quickcheck_ATAC.sh)
echo "Integration job: ${INT_JOBID}"

EVAL_JOBID=$(sbatch --parsable --dependency=afterok:${INT_JOBID} jobs/run_scalemap_eval_ATAC.sh)
echo "Evaluation job: ${EVAL_JOBID}"

UMAP_JOBID=$(sbatch --parsable --dependency=afterok:${INT_JOBID} jobs/run_scalemap_umap_ATAC.sh)
echo "UMAP job: ${UMAP_JOBID}"

COMBINE_JOBID=$(sbatch --parsable --dependency=afterok:${EVAL_JOBID} \
  --job-name scalemap_atac_v2_combine \
  --output logs/scalemap_atac_v2_combine-%j.out \
  --error logs/scalemap_atac_v2_combine-%j.err \
  --partition=scavenge \
  --requeue \
  --mem=8g \
  --cpus-per-task=1 \
  --time=00:30:00 \
  --wrap="cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script && \
python combine_method_metrics_ATAC.py scalemap quickcheck_scalemap_atac_v2 ./results")
echo "Combine job: ${COMBINE_JOBID}"

echo "All RNA-ATAC SCALEMAP v2 jobs submitted."