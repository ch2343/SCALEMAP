#!/bin/bash
set -euo pipefail

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

VERSION="benchmark_v2"
OUTPUT_ROOT="./results"

#METHODS=("scalemap" "maxfuse")
METHODS=("scalemap")

mkdir -p logs jobs

for METHOD in "${METHODS[@]}"; do
  echo "Submitting pipeline for method=${METHOD}"

  INT_JOBID=$(sbatch \
    --parsable \
    --job-name="int_${METHOD}" \
    --export=ALL,METHOD="${METHOD}",VERSION="${VERSION}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
    jobs/run_integration_method.sh)

  echo "  Integration job: ${INT_JOBID}"

  EVAL_JOBID=$(sbatch \
    --parsable \
    --dependency=afterok:${INT_JOBID} \
    --job-name="eval_${METHOD}" \
    --export=ALL,METHOD="${METHOD}",VERSION="${VERSION}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
    jobs/run_evaluation_method.sh)

  echo "  Evaluation job: ${EVAL_JOBID}"

  UMAP_JOBID=$(sbatch \
    --parsable \
    --dependency=afterok:${INT_JOBID} \
    --job-name="umap_${METHOD}" \
    --export=ALL,METHOD="${METHOD}",VERSION="${VERSION}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
    jobs/run_umap_method.sh)

  echo "  UMAP job: ${UMAP_JOBID}"

  COMBINE_JOBID=$(sbatch \
    --parsable \
    --dependency=afterok:${EVAL_JOBID} \
    --partition=scavenge \
    --requeue \
    --job-name="combine_${METHOD}" \
    --output="logs/combine_${METHOD}_%j.out" \
    --error="logs/combine_${METHOD}_%j.err" \
    --wrap="cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script && python -u combine_method_metrics.py ${METHOD} ${VERSION} ${OUTPUT_ROOT}")

  echo "  Combine job: ${COMBINE_JOBID}"
done

echo "All benchmark pipelines submitted."