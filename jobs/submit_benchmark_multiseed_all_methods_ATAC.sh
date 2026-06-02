#!/bin/bash
set -euo pipefail

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

VERSION_BASE="benchmark_v3"
OUTPUT_ROOT="./results"

#METHODS=("scalemap" "scmodal" "scglue" "bindsc" "maxfuse")
METHODS=("scalemap")
SEEDS=(100 200)

# choose ONE seed for UMAP
UMAP_SEED=20

mkdir -p logs jobs

for METHOD in "${METHODS[@]}"; do
  echo "Submitting multi-seed benchmark for method=${METHOD}"

  EVAL_JOBIDS=()

  for SEED in "${SEEDS[@]}"; do
    VERSION="${VERSION_BASE}_seed${SEED}"

    echo "  -> seed=${SEED}, version=${VERSION}"

    INT_JOBID=$(sbatch \
      --parsable \
      --job-name="int_${METHOD}_s${SEED}" \
      --export=ALL,METHOD="${METHOD}",VERSION="${VERSION}",OUTPUT_ROOT="${OUTPUT_ROOT}",SEED="${SEED}" \
      jobs/run_integration_method_ATAC.sh)

    echo "     Integration job: ${INT_JOBID}"

    EVAL_JOBID=$(sbatch \
      --parsable \
      --dependency=afterok:${INT_JOBID} \
      --job-name="eval_${METHOD}_s${SEED}" \
      --export=ALL,METHOD="${METHOD}",VERSION="${VERSION}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
      jobs/run_evaluation_method_ATAC.sh)

    echo "     Evaluation job: ${EVAL_JOBID}"
    EVAL_JOBIDS+=("${EVAL_JOBID}")

    if [[ "${SEED}" == "${UMAP_SEED}" ]]; then
      UMAP_JOBID=$(sbatch \
        --parsable \
        --dependency=afterok:${INT_JOBID} \
        --job-name="umap_${METHOD}_s${SEED}" \
        --export=ALL,METHOD="${METHOD}",VERSION="${VERSION}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
        jobs/run_umap_method_ATAC.sh)

      echo "     UMAP job: ${UMAP_JOBID}"
    fi
  done

  DEP_STR=$(IFS=:; echo "${EVAL_JOBIDS[*]}")

  COMBINE_JOBID=$(sbatch \
    --parsable \
    --dependency=afterok:${DEP_STR} \
    --partition=scavenge \
    --requeue \
    --job-name="combine_${METHOD}" \
    --output="logs/combine_${METHOD}_%j.out" \
    --error="logs/combine_${METHOD}_%j.err" \
    --wrap="cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script && python -u combine_method_metrics_across_seeds_ATAC.py ${METHOD} ${VERSION_BASE} ${OUTPUT_ROOT} --seeds 10 25 20 100 200")

  echo "  Cross-seed combine job: ${COMBINE_JOBID}"
done

echo "All multi-seed benchmark pipelines submitted."