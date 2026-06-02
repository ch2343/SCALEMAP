#!/bin/bash
set -euo pipefail

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

OUTPUT_ROOT="./results"
mkdir -p logs jobs

METHODS=("scalemap" "scmodal" "scglue" "maxfuse" "bindsc")

for METHOD in "${METHODS[@]}"; do
  echo "Submitting evaluation+combine pipeline for method=${METHOD}"

  if [[ "${METHOD}" == "scalemap" ]]; then
    VERSION_BASE="benchmark_v3"
    SEEDS=(10 25 35 45 42)
  else
    VERSION_BASE="benchmark_v2"
    SEEDS=(10 20 42 52 62)
  fi

  EVAL_JOBIDS=()

  for SEED in "${SEEDS[@]}"; do
    VERSION="${VERSION_BASE}_seed${SEED}"

    echo "  -> seed=${SEED}, version=${VERSION}"

    EVAL_JOBID=$(sbatch \
      --parsable \
      --job-name="eval_${METHOD}_s${SEED}" \
      --export=ALL,METHOD="${METHOD}",VERSION="${VERSION}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
      jobs/run_evaluation_method.sh)

    echo "     Evaluation array job: ${EVAL_JOBID}"
    EVAL_JOBIDS+=("${EVAL_JOBID}")
  done

  DEP_STR=$(IFS=:; echo "${EVAL_JOBIDS[*]}")
  SEED_STR="${SEEDS[*]}"

  COMBINE_JOBID=$(sbatch \
    --parsable \
    --dependency=afterok:${DEP_STR} \
    --partition=scavenge \
    --requeue \
    --job-name="combine_${METHOD}" \
    --output="logs/combine_${METHOD}_%j.out" \
    --error="logs/combine_${METHOD}_%j.err" \
    --wrap="cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script && python -u combine_method_metrics_across_seeds.py ${METHOD} ${VERSION_BASE} ${OUTPUT_ROOT} --seeds ${SEED_STR}")

  echo "  Combine job: ${COMBINE_JOBID}"
done

echo "All evaluation + combine jobs submitted."